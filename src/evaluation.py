import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio import features
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from shapely.geometry import box
from tqdm import tqdm

from src.common.utils import get_utm_zone_epsg, log
from src.constants import (
    CROP_TYPE_PREDICTION_CONFIDENCE_BAND,
    CROP_TYPE_PREDICTION_INDEX_BAND,
    LABELS_INDEX,
    MISSING_VALUE,
    TWO_WAY_LABELS_DICT,
)
from src.extract import extract_labels_and_indices
from src.metrics import calculate_crop_performance, calculate_statistics_per_field


def join_prediction_with_labels(
    prediction_path: Path,
    valid_labels: gpd.GeoDataFrame,
    pred_index_band: int = CROP_TYPE_PREDICTION_INDEX_BAND,
    confidence_index_band: int = CROP_TYPE_PREDICTION_CONFIDENCE_BAND,
    in_utm: bool = False,
):
    if in_utm:
        valid_labels.loc[:, ["utm_zone"]] = valid_labels["geometry"].apply(
            get_utm_zone_epsg
        )
        try:
            unique_utm_zones = valid_labels["utm_zone"].unique()
            assert len(unique_utm_zones) == 1
        except AssertionError:
            raise NotImplementedError(
                f"Workflow currently does not support operation on annotations that span more than one UTM zone. Inferred UTM zones were {unique_utm_zones}"
            )

        log.info(f"Conducting analysis in EPSG:{valid_labels['utm_zone'].iloc[0]}")
        valid_labels_utm = valid_labels.to_crs(
            f"EPSG:{valid_labels['utm_zone'].iloc[0]}"
        )

        with rasterio.open(prediction_path) as src:
            vrt_options = {
                "resampling": Resampling.nearest,
                "crs": valid_labels_utm.crs,
                "nodata": 255,
            }

            with WarpedVRT(src, **vrt_options) as vrt:
                predictions = vrt.read(pred_index_band)
                confidence = vrt.read(confidence_index_band)
                rasterized_labels, rasterized_field_indices = (
                    extract_labels_and_indices(vrt, valid_labels_utm)
                )
    else:
        with rasterio.open(prediction_path) as src:
            predictions = src.read(pred_index_band)
            confidence = src.read(confidence_index_band)
            rasterized_labels, rasterized_field_indices = extract_labels_and_indices(
                src, valid_labels
            )

    return predictions, confidence, rasterized_labels, rasterized_field_indices


def evaluate(
    prediction_path: Path,
    label_path: Path,
    output_path: Path,
    pred_index_band=CROP_TYPE_PREDICTION_INDEX_BAND,
    confidence_index_band=CROP_TYPE_PREDICTION_CONFIDENCE_BAND,
    in_utm=False,
):
    """
    Joins model predictions with ground truth labels and generates performance analysis outputs.

    Args:
        prediction_path (Path): Path to the raster file containing model predictions
        label_path (Path): Path to the vector file containing ground truth labels
        output_path (Path): Directory where output files will be saved
        pred_index_band (int, optional): Band index for predictions in the raster file
        confidence_index_band (int, optional): Band index for confidence scores in the raster file
        in_utm (bool, optional): Whether to perform analysis in UTM coordinates. Defaults to False.
            Note: Currently only supports data within a single UTM zone.

    Outputs:
        Saves multiple files to output_path:

    Raises:
        NotImplementedError: If in_utm=True and labels span multiple UTM zones

    Note:
        The function expects the prediction raster to have two bands:
        - A prediction band containing class indices
        - A confidence band containing prediction confidence scores
    """

    file_suffix = ""
    if in_utm:
        file_suffix = "_utm"

    log.info("Reading ground truth...")
    ground_truth = gpd.read_file(label_path)
    ground_truth["label"] = ground_truth["normalized_label"].apply(
        lambda x: TWO_WAY_LABELS_DICT.get(x, MISSING_VALUE)
    )
    valid_labels = ground_truth[ground_truth["label"] != MISSING_VALUE]
    discarded_labels = ground_truth[ground_truth["label"] == MISSING_VALUE]
    if len(discarded_labels.index) > 0:
        log.warning(
            f"Found {len(discarded_labels.index)} labels with an unknown category."
        )

    valid_labels.loc[:, ["label_index"]] = valid_labels["label"].apply(
        lambda x: LABELS_INDEX.index(x)
    )

    # reset the indices so we can join them to the rasterized dataframe later
    # each field will have a unique index from {0, fields}
    valid_labels = valid_labels.reset_index(drop=True)
    valid_labels = valid_labels.reset_index()
    valid_labels.rename(columns={"index": "field_index"}, inplace=True)

    log.info("Extracting model predictions...")
    predictions, confidence, rasterized_labels, rasterized_field_indices = (
        join_prediction_with_labels(
            prediction_path=prediction_path,
            valid_labels=valid_labels,
            pred_index_band=pred_index_band,
            confidence_index_band=confidence_index_band,
            in_utm=in_utm,
        )
    )

    pixel_result_df = pd.DataFrame(
        {
            "field_index": rasterized_field_indices.flatten(),
            "label_value": rasterized_labels.flatten(),
            "predictions": predictions.flatten(),
            "confidence": confidence.flatten(),
        }
    )
    # certain values like the confidence should be aggregated by mean across the field
    # others like the predicted class should be aggregated by mode

    log.info("Calculating field-wise statistics...")
    field_df_mean = pixel_result_df.groupby("field_index").mean()
    field_df_mode = pixel_result_df.groupby("field_index").agg(pd.Series.mode)
    field_df_count = pixel_result_df.groupby("field_index").count()

    field_df_mean.to_csv(output_path / f"field_df_mean{file_suffix}.csv")
    field_df_mode.to_csv(output_path / f"field_df_mode{file_suffix}.csv")
    field_df_count.to_csv(output_path / f"field_df_count{file_suffix}.csv")

    # join the aggregated predictions and confidence to the ground truth by associated field
    log.info("Doing final join...")
    main_result_df = (
        valid_labels.set_index("field_index")
        .join(field_df_mode[["label_value", "predictions"]])
        .join(field_df_mean[["confidence"]])
        .join(
            field_df_count[["label_value"]].rename(columns={"label_value": "n_pixels"})
        )
    )
    main_result_df.to_csv(output_path / f"main_df{file_suffix}.csv")

    # misclassifications on field basis
    log.info("Determining misclasifications...")
    calculate_statistics_per_field(main_result_df, output_path, file_suffix)

    # performance on crop basis
    log.info("Determining performance per crop...")
    calculate_crop_performance(
        predictions, confidence, rasterized_labels, output_path, file_suffix
    )
