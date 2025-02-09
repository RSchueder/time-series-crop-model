import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio import features
from rasterio.mask import mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from scipy.stats import mode
from shapely.geometry import Polygon, box
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from tqdm import tqdm

from src.common.utils import get_utm_zone_epsg
from src.constants import (
    CROP_TYPE_PREDICTION_CONFIDENCE_BAND,
    CROP_TYPE_PREDICTION_INDEX_BAND,
    LABELS_INDEX,
    MISSING_VALUE,
    TWO_WAY_LABELS_DICT,
)


def extract_field(args):
    raster_path, geometry = args
    with rasterio.open(raster_path) as src:
        chunk, transform = mask(src, [geometry], crop=True)
        modal_prediction = mode(chunk[CROP_TYPE_PREDICTION_INDEX_BAND - 1, :, :])
        mean_confidence = np.mean(chunk[CROP_TYPE_PREDICTION_CONFIDENCE_BAND, :, :])

    return modal_prediction, mean_confidence


def join_predictions_wih_labels(
    prediction_path: Path,
    label_path: Path,
    pred_index_band=CROP_TYPE_PREDICTION_INDEX_BAND,
    confidence_index_band=CROP_TYPE_PREDICTION_CONFIDENCE_BAND,
    in_utm=False,
):
    ground_truth = gpd.read_file(label_path)
    ground_truth["label"] = ground_truth["normalized_label"].apply(
        lambda x: TWO_WAY_LABELS_DICT.get(x, MISSING_VALUE)
    )
    valid_labels = ground_truth[ground_truth["label"] != MISSING_VALUE]
    # discarded_labels = ground_truth[ground_truth["label"] == MISSING_VALUE]

    valid_labels.loc[:, ["label_index"]] = valid_labels["label"].apply(
        lambda x: LABELS_INDEX.index(x)
    )

    valid_labels = valid_labels.reset_index(drop=True)
    valid_labels = valid_labels.reset_index()

    if in_utm:
        # Work in UTM zone, not working...
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
                reprojected_predictions = vrt.read(pred_index_band)
                reprojected_confidence = vrt.read(confidence_index_band)

    with rasterio.open(prediction_path) as src:
        predictions = src.read(3)
        confidence = src.read(4)

        # NOTE: Below should not be done in WGS84
        shapes = [
            (valid_labels["geometry"].loc[idx], valid_labels["label_index"].loc[idx])
            for idx in valid_labels.index
        ]
        rasterized_labels = rasterio.features.rasterize(
            shapes=shapes, out_shape=src.shape, transform=src.transform, nodata=255
        )

        shapes = [
            (valid_labels["geometry"].loc[idx], valid_labels["index"].loc[idx])
            for idx in valid_labels.index
        ]
        rasterized_field_indices = rasterio.features.rasterize(
            shapes=shapes, out_shape=src.shape, transform=src.transform, nodata=255
        )

    pixel_df = pd.DataFrame(
        {
            "field_index": rasterized_field_indices.flatten(),
            "label_value": rasterized_labels.flatten(),
            "predictions": predictions.flatten(),
            "confidence": confidence.flatten(),
        }
    )
    field_df_mean = pixel_df.groupby("field_index").mean()
    field_df_mode = pixel_df.groupby("field_index").agg(pd.Series.mode)
    field_df_mean.to_csv("/code/field_df_mean.csv")
    field_df_mode.to_csv("/code/field_df_mode.csv")

    confidence_per_class = [
        np.mean(confidence[predictions == idx]) for idx in range(len(LABELS_INDEX))
    ]
    precision, recall, _, _ = precision_recall_fscore_support(
        rasterized_labels.flatten(),
        predictions.flatten(),
        labels=[ii for ii in range(len(LABELS_INDEX))],
        average=None,
        zero_division=np.nan,
    )
    f1 = f1_score(
        rasterized_labels.flatten(),
        predictions.flatten(),
        labels=[ii for ii in range(len(LABELS_INDEX))],
        average=None,
        zero_division=np.nan,
    )

    df = {
        "class": LABELS_INDEX,
        "prediction_count": [
            np.sum(predictions.flatten() == idx) for idx in range(len(LABELS_INDEX))
        ],
        "label_count": [
            np.sum(rasterized_labels.flatten() == idx)
            for idx in range(len(LABELS_INDEX))
        ],
        "average_confidence": confidence_per_class,
        "f1-score": f1,
        "precision": precision,
        "recall": recall,
    }
    df = pd.DataFrame(df)
    df = df.sort_values("f1-score", ascending=False)
    df.to_csv("/code/result_per_class.csv")

    cf = confusion_matrix(
        rasterized_labels.flatten(),
        predictions.flatten(),
        labels=[ii for ii in range(len(LABELS_INDEX))],
        normalize="true",
    )
    plt.figure(figsize=(20, 20))

    plot = sns.heatmap(
        cf,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        cbar=True,
        xticklabels=np.arange(len(LABELS_INDEX)),
        yticklabels=np.arange(len(LABELS_INDEX)),
    )
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.savefig("/code/cf.png")

    """
    # this takes longer than 5 minutes...
    cpu_count = os.cpu_count()

    modes = list()
    confidence = list()
    args = [(prediction_path, geom) for geom in valid_labels.geometry]
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        future_to_arg = {executor.submit(extract_field, arg): arg for arg in args}
        # Process results as they complete
        for future in tqdm(as_completed(future_to_arg), total=len(args)):
            mode, conf = future.result()
            modes.append(mode)
            confidence.append(conf)
    """
