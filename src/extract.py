from typing import Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from scipy.stats import mode
from shapely.geometry import Polygon

from src.constants import (
    CROP_TYPE_PREDICTION_CONFIDENCE_BAND,
    CROP_TYPE_PREDICTION_INDEX_BAND,
)


def extract_labels_and_indices(src, labels: gpd.GeoDataFrame):
    shapes = [
        (labels["geometry"].loc[idx], labels["label_index"].loc[idx])
        for idx in labels.index
    ]
    rasterized_labels = rasterio.features.rasterize(
        shapes=shapes, out_shape=src.shape, transform=src.transform, fill=-1, nodata=-1
    )

    shapes = [
        (labels["geometry"].loc[idx], labels["field_index"].loc[idx])
        for idx in labels.index
    ]
    rasterized_field_indices = rasterio.features.rasterize(
        shapes=shapes, out_shape=src.shape, transform=src.transform, fill=-1, nodata=-1
    )
    return rasterized_labels, rasterized_field_indices


"""
# this multiprocessing routine to extract values over fields
#  takes longer than 5 minutes...

def extract_field(args: Tuple[str, Polygon]):
    '''
    Used to extract values within a polygon from a raster in multiprocessing
    '''
    raster_path, geometry = args
    with rasterio.open(raster_path) as src:
        chunk, transform = mask(src, [geometry], crop=True)
        modal_prediction = mode(chunk[CROP_TYPE_PREDICTION_INDEX_BAND - 1, :, :])
        mean_confidence = np.mean(chunk[CROP_TYPE_PREDICTION_CONFIDENCE_BAND, :, :])

    return modal_prediction, mean_confidence


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
