import logging
from typing import Any, Tuple, Union

import numpy as np
from shapely.geometry import Polygon

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


Bounds = Tuple[float, float, float, float]


def map_val_to_int(p: Any):
    try:
        return int(p)
    except (ValueError, TypeError):
        return -1


def get_utm_zone_epsg(aoi: Union[Bounds, Polygon]):
    if isinstance(aoi, Tuple):
        minx, miny, maxx, maxy = aoi
    else:
        minx, miny, maxx, maxy = aoi.bounds

    lon = np.mean([minx, maxx])
    lat = np.mean([miny, maxy])

    utm_band = str(int((np.floor((lon + 180) / 6) % 60) + 1))
    if len(utm_band) == 1:
        utm_band = "0" + utm_band
    if lat >= 0:
        epsg_code = "326" + utm_band
    else:
        epsg_code = "327" + utm_band

    return np.int64(epsg_code)
