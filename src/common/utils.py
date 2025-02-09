from typing import Tuple
from typing import Union
from shapely.geometry import Polygon
import numpy as np


Bounds = Tuple[float, float, float, float]


def get_utm_zone_epsg(aoi: Union[Bounds, Polygon]):
    if isinstance(aoi, Polygon):
        minx, miny, maxx, maxy = aoi.bounds
    else:
        minx, miny, maxx, maxy = aoi

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
