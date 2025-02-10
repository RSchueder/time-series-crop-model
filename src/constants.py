LABELS_DICT = {
    "Grassland Cultivated": ["grassland_cultivated"],
    "Grassland Nature": ["grassland_nature"],
    "Clover": ["clover"],
    "Alfalfa": ["alfalfa"],
    "Ryegrass": ["ryegrass"],
    "Winter Barley": ["barley_winter"],
    "Spring Barley": ["barley_spring", "barley_summer"],
    "Winter Wheat": ["wheat_winter"],
    "Triticale": ["triticale_winter", "triticale_spring"],
    "Winter Rye": ["rye_winter"],
    "Spring Rye": ["rye_spring"],
    "Spring Wheat": ["wheat_spring"],
    "Rice": ["rice"],
    "Millet": ["millet"],
    "Sorghum": ["sorghum", "Sorghum"],
    "Spring Oats": ["oats_spring"],
    "Winter Oats": ["oats_winter"],
    "Sunflowers": ["sunflowers", "sunflower"],
    "Flax": ["flax"],
    "Canola": ["canola_spring", "canola_winter"],
    "Grain Corn": ["corn_grain"],
    "Silage Corn": ["corn_silage"],
    "Potatoes": ["potatoes"],
    "Sugarbeets": ["sugarbeets", "beets"],
    "Soybeans": ["soybeans"],
    "Peas": ["peas", "peas_winter"],
    "Beans": ["beans"],
    "Lentils": ["lentils"],
    "Fallow": ["fallow"],
    "Turnips": ["turnips", "turnip"],
    "Trees": ["trees", "orchard", "orchards", "bananas"],
    "Vineyard": ["vineyard"],
}

MISSING_VALUE = "MISSING_VALUE"

LABELS_INDEX = list(LABELS_DICT.keys())

TWO_WAY_LABELS_DICT = {}

for key, values in LABELS_DICT.items():
    for value in values:
        TWO_WAY_LABELS_DICT[key] = value
        TWO_WAY_LABELS_DICT[value] = key

CROP_TYPE_PREDICTION_INDEX_BAND = 3
CROP_TYPE_PREDICTION_CONFIDENCE_BAND = 4
