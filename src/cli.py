import datetime
import logging
from logging import log
from pathlib import Path
from typing import Optional

import click

from src.common.utils import log
from src.constants import (
    CROP_TYPE_PREDICTION_CONFIDENCE_BAND,
    CROP_TYPE_PREDICTION_INDEX_BAND,
)
from src.evaluation import evaluate


@click.command()
@click.option(
    "--prediction-path",
    "-p",
    type=Path,
    required=True,
    help="path to geotiff of model predictions",
)
@click.option(
    "--label-path",
    "-l",
    type=Path,
    required=True,
    help="path to vector annotations of crop type labels",
)
@click.option(
    "--output-path",
    "-o",
    type=Path,
    required=True,
    help="path to produce outputs at",
)
@click.option(
    "--prediction-channel",
    "-pc",
    type=int,
    default=CROP_TYPE_PREDICTION_INDEX_BAND,
    required=True,
    help="GDAL band number of predictions in the prediction geotif",
)
@click.option(
    "--confidence-channel",
    "-cc",
    type=int,
    default=CROP_TYPE_PREDICTION_CONFIDENCE_BAND,
    required=True,
    help="GDAL band number of confidence in the prediction geotif",
)
@click.option(
    "--top-n",
    "-n",
    type=int,
    default=10,
    required=True,
    help="The number of low confidence fields that should be returned",
)
@click.option(
    "--in-utm",
    "-u",
    type=bool,
    is_flag=True,
    default=False,
    help="Do all work in UTM or not.",
)
def evaluate_performance(
    prediction_path: Path,
    label_path: Path,
    output_path: Path,
    prediction_channel: int = CROP_TYPE_PREDICTION_INDEX_BAND,
    confidence_channel: int = CROP_TYPE_PREDICTION_CONFIDENCE_BAND,
    top_n: int = 10,
    in_utm: bool = False,
) -> None:
    st = datetime.datetime.now()

    log.info("Running evaluation...")
    evaluate(
        prediction_path,
        label_path,
        output_path,
        prediction_channel,
        confidence_channel,
        in_utm,
    )
    et = datetime.datetime.now()

    log.info(f"Finished evaluation in {(et - st).seconds} seconds.")


if __name__ == "__main__":
    evaluate_performance()
