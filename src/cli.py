import datetime
import logging
from logging import log
from pathlib import Path
from typing import Optional

import click

from src.constants import (
    CROP_TYPE_PREDICTION_CONFIDENCE_BAND,
    CROP_TYPE_PREDICTION_INDEX_BAND,
)
from src.evaluation import join_predictions_wih_labels

log = logging.getLogger(__name__)


@click.group()
def cli():
    """CLI tool for performance analysis."""
    pass


@cli.command("determine-poor-performance")
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
def determine_poor_performance(
    prediction_path: Path,
    label_path: Path,
    prediction_channel: Optional[int],
    confidence_channel: Optional[int],
    top_n: Optional[int],
) -> None:
    st = datetime.datetime.now()

    print("Running join_predictions_wih_labels")
    join_predictions_wih_labels(
        prediction_path, label_path, prediction_channel, confidence_channel
    )
    print("Finished join_predictions_wih_labels")
    et = datetime.datetime.now()

    print(f"Finished in {(et - st).seconds} seconds.")


if __name__ == "__main__":
    cli()
