import datetime
from pathlib import Path
from typing import Optional

import click

from src.constants import (
    CROP_TYPE_PREDICTION_CONFIDENCE_BAND,
    CROP_TYPE_PREDICTION_INDEX_BAND,
)
from src.evaluation import evaluate
from src.utils import log


# Custom types/validators
def validate_path(ctx, param, value: Path) -> Path:
    """Validate path exists if it's an input path"""
    if param.name in ["prediction_path", "label_path"] and not value.exists():
        raise click.BadParameter(f"File does not exist: {value}")
    return value


def validate_band(ctx, param, value: int) -> int:
    """Validate band number is positive"""
    if value < 1:
        raise click.BadParameter("Band number must be positive")
    return value


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Crop type segmentation evaluation tools."""
    pass


@cli.command(name="evaluate", help="Evaluate model performance")
@click.option(
    "--prediction-path",
    "-p",
    type=click.Path(path_type=Path),
    callback=validate_path,
    required=True,
    help="Path to GeoTIFF containing model predictions",
)
@click.option(
    "--label-path",
    "-l",
    type=click.Path(path_type=Path),
    callback=validate_path,
    required=True,
    help="Path to vector file containing crop type labels",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(path_type=Path),
    callback=validate_path,
    required=True,
    help="Directory where outputs will be saved",
)
@click.option(
    "--prediction-band",
    "-pb",  # Renamed for clarity
    type=int,
    default=CROP_TYPE_PREDICTION_INDEX_BAND,
    callback=validate_band,
    help="Band number for predictions in the GeoTIFF [default: 3]",
    show_default=True,
)
@click.option(
    "--confidence-band",
    "-cb",  # Renamed for clarity
    type=int,
    default=CROP_TYPE_PREDICTION_CONFIDENCE_BAND,
    callback=validate_band,
    help="Band number for confidence values in the GeoTIFF [default: 4]",
    show_default=True,
)
@click.option(
    "--bottom-n",
    "-n",
    type=int,
    default=10,
    help="Number of lowest confidence fields to analyze",
    show_default=True,
)
@click.option(
    "--in-utm",
    "-u",
    is_flag=True,
    help="Perform analysis in UTM coordinates",
)
def evaluate_performance(
    prediction_path: Path,
    label_path: Path,
    output_path: Path,
    prediction_band: int,
    confidence_band: int,
    bottom_n: int,
    in_utm: bool,
) -> None:
    """
    Evaluate crop type segmentation model performance.

    Analyzes predictions against ground truth labels and generates
    various performance metrics and visualizations.
    """
    start_time = datetime.datetime.now()
    log.info("Starting evaluation...")

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    evaluate(
        prediction_path,
        label_path,
        output_path,
        prediction_band,
        confidence_band,
        bottom_n,
        in_utm,
    )

    duration = datetime.datetime.now() - start_time
    log.info(f"Finished evaluation in {duration.seconds} seconds")


if __name__ == "__main__":
    cli()
