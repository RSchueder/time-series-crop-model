import pytest
from click.testing import CliRunner

from src.cli import evaluate_performance


@pytest.mark.functional
def test_cli():
    runner = CliRunner()
    result = runner.invoke(
        evaluate_performance,
        [
            "-p",
            "/code/data/ml_2021-08-01_2022-12-31_u0c.tif",
            "-l",
            "/code/data/u0c_gt_filtered_2022.gpkg",
            "-o",
            "/code/output/",
            "-pc",
            "3",
            "-cc",
            "4",
            "-n",
            "10",
            "-u",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    # TODO: Check output statistics
