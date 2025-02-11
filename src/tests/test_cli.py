from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from src.cli import evaluate_performance

OUTPUT_PATH = "/code/output/"


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
            OUTPUT_PATH,
            "-pb",
            "3",
            "-cb",
            "4",
            "-n",
            "10",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    
    df = pd.read_csv(Path(OUTPUT_PATH) / "fieldwise_result_per_class.csv")
    assert np.isclose(df["f1-score"].sum(), 12.6862, rtol=1e-4)
