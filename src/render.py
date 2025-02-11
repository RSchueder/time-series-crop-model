from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def df_to_png(df: pd.DataFrame, output_path: Path, figsize=(10, 6)):
    """
    Save a pandas DataFrame as a PNG image.

    Args:
        df: pandas DataFrame
        output_path: path to save the PNG file
        figsize: tuple of (width, height) in inches
    """
    df = df.round(2)  # rounds all float columns to 2 decimal places

    # Create a figure and axis with no frames
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("tight")
    ax.axis("off")

    # Create the table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colColours=["#f2f2f2"] * len(df.columns),  # Light gray header
    )

    # Auto-adjust cell sizes
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Save as PNG
    plt.savefig(output_path, bbox_inches="tight", dpi=300, pad_inches=0.1)
    plt.close()
