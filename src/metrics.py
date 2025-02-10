from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from src.common.utils import map_val_to_int
from src.constants import LABELS_INDEX


def calculate_crop_performance(
    predictions: np.ndarray,
    confidence: np.ndarray,
    rasterized_labels: np.ndarray,
    output_path: Path,
    file_suffix: str,
):
    confidence_per_class = [
        np.mean(confidence[predictions == idx]) for idx in range(len(LABELS_INDEX))
    ]
    precision, recall, _, _ = precision_recall_fscore_support(
        rasterized_labels.flatten(),
        predictions.flatten(),
        labels=[ii for ii in range(len(LABELS_INDEX))],
        average=None,
        zero_division=np.nan,
    )
    f1 = f1_score(
        rasterized_labels.flatten(),
        predictions.flatten(),
        labels=[ii for ii in range(len(LABELS_INDEX))],
        average=None,
        zero_division=np.nan,
    )

    df = {
        "class": LABELS_INDEX,
        "prediction_count": [
            np.sum(predictions.flatten() == idx) for idx in range(len(LABELS_INDEX))
        ],
        "label_count": [
            np.sum(rasterized_labels.flatten() == idx)
            for idx in range(len(LABELS_INDEX))
        ],
        "average_confidence": confidence_per_class,
        "f1-score": f1,
        "precision": precision,
        "recall": recall,
    }
    df = pd.DataFrame(df)
    df = df.sort_values("f1-score", ascending=False)
    df.to_csv(output_path / f"result_per_class{file_suffix}.csv")

    cf = confusion_matrix(
        rasterized_labels.flatten(),
        predictions.flatten(),
        labels=[ii for ii in range(len(LABELS_INDEX))],
        normalize="true",
    )
    plt.figure(figsize=(20, 20))

    plot = sns.heatmap(
        cf,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        cbar=True,
        xticklabels=LABELS_INDEX,
        yticklabels=LABELS_INDEX,
    )
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path / f"pixelwise_confusion_matrix{file_suffix}.png")


def calculate_statistics_per_field(
    main_result_df: pd.DataFrame, output_path: Path, file_suffix: str
):
    main_result_df[
        ~(main_result_df["label_index"] == main_result_df["label_value"])
    ].to_csv(output_path / f"invalid_fields{file_suffix}.csv")

    main_result_df["predictions_clean"] = main_result_df["predictions"].apply(
        lambda x: map_val_to_int(x)
    )
    misclassified_fields = main_result_df[
        main_result_df["label_index"] != main_result_df["predictions_clean"]
    ]
    misclassified_field_counts = misclassified_fields.groupby(
        "normalized_label"
    ).count()

    misclassified_field_counts.reset_index().sort_values(
        "label_index", ascending=False
    ).plot.bar(x="normalized_label", y="label_index")
    plt.ylabel("number of misclassified fields")
    plt.tight_layout()
    plt.savefig(output_path / f"field_level_misclassifications{file_suffix}.png")

    field_count_by_label_crop = (
        main_result_df.groupby("label_index")
        .count()
        .sort_values("field_id", ascending=False)[["field_id"]]
        .rename(columns={"field_id": "field_count"})
    )

    field_count_by_predicted_crop = (
        main_result_df.groupby("predictions_clean")
        .count()
        .sort_values("field_id", ascending=False)[["field_id"]]
        .rename(columns={"field_id": "field_count"})
    )

    true_labels = list()
    for ll in range(len(LABELS_INDEX)):
        try:
            true_labels.append(
                f"{LABELS_INDEX[ll]} (n={field_count_by_label_crop.loc[ll].values[0]})"
            )
        except KeyError:
            true_labels.append(f"{LABELS_INDEX[ll]} (0)")

    pred_labels = list()
    for ll in range(len(LABELS_INDEX)):
        try:
            pred_labels.append(
                f"{LABELS_INDEX[ll]} (n={field_count_by_predicted_crop.loc[ll].values[0]})"
            )
        except KeyError:
            pred_labels.append(f"{LABELS_INDEX[ll]} (0)")

    cf = confusion_matrix(
        main_result_df["label_index"],
        main_result_df["predictions_clean"],
        labels=[ii for ii in range(len(LABELS_INDEX))],
        normalize="true",
    )
    plt.figure(figsize=(20, 20))

    plot = sns.heatmap(
        cf,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        cbar=True,
        xticklabels=LABELS_INDEX,
        yticklabels=true_labels,
    )
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path / f"fieldwise_confusion_matrix_norm_true{file_suffix}.png")

    cf = confusion_matrix(
        main_result_df["label_index"],
        main_result_df["predictions_clean"],
        labels=[ii for ii in range(len(LABELS_INDEX))],
        normalize="pred",
    )
    plt.figure(figsize=(20, 20))

    plot = sns.heatmap(
        cf,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        cbar=True,
        xticklabels=pred_labels,
        yticklabels=LABELS_INDEX,
    )
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path / f"fieldwise_confusion_matrix_norm_pred{file_suffix}.png")

    precision, recall, _, _ = precision_recall_fscore_support(
        main_result_df["label_index"],
        main_result_df["predictions_clean"],
        labels=[ii for ii in range(len(LABELS_INDEX))],
        average=None,
        zero_division=np.nan,
    )
    f1 = f1_score(
        main_result_df["label_index"],
        main_result_df["predictions_clean"],
        labels=[ii for ii in range(len(LABELS_INDEX))],
        average=None,
        zero_division=np.nan,
    )

    metrics_df = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1-score": f1}
    )
    metrics_df.index.name = "label_index"
    metrics_df = field_count_by_label_crop.join(metrics_df)

    metrics_df["label_index"] = metrics_df.index
    metrics_df["label"] = metrics_df["label_index"].apply(lambda x: LABELS_INDEX[x])

    metrics_df.to_csv(output_path / f"fieldwise_result_per_class{file_suffix}.csv")
