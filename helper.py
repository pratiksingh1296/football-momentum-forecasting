# =============================================================================
# Imports
# =============================================================================

import json
import pandas as pd
import matplotlib.pyplot as plt
from config import *
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

# =============================================================================
# Functions
# =============================================================================

def load_report(filename):
    """Load a classification report saved with json.dump()."""
    with open(filename, "r") as f:
        return json.load(f)


def compare_classification_reports(
    report_a,
    report_b,
    labels=("Home Dominant", "Away Dominant", "Balanced"),
    names=("Experiment A", "Experiment B"),
):
    """Compare two sklearn classification_report dictionaries."""

    metrics = {
        "Accuracy": lambda r: r["accuracy"],
        "Macro Precision": lambda r: r["macro avg"]["precision"],
        "Macro Recall": lambda r: r["macro avg"]["recall"],
        "Macro F1": lambda r: r["macro avg"]["f1-score"],
        "Weighted F1": lambda r: r["weighted avg"]["f1-score"],
    }

    for label in labels:
        metrics[f"{label} F1"] = lambda r, label=label: r[label]["f1-score"]

    df = pd.DataFrame({
        "Metric": metrics.keys(),
        names[0]: [fn(report_a) for fn in metrics.values()],
        names[1]: [fn(report_b) for fn in metrics.values()],
    })

    df["Δ"] = df[names[1]] - df[names[0]]

    return df


def plot_model_comparison(
    model1_history,
    model1_name,
    model2_history,
    model2_name,
    save_dir=None,
    filename=None,
    loss_ylim=None,
    f1_ylim=None,
    accuracy_ylim=None,
):
    """Plot training and validation metrics for two model training runs."""
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    fig.suptitle(
        f"Training Dynamics: {model1_name} vs. {model2_name}",
        fontsize=16,
        fontweight="bold",
    )

    metrics = [
        ("loss", "Loss"),
        ("f1", "Macro F1"),
        ("accuracy", "Accuracy"),
    ]

    colors = ["tab:blue", "tab:orange"]

    for ax, (metric, ylabel) in zip(axes, metrics):
        ax.plot(
            model1_history["epoch"],
            model1_history[f"train_{metric}"],
            "--o",
            linewidth=2,
            markersize=4,
            label=f"{model1_name} (Train)",
            color=colors[0],
        )

        ax.plot(
            model1_history["epoch"],
            model1_history[f"val_{metric}"],
            "-o",
            linewidth=2,
            markersize=4,
            label=f"{model1_name} (Validation)",
            color=colors[0],
        )

        ax.plot(
            model2_history["epoch"],
            model2_history[f"train_{metric}"],
            "--o",
            linewidth=2,
            markersize=4,
            label=f"{model2_name} (Train)",
            color=colors[1],
        )

        ax.plot(
            model2_history["epoch"],
            model2_history[f"val_{metric}"],
            "-o",
            linewidth=2,
            markersize=4,
            label=f"{model2_name} (Validation)",
            color=colors[1],
        )

        ax.set_title(f"Training vs Validation {ylabel}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    if loss_ylim:
        axes[0].set_ylim(loss_ylim)

    if f1_ylim:
        axes[1].set_ylim(f1_ylim)

    if accuracy_ylim:
        axes[2].set_ylim(accuracy_ylim)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.12),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_dir and filename:
        plt.savefig(save_dir / filename, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, axes



def plot_confusion_matrices(
    model1_true, 
    model1_preds,
    model2_true, 
    model2_preds, 
    display_labels=CLASS_NAMES,
    model1_name="Model 1",
    model2_name="Model 2",
    normalize="true",
    save_dir=None,
    filename=None):

    """ Plot normalized confusion matrices for two models. """

    model1_f1 = f1_score(model1_true, model1_preds, average="macro")
    model2_f1 = f1_score(model2_true, model2_preds, average="macro")

    fig, axes = plt.subplots(1, 2, figsize=(17,7))

    ConfusionMatrixDisplay.from_predictions(
        model1_true,
        model1_preds,
        display_labels=display_labels,
        cmap="Blues",
        normalize=normalize,
        values_format=".2f",
        colorbar=False,
        ax=axes[0]
    )


    axes[0].set_title(f"{model1_name}\nMacro F1 = {model1_f1:.4f}",fontsize=12,)

    ConfusionMatrixDisplay.from_predictions(
        model2_true,
        model2_preds,
        display_labels=display_labels,
        cmap="Blues",
        normalize=normalize,
        values_format=".2f",
        colorbar=True,
        ax=axes[1]
    )

    axes[1].set_title(f"{model2_name}\nMacro F1 = {model2_f1:.4f}",fontsize=12,)

    plt.tight_layout(pad=2)

    if save_dir is not None and filename is not None:
        plt.savefig(save_dir / filename, dpi=150, bbox_inches="tight")

    plt.show()