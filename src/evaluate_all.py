"""
evaluate_all.py

Analysis-only script for LOSO cross-validation results.

Does NOT train, does NOT load models, does NOT run predictions.
Reads results/loso_metrics.csv (written by train_all.py) and produces:
  - results/loso_summary.csv   (mean/std per model, rounded, sorted)
  - results/summary.md         (markdown table)
  - results/summary.tex        (LaTeX table)
  - boxplots per metric, saved as PNGs in results/
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------
# Constants
# --------------------------------------------------

RESULTS_DIR = os.path.join("..", "results")
METRICS_CSV = os.path.join(RESULTS_DIR, "loso_metrics.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "loso_summary.csv")
SUMMARY_MD = os.path.join(RESULTS_DIR, "summary.md")
SUMMARY_TEX = os.path.join(RESULTS_DIR, "summary.tex")

METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
ROUND_DECIMALS = 4


# --------------------------------------------------
# Loading
# --------------------------------------------------

def load_metrics():
    if not os.path.exists(METRICS_CSV):
        raise FileNotFoundError(
            f"{METRICS_CSV} not found. Run train_all.py first."
        )
    return pd.read_csv(METRICS_CSV)


# --------------------------------------------------
# Summary
# --------------------------------------------------

def build_summary(metrics_df):
    """Groupby Model, compute mean/std, round, sort alphabetically."""

    summary = metrics_df.groupby("Model")[METRIC_COLS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.round(ROUND_DECIMALS)
    summary = summary.sort_index()
    summary = summary.reset_index()

    return summary


def print_summary(summary_df):
    print("\n" + "=" * 70)
    print("LOSO Cross-Validation Summary")
    print("=" * 70)

    for _, row in summary_df.iterrows():
        print(f"\nModel: {row['Model']}")
        for metric in METRIC_COLS:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            print(f"  {metric:<10}: {mean:.4f} +/- {std:.4f}")


# --------------------------------------------------
# Exports
# --------------------------------------------------

def save_csv(summary_df):
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved summary CSV to {SUMMARY_CSV}")


def save_markdown(summary_df):
    with open(SUMMARY_MD, "w") as f:
        f.write(summary_df.to_markdown(index=False))
    print(f"Saved markdown summary to {SUMMARY_MD}")


def save_latex(summary_df):
    latex_str = summary_df.to_latex(index=False, float_format="%.4f")
    with open(SUMMARY_TEX, "w") as f:
        f.write(latex_str)
    print(f"Saved LaTeX summary to {SUMMARY_TEX}")


def save_boxplots(metrics_df):
    for metric in METRIC_COLS:
        fig, ax = plt.subplots(figsize=(8, 5))

        metrics_df.boxplot(column=metric, by="Model", ax=ax)

        ax.set_title(f"{metric} across LOSO folds by model")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        plt.suptitle("")

        out_path = os.path.join(RESULTS_DIR, f"{metric.lower()}_boxplot.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"Saved plot: {out_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    metrics_df = load_metrics()
    summary_df = build_summary(metrics_df)

    save_csv(summary_df)
    save_markdown(summary_df)
    save_latex(summary_df)

    print_summary(summary_df)

    save_boxplots(metrics_df)


if __name__ == "__main__":
    main()