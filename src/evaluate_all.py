import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = "../results"
METRICS_CSV = os.path.join(RESULTS_DIR, "loso_metrics.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "loso_summary.csv")

METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]


def load_metrics():
    if not os.path.exists(METRICS_CSV):
        raise FileNotFoundError(
            f"{METRICS_CSV} not found. Run train_all.py first."
        )
    return pd.read_csv(METRICS_CSV)


def summarize(df):
    agg = df.groupby("Model")[METRIC_COLS].agg(["mean", "std"])

    # Flatten multi-index columns: ("Accuracy", "mean") -> "Accuracy_mean"
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index()

    return agg


def print_summary(summary_df):
    print("\n" + "=" * 70)
    print("LOSO Cross-Validation Summary")
    print("=" * 70)

    for _, row in summary_df.iterrows():
        print(f"\nModel: {row['Model']}")
        for metric in METRIC_COLS:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            print(f"  {metric:<10}: {mean:.4f} ± {std:.4f}")


def plot_metrics(df):
    for metric in ["Accuracy", "F1", "ROC_AUC"]:
        fig, ax = plt.subplots(figsize=(8, 5))

        df.boxplot(column=metric, by="Model", ax=ax)

        ax.set_title(f"{metric} across LOSO folds by model")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        plt.suptitle("")

        out_path = os.path.join(RESULTS_DIR, f"{metric.lower()}_boxplot.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"Saved plot: {out_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_metrics()
    summary_df = summarize(df)

    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary to {SUMMARY_CSV}")

    print_summary(summary_df)
    plot_metrics(df)


if __name__ == "__main__":
    main()