import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "online_shoppers_intention.csv"
OUTPUT_DIR = ROOT / "outputs" / "generated"
FIGURE_DIR = ROOT / "reports" / "figures" / "generated" / "eda"
MAIN_FIGURE_DIR = ROOT / "reports" / "figures" / "main"
APPENDIX_EDA_FIGURE_DIR = ROOT / "reports" / "figures" / "appendix" / "eda"
MAIN_OUTPUT_DIR = ROOT / "outputs" / "main"
APPENDIX_EDA_OUTPUT_DIR = ROOT / "outputs" / "appendix" / "eda"


MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def save_basic_dataset_profile(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    profile = pd.DataFrame(
        {
            "rows": [len(df)],
            "columns": [df.shape[1]],
            "features": [df.shape[1] - 1],
            "duplicated_rows": [df.duplicated().sum()],
            "total_missing_values": [df.isna().sum().sum()],
            "target_positive_count": [df["Revenue"].sum()],
            "target_positive_rate": [df["Revenue"].mean()],
        }
    )
    profile.to_csv(OUTPUT_DIR / "eda_basic_profile.csv", index=False)

    schema = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_count": df.isna().sum(),
            "missing_rate": df.isna().mean(),
            "unique_values": df.nunique(),
            "example_value": df.apply(lambda column: column.dropna().iloc[0] if column.dropna().shape[0] else np.nan),
        }
    )
    schema.to_csv(OUTPUT_DIR / "eda_schema_missing_cardinality.csv")


def save_classic_statistical_tables(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include=["number"]).copy()
    categorical = df.select_dtypes(include=["object", "bool"]).copy()

    numeric.describe().T.to_csv(OUTPUT_DIR / "eda_numeric_descriptive_stats.csv")

    categorical_summary = pd.DataFrame(
        {
            "unique_values": categorical.nunique(),
            "most_frequent_value": categorical.mode(dropna=True).iloc[0],
            "most_frequent_count": categorical.apply(lambda column: column.value_counts(dropna=False).iloc[0]),
            "most_frequent_rate": categorical.apply(lambda column: column.value_counts(dropna=False).iloc[0] / len(column)),
        }
    )
    categorical_summary.to_csv(OUTPUT_DIR / "eda_categorical_summary.csv")

    target_encoded = df["Revenue"].astype(int)
    corr_with_target = (
        numeric.assign(Revenue=target_encoded)
        .corr(numeric_only=True)["Revenue"]
        .drop("Revenue")
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .rename("correlation_with_revenue")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    corr_with_target.to_csv(OUTPUT_DIR / "eda_numeric_correlation_with_revenue.csv", index=False)

    outlier_rows = []
    for column in numeric.columns:
        q1 = numeric[column].quantile(0.25)
        q3 = numeric[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((numeric[column] < lower) | (numeric[column] > upper)).sum()
        outlier_rows.append(
            {
                "feature": column,
                "q1": q1,
                "median": numeric[column].median(),
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": outliers,
                "outlier_rate": outliers / len(df),
            }
        )
    pd.DataFrame(outlier_rows).sort_values("outlier_rate", ascending=False).to_csv(
        OUTPUT_DIR / "eda_iqr_outliers.csv",
        index=False,
    )


def save_categorical_frequency_tables(df: pd.DataFrame) -> None:
    categorical_columns = [
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend",
        "Revenue",
    ]
    rows = []
    for column in categorical_columns:
        frequencies = df[column].value_counts(dropna=False)
        rates = df[column].value_counts(normalize=True, dropna=False)
        for value, count in frequencies.items():
            rows.append(
                {
                    "feature": column,
                    "value": value,
                    "count": count,
                    "rate": rates.loc[value],
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "eda_categorical_frequencies.csv", index=False)


def save_numeric_distribution_grid(df: pd.DataFrame) -> None:
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3 * n_rows))
    axes = axes.ravel()

    for index, column in enumerate(numeric_columns):
        axes[index].hist(df[column], bins=35, color="#3f7f93", alpha=0.85)
        axes[index].set_title(column, fontsize=10)
        axes[index].tick_params(axis="both", labelsize=8)

    for index in range(len(numeric_columns), len(axes)):
        axes[index].axis("off")

    fig.suptitle("Numerical feature distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "numeric_distributions.png", dpi=160)
    plt.close(fig)


def save_target_distribution(df: pd.DataFrame) -> None:
    counts = df["Revenue"].value_counts().sort_index()
    labels = ["No revenue", "Revenue"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, counts.values, color=["#6f7f8f", "#3f7f93"])
    ax.set_title("Target distribution")
    ax.set_ylabel("Number of sessions")
    ax.set_ylim(0, counts.max() * 1.15)

    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}\n{value / total:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "target_distribution.png", dpi=160)
    plt.close(fig)


def save_conversion_by_category(df: pd.DataFrame, column: str, filename: str, title: str, order=None) -> None:
    rates = df.groupby(column)["Revenue"].mean()
    counts = df.groupby(column)["Revenue"].size()

    if order is not None:
        rates = rates.reindex([item for item in order if item in rates.index])
        counts = counts.reindex(rates.index)
    else:
        rates = rates.sort_values(ascending=False)
        counts = counts.reindex(rates.index)

    labels = [str(item) for item in rates.index]
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.bar(labels, rates.values, color="#3f7f93")
    ax1.set_title(title)
    ax1.set_ylabel("Conversion rate")
    ax1.set_ylim(0, max(rates.max() * 1.25, 0.05))
    ax1.tick_params(axis="x", rotation=35)

    ax2 = ax1.twinx()
    ax2.plot(labels, counts.values, color="#9a6f2d", marker="o", linewidth=2)
    ax2.set_ylabel("Number of sessions")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=160)
    plt.close(fig)


def save_boxplot_by_revenue(df: pd.DataFrame, column: str, filename: str, title: str, log_scale=False) -> None:
    groups = [
        df.loc[df["Revenue"] == False, column],
        df.loc[df["Revenue"] == True, column],
    ]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.boxplot(
        groups,
        labels=["No revenue", "Revenue"],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#d9e6ea", "color": "#385a64"},
        medianprops={"color": "#9a3f2d", "linewidth": 2},
    )
    ax.set_title(title)
    ax.set_ylabel(column)
    if log_scale:
        ax.set_yscale("symlog")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=160)
    plt.close(fig)


def save_rate_scatter(df: pd.DataFrame) -> None:
    sample = df.sample(n=min(4000, len(df)), random_state=42)
    colors = sample["Revenue"].map({False: "#6f7f8f", True: "#b85c38"})

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        sample["BounceRates"],
        sample["ExitRates"],
        c=colors,
        alpha=0.35,
        s=18,
        edgecolors="none",
    )
    ax.set_title("Bounce rate vs exit rate")
    ax.set_xlabel("BounceRates")
    ax.set_ylabel("ExitRates")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", label="No revenue", markerfacecolor="#6f7f8f", markersize=8),
            plt.Line2D([0], [0], marker="o", color="w", label="Revenue", markerfacecolor="#b85c38", markersize=8),
        ],
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "bounce_exit_scatter.png", dpi=160)
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include=["number", "bool"]).copy()
    numeric["Revenue"] = numeric["Revenue"].astype(int)
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("Correlation heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)


def save_eda_tables(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    month_rates = (
        df.groupby("Month")["Revenue"]
        .agg(conversion_rate="mean", sessions="size")
        .reindex([month for month in MONTH_ORDER if month in df["Month"].unique()])
    )
    month_rates.to_csv(OUTPUT_DIR / "eda_conversion_by_month.csv")

    visitor_rates = df.groupby("VisitorType")["Revenue"].agg(conversion_rate="mean", sessions="size")
    visitor_rates.to_csv(OUTPUT_DIR / "eda_conversion_by_visitor_type.csv")

    numeric_summary = df.groupby("Revenue")[
        ["PageValues", "BounceRates", "ExitRates", "ProductRelated", "ProductRelated_Duration"]
    ].median()
    numeric_summary.to_csv(OUTPUT_DIR / "eda_numeric_medians_by_revenue.csv")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_EDA_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["Revenue"] = df["Revenue"].astype(bool)

    save_basic_dataset_profile(df)
    save_classic_statistical_tables(df)
    save_categorical_frequency_tables(df)
    save_target_distribution(df)
    save_conversion_by_category(
        df,
        "Month",
        "conversion_by_month.png",
        "Conversion rate and session volume by month",
        order=MONTH_ORDER,
    )
    save_conversion_by_category(
        df,
        "VisitorType",
        "conversion_by_visitor_type.png",
        "Conversion rate and session volume by visitor type",
    )
    save_boxplot_by_revenue(
        df,
        "PageValues",
        "page_values_by_revenue.png",
        "Page values by revenue outcome",
        log_scale=True,
    )
    save_boxplot_by_revenue(
        df,
        "ProductRelated_Duration",
        "product_duration_by_revenue.png",
        "Product-related duration by revenue outcome",
        log_scale=True,
    )
    save_rate_scatter(df)
    save_correlation_heatmap(df)
    save_numeric_distribution_grid(df)
    save_eda_tables(df)

    MAIN_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_EDA_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _main_copies = {
        "target_distribution.png": "01_target_distribution.png",
        "conversion_by_month.png": "02_conversion_by_month.png",
        "page_values_by_revenue.png": "03_page_values_by_revenue.png",
    }
    for src_name, dst_name in _main_copies.items():
        src = FIGURE_DIR / src_name
        if src.exists():
            shutil.copy2(src, MAIN_FIGURE_DIR / dst_name)
    _appendix_copies = {
        "conversion_by_visitor_type.png": "A1_conversion_by_visitor_type.png",
        "product_duration_by_revenue.png": "A2_product_duration_by_revenue.png",
        "correlation_heatmap.png": "A3_correlation_heatmap.png",
        "bounce_exit_scatter.png": "A4_bounce_exit_scatter.png",
        "numeric_distributions.png": "A5_numeric_distributions.png",
    }
    for src_name, dst_name in _appendix_copies.items():
        src = FIGURE_DIR / src_name
        if src.exists():
            shutil.copy2(src, APPENDIX_EDA_FIGURE_DIR / dst_name)

    # ── Copy canonical CSV outputs ────────────────────────────────────────────
    _main_csv_copies = {
        "eda_basic_profile.csv": "01_dataset_profile.csv",
    }
    for src_name, dst_name in _main_csv_copies.items():
        src = OUTPUT_DIR / src_name
        if src.exists():
            shutil.copy2(src, MAIN_OUTPUT_DIR / dst_name)

    _appendix_eda_csv_copies = {
        "eda_basic_profile.csv": "A1_dataset_summary.csv",
        "eda_schema_missing_cardinality.csv": "A2_schema_missing_cardinality.csv",
        "eda_numeric_descriptive_stats.csv": "A3_numeric_descriptive_stats.csv",
        "eda_categorical_summary.csv": "A4_categorical_summary.csv",
        "eda_categorical_frequencies.csv": "A5_categorical_frequencies.csv",
        "eda_numeric_correlation_with_revenue.csv": "A6_numeric_correlation_with_revenue.csv",
        "eda_iqr_outliers.csv": "A7_iqr_outliers.csv",
        "eda_conversion_by_month.csv": "A8_conversion_by_month.csv",
        "eda_conversion_by_visitor_type.csv": "A9_conversion_by_visitor_type.csv",
        "eda_numeric_medians_by_revenue.csv": "A10_numeric_medians_by_revenue.csv",
    }
    for src_name, dst_name in _appendix_eda_csv_copies.items():
        src = OUTPUT_DIR / src_name
        if src.exists():
            shutil.copy2(src, APPENDIX_EDA_OUTPUT_DIR / dst_name)

    print(f"EDA figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
