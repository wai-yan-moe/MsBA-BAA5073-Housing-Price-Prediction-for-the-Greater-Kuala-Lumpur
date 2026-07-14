"""
Exploratory Data Analysis visualisations for the GKL housing dataset.

Generates publication-quality figures saved to outputs/figures/.
"""

import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.config import (
    DATA_PROCESSED, FIGURES_DIR,
    PALETTE, CATEGORICAL_PALETTE, KLCC_LAT, KLCC_LON,
)

warnings.filterwarnings("ignore", category=FutureWarning)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": PALETTE["grid"],
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    })


def load_eda_data():
    return pd.read_csv(DATA_PROCESSED / "housing_eda.csv")


def plot_target_distribution(df):
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.hist(df["Price"] / 1e6, bins=60, color=PALETTE["primary"], edgecolor="white", alpha=0.85)
    ax.set_xlabel("Price (RM, millions)")
    ax.set_ylabel("Frequency")
    ax.set_title("Original Price Distribution")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    ax = axes[0, 1]
    stats.probplot(df["Price"], dist="norm", plot=ax)
    ax.get_lines()[0].set(color=PALETTE["primary"], markersize=2, alpha=0.4)
    ax.get_lines()[1].set(color=PALETTE["accent2"], linewidth=1.5)
    ax.set_title("Q-Q Plot (Original Price)")

    ax = axes[1, 0]
    ax.hist(df["Price_Log"], bins=60, color=PALETTE["secondary"], edgecolor="white", alpha=0.85)
    ax.set_xlabel("log(1 + Price)")
    ax.set_ylabel("Frequency")
    ax.set_title("Log-Transformed Price Distribution")

    ax = axes[1, 1]
    stats.probplot(df["Price_Log"], dist="norm", plot=ax)
    ax.get_lines()[0].set(color=PALETTE["secondary"], markersize=2, alpha=0.4)
    ax.get_lines()[1].set(color=PALETTE["accent2"], linewidth=1.5)
    ax.set_title("Q-Q Plot (Log Price)")

    skew_raw = df["Price"].skew()
    kurt_raw = df["Price"].kurtosis()
    skew_log = df["Price_Log"].skew()
    kurt_log = df["Price_Log"].kurtosis()
    fig.suptitle(
        f"Target Variable Distribution\n"
        f"Raw: skew={skew_raw:.2f}, kurtosis={kurt_raw:.2f}  |  "
        f"Log: skew={skew_log:.2f}, kurtosis={kurt_log:.2f}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "01_target_distribution.png")
    plt.close()
    print("Saved 01_target_distribution.png")


def plot_numerical_distributions(df):
    _style()
    structural = ["Built_up_sqft", "Land_Area_sqft", "Rooms", "Bathrooms", "Car_Parks"]
    distance = [c for c in df.columns if c.startswith("Dist_")]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, col in enumerate(structural):
        ax = axes.flat[i]
        ax.hist(df[col], bins=50, color=CATEGORICAL_PALETTE[i], edgecolor="white", alpha=0.85)
        skew = df[col].skew()
        ax.set_title(f"{col}\n(skew: {skew:.2f})")
        ax.set_ylabel("Frequency")
    axes.flat[-1].set_visible(False)
    fig.suptitle("Structural Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "02_structural_distributions.png")
    plt.close()

    n_dist = len(distance)
    ncols = 4
    nrows = (n_dist + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    for i, col in enumerate(distance):
        ax = axes.flat[i]
        ax.hist(df[col], bins=50, color=CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)],
                edgecolor="white", alpha=0.85)
        skew = df[col].skew()
        ax.set_title(f"{col.replace('Dist_', '')}\n(skew: {skew:.2f})")
        ax.set_ylabel("Frequency")
    for j in range(i + 1, nrows * ncols):
        axes.flat[j].set_visible(False)
    fig.suptitle("Distance Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "03_distance_distributions.png")
    plt.close()
    print("Saved 02_structural_distributions.png, 03_distance_distributions.png")


def plot_categorical_distributions(df):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    prop_counts = df["Property_Type_Simplified"].value_counts()
    bars = axes[0].barh(
        prop_counts.index[::-1], prop_counts.values[::-1],
        color=CATEGORICAL_PALETTE[:len(prop_counts)][::-1], edgecolor="white"
    )
    for bar, val in zip(bars, prop_counts.values[::-1]):
        axes[0].text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
                     f"{val:,}", va="center", fontsize=9, color=PALETTE["text"])
    axes[0].set_xlabel("Count")
    axes[0].set_title("Property Types")

    furn_counts = df["Furnishing_Cleaned"].value_counts()
    furn_colours = [PALETTE["primary"], PALETTE["accent1"], PALETTE["secondary"], PALETTE["grid"]]
    bars = axes[1].barh(
        furn_counts.index[::-1], furn_counts.values[::-1],
        color=furn_colours[::-1], edgecolor="white"
    )
    for bar, val in zip(bars, furn_counts.values[::-1]):
        axes[1].text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
                     f"{val:,}", va="center", fontsize=9, color=PALETTE["text"])
    axes[1].set_xlabel("Count")
    axes[1].set_title("Furnishing Status")

    fig.suptitle("Categorical Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "04_categorical_distributions.png")
    plt.close()
    print("Saved 04_categorical_distributions.png")


def plot_bivariate(df):
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.scatter(df["Built_up_sqft"], df["Price_Log"], alpha=0.08, s=4, color=PALETTE["primary"])
    ax.set_xlabel("Built-up Area (sq ft)")
    ax.set_ylabel("Price (log)")
    ax.set_title("Price vs Built-up Area")

    ax = axes[0, 1]
    ax.scatter(df["Dist_KLCC"], df["Price_Log"], alpha=0.08, s=4, color=PALETTE["secondary"])
    ax.set_xlabel("Distance to KLCC (km)")
    ax.set_ylabel("Price (log)")
    ax.set_title("Price vs Distance to KLCC")

    ax = axes[1, 0]
    order = df.groupby("Property_Type_Simplified")["Price_Log"].median().sort_values().index
    sns.boxplot(
        data=df, y="Property_Type_Simplified", x="Price_Log",
        order=order, ax=ax, palette=CATEGORICAL_PALETTE[:len(order)],
        fliersize=1, linewidth=0.8,
    )
    ax.set_ylabel("")
    ax.set_xlabel("Price (log)")
    ax.set_title("Price by Property Type")

    ax = axes[1, 1]
    order_f = df.groupby("Furnishing_Cleaned")["Price_Log"].median().sort_values().index
    sns.boxplot(
        data=df, y="Furnishing_Cleaned", x="Price_Log",
        order=order_f, ax=ax,
        palette=[PALETTE["primary"], PALETTE["accent1"], PALETTE["secondary"], PALETTE["grid"]],
        fliersize=1, linewidth=0.8,
    )
    ax.set_ylabel("")
    ax.set_xlabel("Price (log)")
    ax.set_title("Price by Furnishing Status")

    fig.suptitle("Bivariate Analysis", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "05_bivariate_analysis.png")
    plt.close()
    print("Saved 05_bivariate_analysis.png")


def plot_correlation_matrix(df):
    _style()
    num_cols = [
        "Price_Log", "Built_up_sqft", "Land_Area_sqft", "Rooms", "Bathrooms", "Car_Parks",
        "Dist_KLCC", "Dist_Mall", "Dist_Hospital", "Dist_School",
        "Dist_College", "Dist_University", "Dist_BusStation", "Dist_RailStation",
    ]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Correlation Matrix of Numerical Features", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "06_correlation_matrix.png")
    plt.close()
    print("Saved 06_correlation_matrix.png")


def plot_geospatial_map(df):
    _style()
    try:
        import contextily as ctx
    except ImportError:
        print("contextily not installed -- skipping geospatial map")
        return

    agg = df.groupby(["Latitude", "Longitude"]).agg(
        median_price=("Price_Log", "median"),
        count=("Price_Log", "size"),
    ).reset_index()

    size_scale = np.clip(agg["count"] / agg["count"].max() * 400, 25, 400)

    fig, ax = plt.subplots(figsize=(10, 12))

    scatter = ax.scatter(
        agg["Longitude"], agg["Latitude"],
        c=agg["median_price"], cmap="RdYlGn_r",
        s=size_scale, alpha=0.75, edgecolors="#333333", linewidths=0.5,
        zorder=5,
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Median Price (log scale)", fontsize=10)

    ax.plot(KLCC_LON, KLCC_LAT, marker="*", color="#1a1a2e",
            markersize=18, markeredgecolor="white", markeredgewidth=1.5, zorder=10)
    ax.annotate("KLCC", (KLCC_LON, KLCC_LAT), fontsize=10, fontweight="bold",
                xytext=(7, 7), textcoords="offset points", color="#1a1a2e",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    try:
        ctx.add_basemap(
            ax, crs="EPSG:4326",
            source=ctx.providers.OpenStreetMap.Mapnik,
            zoom=12, alpha=0.55,
        )
    except Exception:
        pass

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "Property Locations in Greater Kuala Lumpur\n"
        "(dot size = listing count, colour = median log price)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "07_geospatial_map.png")
    plt.close()
    print("Saved 07_geospatial_map.png")


def run_all_eda():
    df = load_eda_data()
    print(f"Loaded EDA dataset: {len(df):,} rows\n")
    plot_target_distribution(df)
    plot_numerical_distributions(df)
    plot_categorical_distributions(df)
    plot_bivariate(df)
    plot_correlation_matrix(df)
    plot_geospatial_map(df)
    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    run_all_eda()
