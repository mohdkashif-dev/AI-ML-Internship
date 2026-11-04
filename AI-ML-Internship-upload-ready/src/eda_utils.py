"""Reusable helpers for loading, cleaning and plotting data (Task 1)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")


def load_csv(path) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame."""
    return pd.read_csv(path)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return count and percentage of missing values per column."""
    out = pd.DataFrame({"missing": df.isnull().sum(),
                        "percent": (df.isnull().mean() * 100).round(2)})
    return out[out["missing"] > 0].sort_values("missing", ascending=False)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to snake_case (e.g. 'Math Score' -> 'math_score')."""
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))
    return df


def iqr_outliers(series: pd.Series) -> pd.Series:
    """Return values lying outside 1.5 * IQR of the series."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]


def save_fig(name: str, images_dir="../images") -> str:
    """Save the current matplotlib figure into the images folder."""
    path = Path(images_dir)
    path.mkdir(exist_ok=True)
    file = path / name
    plt.savefig(file, dpi=150, bbox_inches="tight")
    return str(file)
