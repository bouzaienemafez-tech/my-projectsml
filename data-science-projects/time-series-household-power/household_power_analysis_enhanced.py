from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
plt.style.use("default")

TARGET = "Global_active_power"
DEFAULT_MODEL_SPECS = [
    {"name": "ARIMA(0,1,1)", "order": (0, 1, 1)},
    {"name": "ARIMA(1,1,1)", "order": (1, 1, 1)},
    {"name": "SARIMA(0,1,1)(0,1,1,12)", "order": (0, 1, 1), "seasonal_order": (0, 1, 1, 12)},
    {"name": "SARIMA(1,1,1)(0,1,1,12)", "order": (1, 1, 1), "seasonal_order": (0, 1, 1, 12)},
    {"name": "SARIMA(0,1,1)(1,1,1,12)", "order": (0, 1, 1), "seasonal_order": (1, 1, 1, 12)},
]
BAND_LABELS = {
    "DA": "< 1,000 kWh/year",
    "DB": "1,000-2,499 kWh/year",
    "DC": "2,500-4,999 kWh/year",
    "DD": "5,000-14,999 kWh/year",
    "DE": ">= 15,000 kWh/year",
}


def build_business_problem_frame(target: str = TARGET, forecast_horizon_months: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"aspect": "Business problem", "detail": "Understand and forecast household electricity consumption over time."},
            {"aspect": "Target variable", "detail": f"{target} measured in kilowatts (kW)."},
            {"aspect": "Observation level", "detail": "Minute-level meter readings aggregated to monthly averages for modeling."},
            {"aspect": "Forecast horizon", "detail": f"{forecast_horizon_months} months ahead."},
            {"aspect": "Business value", "detail": "Support consumption planning, cost estimation, and seasonal energy analysis."},
            {"aspect": "Main analytical goal", "detail": "Detect trend, yearly seasonality, and produce reliable monthly forecasts."},
            {"aspect": "Success criteria", "detail": "Low holdout error, interpretable seasonal behavior, and stable residual diagnostics."},
            {"aspect": "Main constraints", "detail": "Short monthly history after aggregation and imperfect raw minute-level coverage."},
        ]
    )


def build_analysis_objectives_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"objective": "Describe the series", "why_it_matters": "Establish whether the household has regular seasonal consumption patterns."},
            {"objective": "Test stationarity", "why_it_matters": "Choose appropriate differencing before ARIMA/SARIMA modeling."},
            {"objective": "Compare candidate models", "why_it_matters": "Avoid assuming one model is best before checking holdout performance."},
            {"objective": "Estimate future consumption", "why_it_matters": "Provide actionable monthly forecasts and uncertainty intervals."},
            {"objective": "Link usage to official prices", "why_it_matters": "Translate technical forecasts into approximate monetary cost."},
        ]
    )


def ensure_output_dir(base_dir: str | Path | None = None, dirname: str = "outputs_enhanced") -> Path:
    root = Path.cwd() if base_dir is None else Path(base_dir)
    out_dir = root / dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path.resolve()
    return None


def _search_named_file(filename: str, roots: list[Path]) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        try:
            return next(root.rglob(filename)).resolve()
        except StopIteration:
            continue
    return None


def resolve_dataset_path(
    explicit_path: str | Path | None = None,
    filename: str = "household_power_consumption.txt",
) -> Path:
    home = Path.home()
    candidates = [
        Path(explicit_path).expanduser() if explicit_path else None,
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
        home / "Downloads" / filename,
        home / "Documents" / "se\u0301rie temporelle" / "data" / filename,
        home / "Documents" / "serie temporelle" / "data" / filename,
    ]
    concrete_candidates = [path for path in candidates if path is not None]
    direct_match = _first_existing_path(concrete_candidates)
    if direct_match is not None:
        return direct_match

    nested_match = _search_named_file(
        filename,
        [Path.cwd(), home / "Documents", home / "Downloads"],
    )
    if nested_match is not None:
        return nested_match

    raise FileNotFoundError(
        f"Unable to find '{filename}'. Pass an explicit path or place the file in the project/data directory."
    )


def build_source_schema_table(dataset_path: str | Path) -> pd.DataFrame:
    schema = pd.read_csv(dataset_path, sep=";", nrows=0)
    role_map = {
        "Date": "Calendar date",
        "Time": "Clock time",
        "Global_active_power": "Target variable used for forecasting",
        "Global_reactive_power": "Electrical measurement",
        "Voltage": "Electrical measurement",
        "Global_intensity": "Electrical measurement",
        "Sub_metering_1": "Sub-metering channel",
        "Sub_metering_2": "Sub-metering channel",
        "Sub_metering_3": "Sub-metering channel",
    }
    rows = []
    for idx, column in enumerate(schema.columns, start=1):
        clean_name = str(column).replace("\ufeff", "").strip()
        rows.append(
            {
                "position": idx,
                "column": clean_name,
                "used_in_this_notebook": clean_name in {"Date", "Time", TARGET},
                "role": role_map.get(clean_name, "Input feature"),
            }
        )
    return pd.DataFrame(rows)


def load_monthly_series(
    dataset_path: str | Path,
    target: str = TARGET,
    return_metadata: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series] | tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    data_path = Path(dataset_path)
    df_raw = pd.read_csv(
        data_path,
        sep=";",
        usecols=["Date", "Time", target],
        na_values=["?", "NA", ""],
        low_memory=False,
    )
    df_raw.columns = df_raw.columns.str.replace("\ufeff", "", regex=True).str.strip()
    raw_row_count = int(len(df_raw))
    df_raw[target] = pd.to_numeric(df_raw[target], errors="coerce")
    target_missing_before_drop = int(df_raw[target].isna().sum())
    df_raw["Datetime"] = pd.to_datetime(
        df_raw["Date"].astype(str) + " " + df_raw["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    datetime_missing_before_drop = int(df_raw["Datetime"].isna().sum())
    df = (
        df_raw.dropna(subset=["Datetime", target])
        .set_index("Datetime")
        .sort_index()
        [[target]]
    )
    duplicate_timestamps = int(df.index.duplicated().sum())
    unique_timestamps = int(df.index.nunique())
    if len(df) > 1:
        median_step_minutes = float(df.index.to_series().diff().dropna().dt.total_seconds().median() / 60.0)
    else:
        median_step_minutes = np.nan

    expected_minute_points = int(((df.index.max() - df.index.min()).total_seconds() / 60) + 1) if not df.empty else 0
    estimated_missing_timestamps = max(expected_minute_points - unique_timestamps, 0)
    estimated_coverage_pct = (100.0 * unique_timestamps / expected_minute_points) if expected_minute_points else np.nan
    monthly_power_kw = df[target].resample("MS").mean().dropna().rename("monthly_power_kw")
    monthly_energy_kwh = (
        monthly_power_kw * monthly_power_kw.index.days_in_month * 24
    ).rename("monthly_energy_kwh")
    metadata = {
        "dataset_path": str(data_path.resolve()),
        "raw_row_count": raw_row_count,
        "clean_row_count": int(len(df)),
        "rows_removed": int(raw_row_count - len(df)),
        "target_missing_before_drop": target_missing_before_drop,
        "datetime_missing_before_drop": datetime_missing_before_drop,
        "duplicate_timestamps": duplicate_timestamps,
        "unique_timestamps": unique_timestamps,
        "expected_minute_points": expected_minute_points,
        "estimated_missing_timestamps": estimated_missing_timestamps,
        "estimated_coverage_pct": estimated_coverage_pct,
        "median_step_minutes": median_step_minutes,
        "minute_start": df.index.min(),
        "minute_end": df.index.max(),
        "monthly_observation_count": int(len(monthly_power_kw)),
    }
    if return_metadata:
        return df, monthly_power_kw, monthly_energy_kwh, metadata
    return df, monthly_power_kw, monthly_energy_kwh


def build_data_understanding_tables(
    df_minute: pd.DataFrame,
    monthly_power_kw: pd.Series,
    metadata: dict[str, Any],
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_table = pd.DataFrame(
        {
            "metric": [
                "dataset_path",
                "raw_rows_read",
                "rows_after_cleaning",
                "rows_removed",
                "target_missing_before_drop",
                "datetime_parsing_failures",
                "duplicate_timestamps",
                "estimated_missing_timestamps",
                "estimated_coverage_pct",
                "median_sampling_step_minutes",
                "minute_series_start",
                "minute_series_end",
                "monthly_observations",
            ],
            "value": [
                metadata["dataset_path"],
                metadata["raw_row_count"],
                metadata["clean_row_count"],
                metadata["rows_removed"],
                metadata["target_missing_before_drop"],
                metadata["datetime_missing_before_drop"],
                metadata["duplicate_timestamps"],
                metadata["estimated_missing_timestamps"],
                round(float(metadata["estimated_coverage_pct"]), 4) if pd.notna(metadata["estimated_coverage_pct"]) else np.nan,
                round(float(metadata["median_step_minutes"]), 4) if pd.notna(metadata["median_step_minutes"]) else np.nan,
                metadata["minute_start"],
                metadata["minute_end"],
                metadata["monthly_observation_count"],
            ],
        }
    )

    monthly_coverage = df_minute[target].resample("MS").agg(["size", "mean", "min", "max"]).rename(
        columns={
            "size": "minute_observations",
            "mean": "mean_power_kw",
            "min": "min_power_kw",
            "max": "max_power_kw",
        }
    )
    monthly_coverage["expected_minute_observations"] = monthly_coverage.index.days_in_month * 24 * 60
    monthly_coverage["coverage_pct"] = (
        100.0 * monthly_coverage["minute_observations"] / monthly_coverage["expected_minute_observations"]
    )
    monthly_coverage["year"] = monthly_coverage.index.year
    monthly_coverage["month"] = monthly_coverage.index.month
    monthly_coverage["monthly_mean_after_resample"] = monthly_power_kw.reindex(monthly_coverage.index)
    monthly_coverage = monthly_coverage.reset_index().rename(columns={"Datetime": "month_start", "index": "month_start"})
    return summary_table, monthly_coverage


def plot_minute_level_profiles(df_minute: pd.DataFrame, out_dir: str | Path, target: str = TARGET) -> Path:
    out_path = Path(out_dir) / "minute_level_profiles_enhanced.png"

    hourly_profile = df_minute[target].groupby(df_minute.index.hour).mean()
    weekday_profile = df_minute[target].groupby(df_minute.index.dayofweek).mean().reindex(range(7))
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].plot(hourly_profile.index, hourly_profile.values, color="steelblue", linewidth=2, marker="o", markersize=4)
    axes[0].set_title("Average power by hour of day", fontsize=12)
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Average kW")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(weekday_labels, weekday_profile.values, color="darkorange", alpha=0.85)
    axes[1].set_title("Average power by day of week", fontsize=12)
    axes[1].set_xlabel("Weekday")
    axes[1].set_ylabel("Average kW")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    return out_path


def build_distribution_summary_table(df_minute: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    series = df_minute[target].dropna()
    quantiles = series.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return pd.DataFrame(
        {
            "metric": [
                "count",
                "mean",
                "std",
                "min",
                "p01",
                "p05",
                "p25",
                "median",
                "p75",
                "p95",
                "p99",
                "max",
                "skewness",
                "kurtosis",
            ],
            "value": [
                int(series.shape[0]),
                float(series.mean()),
                float(series.std()),
                float(series.min()),
                float(quantiles.loc[0.01]),
                float(quantiles.loc[0.05]),
                float(quantiles.loc[0.25]),
                float(quantiles.loc[0.50]),
                float(quantiles.loc[0.75]),
                float(quantiles.loc[0.95]),
                float(quantiles.loc[0.99]),
                float(series.max()),
                float(series.skew()),
                float(series.kurtosis()),
            ],
        }
    )


def plot_data_coverage_and_daily_patterns(
    df_minute: pd.DataFrame,
    monthly_coverage: pd.DataFrame,
    out_dir: str | Path,
    target: str = TARGET,
) -> Path:
    out_path = Path(out_dir) / "data_coverage_daily_patterns_enhanced.png"

    daily_mean = df_minute[target].resample("D").mean().dropna()
    daily_rolling = daily_mean.rolling(30, min_periods=7).mean()
    coverage_df = monthly_coverage.copy()
    coverage_df["missing_minutes"] = (
        coverage_df["expected_minute_observations"] - coverage_df["minute_observations"]
    )

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))

    axes[0].plot(daily_mean.index, daily_mean.values, color="lightsteelblue", linewidth=1, label="Daily mean")
    axes[0].plot(daily_rolling.index, daily_rolling.values, color="navy", linewidth=2, label="30-day rolling mean")
    axes[0].set_title("Daily average consumption before monthly aggregation", fontsize=12)
    axes[0].set_ylabel("kW")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(coverage_df["month_start"], coverage_df["coverage_pct"], color="teal", alpha=0.85)
    axes[1].axhline(100, color="red", linestyle="--", linewidth=1, label="100% expected coverage")
    axes[1].set_title("Monthly temporal coverage of minute-level data", fontsize=12)
    axes[1].set_ylabel("Coverage (%)")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(coverage_df["month_start"], coverage_df["missing_minutes"], color="darkorange", alpha=0.85)
    axes[2].set_title("Estimated missing minutes by month", fontsize=12)
    axes[2].set_ylabel("Missing minutes")
    axes[2].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    return out_path


def plot_calendar_heatmaps(df_minute: pd.DataFrame, out_dir: str | Path, target: str = TARGET) -> Path:
    out_path = Path(out_dir) / "calendar_heatmaps_enhanced.png"

    weekday_heatmap = (
        df_minute[target]
        .groupby([df_minute.index.hour, df_minute.index.dayofweek])
        .mean()
        .unstack()
        .reindex(index=range(24), columns=range(7))
    )
    month_heatmap = (
        df_minute[target]
        .groupby([df_minute.index.hour, df_minute.index.month])
        .mean()
        .unstack()
        .reindex(index=range(24), columns=range(1, 13))
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    im1 = axes[0].imshow(weekday_heatmap.values, aspect="auto", origin="lower", cmap="YlGnBu")
    axes[0].set_title("Average kW by hour and weekday", fontsize=12)
    axes[0].set_xlabel("Weekday")
    axes[0].set_ylabel("Hour")
    axes[0].set_xticks(range(7))
    axes[0].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[0].set_yticks(range(0, 24, 2))
    axes[0].set_yticklabels(range(0, 24, 2))
    plt.colorbar(im1, ax=axes[0], fraction=0.046, label="Average kW")

    im2 = axes[1].imshow(month_heatmap.values, aspect="auto", origin="lower", cmap="YlOrRd")
    axes[1].set_title("Average kW by hour and month", fontsize=12)
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Hour")
    axes[1].set_xticks(range(12))
    axes[1].set_xticklabels(range(1, 13))
    axes[1].set_yticks(range(0, 24, 2))
    axes[1].set_yticklabels(range(0, 24, 2))
    plt.colorbar(im2, ax=axes[1], fraction=0.046, label="Average kW")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    return out_path


def plot_distribution_overview(
    df_minute: pd.DataFrame,
    out_dir: str | Path,
    target: str = TARGET,
    sample_size: int = 120_000,
) -> Path:
    out_path = Path(out_dir) / "distribution_overview_enhanced.png"

    sampled = df_minute[[target]].dropna().copy()
    if len(sampled) > sample_size:
        sampled = sampled.sample(sample_size, random_state=42).sort_index()

    sampled["hour"] = sampled.index.hour
    sampled["weekday"] = sampled.index.dayofweek
    sampled["month"] = sampled.index.month

    weekday_groups = [sampled.loc[sampled["weekday"] == day, target].values for day in range(7)]
    hour_groups = [sampled.loc[sampled["hour"] == hour, target].values for hour in range(24)]
    month_groups = [sampled.loc[sampled["month"] == month, target].values for month in range(1, 13)]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    axes[0, 0].hist(sampled[target], bins=60, color="steelblue", alpha=0.85, edgecolor="white")
    axes[0, 0].set_title("Distribution of minute-level power values", fontsize=12)
    axes[0, 0].set_xlabel("kW")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    axes[0, 1].boxplot(weekday_groups, labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], showfliers=False)
    axes[0, 1].set_title("Distribution by weekday", fontsize=12)
    axes[0, 1].set_ylabel("kW")
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    axes[1, 0].boxplot(hour_groups, labels=list(range(24)), showfliers=False)
    axes[1, 0].set_title("Distribution by hour of day", fontsize=12)
    axes[1, 0].set_xlabel("Hour")
    axes[1, 0].set_ylabel("kW")
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    axes[1, 1].boxplot(month_groups, labels=list(range(1, 13)), showfliers=False)
    axes[1, 1].set_title("Distribution by calendar month", fontsize=12)
    axes[1, 1].set_xlabel("Month")
    axes[1, 1].set_ylabel("kW")
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    return out_path


def build_overview_table(monthly_power_kw: pd.Series, monthly_energy_kwh: pd.Series, dataset_path: str | Path) -> pd.DataFrame:
    annual_energy = monthly_energy_kwh.resample("YS").sum()
    return pd.DataFrame(
        {
            "metric": [
                "dataset_path",
                "start",
                "end",
                "observations",
                "mean_power_kw",
                "std_power_kw",
                "avg_annual_energy_kwh",
            ],
            "value": [
                str(Path(dataset_path).resolve()),
                monthly_power_kw.index.min().date().isoformat(),
                monthly_power_kw.index.max().date().isoformat(),
                int(len(monthly_power_kw)),
                round(float(monthly_power_kw.mean()), 4),
                round(float(monthly_power_kw.std()), 4),
                round(float(annual_energy.mean()), 2),
            ],
        }
    )


def plot_monthly_eda(monthly_power_kw: pd.Series, out_dir: str | Path) -> Path:
    out_path = Path(out_dir) / "eda_monthly_enhanced.png"
    rolling = monthly_power_kw.rolling(12)
    seasonal_profile = monthly_power_kw.groupby(monthly_power_kw.index.month).mean()

    fig, axes = plt.subplots(2, 1, figsize=(13, 7))

    axes[0].plot(monthly_power_kw, color="steelblue", linewidth=1.6, marker="o", markersize=4, label="Monthly mean")
    axes[0].plot(rolling.mean(), color="firebrick", linewidth=2, label="12-month rolling mean")
    axes[0].fill_between(
        monthly_power_kw.index,
        rolling.mean() - rolling.std(),
        rolling.mean() + rolling.std(),
        color="firebrick",
        alpha=0.15,
        label="+/- 1 rolling std",
    )
    axes[0].set_title("Monthly household power consumption", fontsize=13)
    axes[0].set_ylabel("kW")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].bar(seasonal_profile.index, seasonal_profile.values, color="steelblue", alpha=0.85)
    axes[1].set_title("Average profile by calendar month", fontsize=13)
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Average kW")
    axes[1].set_xticks(range(1, 13))
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    return out_path


def plot_decomposition(monthly_power_kw: pd.Series, out_dir: str | Path) -> Path:
    out_path = Path(out_dir) / "decomposition_monthly_enhanced.png"
    decomp = seasonal_decompose(monthly_power_kw, model="additive", period=12)
    fig = decomp.plot()
    fig.set_size_inches(13, 9)
    plt.suptitle("Additive decomposition (period = 12)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    return out_path


def _safe_adf_maxlag(series: pd.Series, requested: int = 12) -> int:
    schwert = int(np.floor(12 * (len(series) / 100.0) ** 0.25))
    upper = max(1, min(requested, schwert if schwert > 0 else requested, len(series) // 2 - 1, len(series) - 2))
    return upper


def adf_report(series: pd.Series, label: str, requested_maxlag: int = 12, regression: str = "c") -> dict[str, Any]:
    clean_series = series.dropna()
    if len(clean_series) < 10:
        raise ValueError(f"Series '{label}' is too short for a stable ADF test.")
    max_lag = _safe_adf_maxlag(clean_series, requested=requested_maxlag)
    result = adfuller(clean_series, autolag="AIC", maxlag=max_lag, regression=regression)
    stat, pvalue, used_lag, nobs, critical_values = result[:5]
    return {
        "series": label,
        "nobs": int(nobs),
        "maxlag_cap": int(max_lag),
        "used_lag": int(used_lag),
        "adf_stat": float(stat),
        "pvalue": float(pvalue),
        "critical_1pct": float(critical_values["1%"]),
        "critical_5pct": float(critical_values["5%"]),
        "critical_10pct": float(critical_values["10%"]),
        "stationary_5pct": bool(pvalue < 0.05),
    }


def build_stationarity_table(monthly_power_kw: pd.Series) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    variants = {
        "Raw": monthly_power_kw,
        "Diff(1)": monthly_power_kw.diff(1).dropna(),
        "Diff(12)": monthly_power_kw.diff(12).dropna(),
        "Diff(1)+Diff(12)": monthly_power_kw.diff(1).diff(12).dropna(),
    }
    rows = [adf_report(series, label) for label, series in variants.items()]
    return pd.DataFrame(rows), variants


def plot_stationarity_variants(monthly_power_kw: pd.Series, variants: dict[str, pd.Series], table: pd.DataFrame, out_dir: str | Path) -> Path:
    out_path = Path(out_dir) / "stationarity_variants_enhanced.png"
    pvalues = dict(zip(table["series"], table["pvalue"]))

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    selected = [
        ("Raw", monthly_power_kw, "steelblue", "kW"),
        ("Diff(12)", variants["Diff(12)"], "darkorange", "Seasonal diff"),
        ("Diff(1)+Diff(12)", variants["Diff(1)+Diff(12)"], "seagreen", "Double diff"),
    ]

    for ax, (label, series, color, ylabel) in zip(axes, selected):
        ax.plot(series, color=color, linewidth=1.5, marker="o", markersize=3)
        if label != "Raw":
            ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
        ax.set_title(f"{label} (ADF p-value = {pvalues[label]:.4f})", fontsize=12)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    return out_path


def plot_correlogram(series: pd.Series, title: str, out_dir: str | Path, slug: str, max_lags: int | None = None) -> Path:
    clean_series = series.dropna()
    if max_lags is None:
        max_lags = max(3, min(24, len(clean_series) // 2 - 1))
    out_path = Path(out_dir) / f"{slug}.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(clean_series, lags=max_lags, ax=axes[0], alpha=0.05, use_vlines=True, markersize=4)
    axes[0].set_title(f"ACF - {title}", fontsize=12)
    axes[0].set_xlabel("Lag (months)")
    axes[0].grid(True, alpha=0.3, linestyle="--")

    plot_pacf(clean_series, lags=max_lags, ax=axes[1], alpha=0.05, method="ywm", use_vlines=True, markersize=4)
    axes[1].set_title(f"PACF - {title}", fontsize=12)
    axes[1].set_xlabel("Lag (months)")
    axes[1].grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    return out_path


def split_train_test(series: pd.Series, train_ratio: float = 0.8) -> tuple[pd.Series, pd.Series]:
    split_idx = int(len(series) * train_ratio)
    return series.iloc[:split_idx], series.iloc[split_idx:]


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    safe_denominator = np.where(np.abs(y_true.values) < 1e-9, np.nan, y_true.values)
    mape = np.nanmean(np.abs((y_true.values - y_pred.values) / safe_denominator)) * 100
    smape = np.nanmean(
        2 * np.abs(y_true.values - y_pred.values) / (np.abs(y_true.values) + np.abs(y_pred.values))
    ) * 100
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mape),
        "smape": float(smape),
    }


def fit_candidate_models(
    train: pd.Series,
    test: pd.Series,
    model_specs: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    specs = DEFAULT_MODEL_SPECS if model_specs is None else model_specs
    comparison_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    forecasts: dict[str, dict[str, Any]] = {}

    for spec in specs:
        model_name = spec["name"]
        kwargs = {
            "order": spec["order"],
            "enforce_stationarity": False,
            "enforce_invertibility": False,
        }
        if "seasonal_order" in spec:
            kwargs["seasonal_order"] = spec["seasonal_order"]

        try:
            fitted = ARIMA(train, **kwargs).fit()
            forecast_res = fitted.get_forecast(steps=len(test))
            y_pred = pd.Series(forecast_res.predicted_mean, index=test.index, name=model_name)
            metrics = evaluate_predictions(test, y_pred)

            comparison_rows.append(
                {
                    "name": model_name,
                    "order": spec["order"],
                    "seasonal_order": spec.get("seasonal_order"),
                    "aic": float(fitted.aic),
                    "bic": float(fitted.bic),
                    **metrics,
                    "successful": True,
                    "error": None,
                }
            )
            fitted_models[model_name] = fitted
            forecasts[model_name] = {"pred": y_pred, "conf_int": forecast_res.conf_int()}
        except Exception as exc:
            comparison_rows.append(
                {
                    "name": model_name,
                    "order": spec["order"],
                    "seasonal_order": spec.get("seasonal_order"),
                    "aic": np.nan,
                    "bic": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "mape": np.nan,
                    "smape": np.nan,
                    "successful": False,
                    "error": str(exc),
                }
            )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        by=["successful", "rmse", "aic"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return comparison, fitted_models, forecasts


def plot_forecast_comparison(
    train: pd.Series,
    test: pd.Series,
    forecasts: dict[str, dict[str, Any]],
    best_model_name: str,
    benchmark_name: str | None,
    out_dir: str | Path,
) -> Path:
    out_path = Path(out_dir) / "forecast_comparison_enhanced.png"
    best_forecast = forecasts[best_model_name]["pred"]
    conf_int = forecasts[best_model_name]["conf_int"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    axes[0].plot(train, color="steelblue", linewidth=1.5, marker="o", markersize=4, label="Train")
    axes[0].plot(test, color="darkorange", linewidth=1.8, marker="o", markersize=4, label="Test")
    axes[0].plot(best_forecast, color="firebrick", linewidth=2, linestyle="--", marker="s", markersize=5, label=best_model_name)
    axes[0].fill_between(
        test.index,
        conf_int.iloc[:, 0].values,
        conf_int.iloc[:, 1].values,
        color="firebrick",
        alpha=0.12,
        label="95% interval",
    )
    axes[0].axvline(test.index[0], color="gray", linestyle=":", linewidth=1.3)
    axes[0].set_title("Forecast comparison on the holdout set", fontsize=13)
    axes[0].set_ylabel("kW")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(test, color="darkorange", linewidth=2, marker="o", markersize=6, label="Observed")
    axes[1].plot(best_forecast, color="firebrick", linewidth=2, linestyle="--", marker="s", markersize=6, label=best_model_name)
    if benchmark_name and benchmark_name in forecasts and benchmark_name != best_model_name:
        benchmark_pred = forecasts[benchmark_name]["pred"]
        axes[1].plot(
            benchmark_pred,
            color="purple",
            linewidth=1.5,
            linestyle=":",
            marker="^",
            markersize=5,
            label=benchmark_name,
        )
    axes[1].set_title("Zoom on the test period", fontsize=13)
    axes[1].set_ylabel("kW")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    return out_path


def plot_residual_diagnostics(fitted_model: Any, out_dir: str | Path, model_name: str) -> tuple[pd.DataFrame, Path]:
    out_path = Path(out_dir) / "residual_diagnostics_enhanced.png"
    residuals = fitted_model.resid.dropna()
    valid_lags = [lag for lag in [6, 12, 18, 24] if lag < len(residuals)]
    lb = acorr_ljungbox(residuals, lags=valid_lags, return_df=True) if valid_lags else pd.DataFrame()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(residuals, color="steelblue", linewidth=1.2, marker="o", markersize=4)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0, 0].set_title(f"Residuals - {model_name}", fontsize=12)
    axes[0, 0].set_ylabel("Residual")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(residuals, bins=12, density=True, alpha=0.75, color="darkorange", edgecolor="black")
    if residuals.nunique() > 1:
        x_grid = np.linspace(residuals.min(), residuals.max(), 200)
        kde = stats.gaussian_kde(residuals)(x_grid)
        axes[0, 1].plot(x_grid, kde, "b-", linewidth=2, label="KDE")
        axes[0, 1].plot(
            x_grid,
            stats.norm.pdf(x_grid, residuals.mean(), residuals.std()),
            "r--",
            linewidth=2,
            label="Normal",
        )
    axes[0, 1].set_title("Residual distribution", fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    qqplot(residuals, line="s", ax=axes[1, 0], alpha=0.6)
    axes[1, 0].set_title("Q-Q plot", fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)

    acf_lags = max(3, min(20, len(residuals) // 2 - 1))
    plot_acf(residuals, lags=acf_lags, ax=axes[1, 1], alpha=0.05, use_vlines=True, markersize=4)
    axes[1, 1].set_title("Residual ACF", fontsize=12)
    axes[1, 1].set_xlabel("Lag (months)")
    axes[1, 1].grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    return lb, out_path


def build_future_forecast(fitted_model: Any, periods: int = 12) -> pd.DataFrame:
    forecast_res = fitted_model.get_forecast(steps=periods)
    conf_int = forecast_res.conf_int()
    return pd.DataFrame(
        {
            "forecast": forecast_res.predicted_mean,
            "lower_95": conf_int.iloc[:, 0],
            "upper_95": conf_int.iloc[:, 1],
        }
    )


def make_requests_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; household-power-analysis/1.0)",
            "Accept": "application/json, text/csv, */*",
        }
    )
    return session


def _empty_remote_frame(source: str, error: Exception | str) -> pd.DataFrame:
    print(f"Skipping {source}: {error}")
    return pd.DataFrame()


def load_world_bank(session: requests.Session, timeout: int = 15) -> pd.DataFrame:
    url = (
        "https://api.worldbank.org/v2/country/FR/indicator/"
        "EG.USE.ELEC.KH.PC?format=json&per_page=100&mrv=40"
    )
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        records = payload[1]
        df = pd.DataFrame(
            [{"year": int(row["date"]), "elec_kwh_per_capita": row["value"]} for row in records]
        )
        return df.dropna().sort_values("year").reset_index(drop=True)
    except Exception as exc:
        return _empty_remote_frame("World Bank", exc)


def _jsonstat_codes_and_labels(dim_meta: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    category = dim_meta.get("category", {})
    index_meta = category.get("index", {})
    label_meta = category.get("label", {})

    if isinstance(index_meta, dict):
        codes = [code for code, _ in sorted(index_meta.items(), key=lambda item: item[1])]
    elif isinstance(index_meta, list):
        codes = list(index_meta)
    else:
        codes = []

    labels = {code: str(label_meta.get(code, code)) for code in codes}
    return codes, labels


def _unravel_index(flat_index: int, sizes: list[int]) -> list[int]:
    coords: list[int] = []
    for size in reversed(sizes):
        coords.append(flat_index % size)
        flat_index //= size
    return list(reversed(coords))


def flatten_jsonstat_dataset(payload: dict[str, Any]) -> pd.DataFrame:
    ids = payload["id"]
    sizes = payload["size"]
    dimension = payload["dimension"]
    values = payload["value"]

    code_maps: dict[str, list[str]] = {}
    label_maps: dict[str, dict[str, str]] = {}
    for dim_id in ids:
        codes, labels = _jsonstat_codes_and_labels(dimension[dim_id])
        code_maps[dim_id] = codes
        label_maps[dim_id] = labels

    if isinstance(values, list):
        iterator = enumerate(values)
    else:
        iterator = ((int(idx), value) for idx, value in values.items())

    rows: list[dict[str, Any]] = []
    for flat_index, value in iterator:
        if value is None:
            continue
        coords = _unravel_index(flat_index, sizes)
        row: dict[str, Any] = {"value": value}
        for dim_id, coord in zip(ids, coords):
            code = code_maps[dim_id][coord]
            row[dim_id] = code
            row[f"{dim_id}_label"] = label_maps[dim_id].get(code, code)
        rows.append(row)
    return pd.DataFrame(rows)


def estimate_eurostat_band(monthly_energy_kwh: pd.Series) -> dict[str, Any]:
    annual_energy = monthly_energy_kwh.resample("YS").sum()
    avg_annual_kwh = float(annual_energy.mean())

    if avg_annual_kwh < 1000:
        band_code = "DA"
    elif avg_annual_kwh < 2500:
        band_code = "DB"
    elif avg_annual_kwh < 5000:
        band_code = "DC"
    elif avg_annual_kwh < 15000:
        band_code = "DD"
    else:
        band_code = "DE"

    return {
        "band_code": band_code,
        "band_label": BAND_LABELS[band_code],
        "avg_annual_kwh": round(avg_annual_kwh, 2),
    }


def _band_mask(df: pd.DataFrame, band_code: str) -> pd.Series:
    band_col = next((col for col in df.columns if "cons" in col.lower() and not col.endswith("_label")), None)
    if band_col is None:
        return pd.Series(False, index=df.index)

    code = df[band_col].fillna("").astype(str).str.upper()
    label = df.get(f"{band_col}_label", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()

    if band_code == "DA":
        return code.eq("DA") | label.str.contains("BAND DA") | label.str.contains("LESS THAN 1000")
    if band_code == "DB":
        return code.eq("DB") | code.str.contains("1000-2499") | (label.str.contains("1000") & label.str.contains("2499"))
    if band_code == "DC":
        return code.eq("DC") | code.str.contains("2500-4999") | (label.str.contains("2500") & label.str.contains("4999"))
    if band_code == "DD":
        return code.eq("DD") | code.str.contains("5000-14999") | (label.str.contains("5000") & (label.str.contains("14999") | label.str.contains("15000")))
    return code.eq("DE") | label.str.contains("BAND DE") | label.str.contains("15000")


def load_eurostat_prices(
    session: requests.Session,
    preferred_band_code: str,
    timeout: int = 15,
) -> pd.DataFrame:
    url = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/nrg_pc_204"
    params = {
        "lang": "EN",
        "geo": "FR",
        "currency": "EUR",
        "tax": "I_TAX",
        "unit": "KWH",
    }
    try:
        response = session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        raw_df = flatten_jsonstat_dataset(response.json())
        if raw_df.empty:
            raise ValueError("Eurostat returned an empty dataset.")

        df = raw_df.copy()
        if "geo" in df.columns:
            df = df[df["geo"].eq("FR")]

        freq_col = "freq" if "freq" in df.columns else None
        if freq_col and ((df[freq_col].astype(str).str.upper() == "A").any() or df[f"{freq_col}_label"].astype(str).str.contains("annual", case=False, na=False).any()):
            annual_mask = (df[freq_col].astype(str).str.upper() == "A") | df[f"{freq_col}_label"].astype(str).str.contains("annual", case=False, na=False)
            df = df[annual_mask]

        product_col = next((col for col in df.columns if col == "product"), None)
        if product_col and df[product_col].nunique() > 1:
            product_mask = df[product_col].astype(str).eq("6000") | df[f"{product_col}_label"].astype(str).str.contains("elect", case=False, na=False)
            if product_mask.any():
                df = df[product_mask]

        mask = _band_mask(df, preferred_band_code)
        if mask.any():
            df = df[mask]

        time_col = "time"
        if time_col not in df.columns:
            raise ValueError("The Eurostat response does not contain a time dimension.")

        df["year"] = pd.to_numeric(df[time_col].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
        df = df.dropna(subset=["year", "value"]).copy()
        df["year"] = df["year"].astype(int)
        df["price_eur_per_kwh"] = pd.to_numeric(df["value"], errors="coerce")

        result = (
            df.groupby("year", as_index=False)["price_eur_per_kwh"]
            .mean()
            .sort_values("year")
            .reset_index(drop=True)
        )
        result["consumption_band"] = preferred_band_code
        return result
    except Exception as exc:
        return _empty_remote_frame("Eurostat", exc)


def load_owid_energy(session: requests.Session, timeout: int = 20) -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        df_raw = pd.read_csv(io.StringIO(response.text), low_memory=False)
        df_fr = df_raw[df_raw["country"].eq("France")].copy()
        columns = [
            "year",
            "electricity_demand",
            "renewables_share_energy",
            "nuclear_share_energy",
            "fossil_share_energy",
            "per_capita_electricity",
        ]
        available = [column for column in columns if column in df_fr.columns]
        df = df_fr[available].dropna().sort_values("year").reset_index(drop=True)
        if "year" in df.columns:
            df["year"] = df["year"].astype(int)
        return df
    except Exception as exc:
        return _empty_remote_frame("OWID", exc)


def build_energy_merge(df_wb: pd.DataFrame, df_es_annual: pd.DataFrame, df_owid: pd.DataFrame) -> pd.DataFrame:
    if df_wb.empty or df_es_annual.empty or df_owid.empty:
        return pd.DataFrame()

    required_owid_cols = [col for col in ["renewables_share_energy", "nuclear_share_energy", "fossil_share_energy"] if col in df_owid.columns]
    merged = (
        df_es_annual.merge(df_wb[["year", "elec_kwh_per_capita"]], on="year", how="inner")
        .merge(df_owid[["year", *required_owid_cols]], on="year", how="inner")
        .dropna()
        .sort_values("year")
        .reset_index(drop=True)
    )
    return merged


def plot_correlation_heatmap(df_merge: pd.DataFrame, out_dir: str | Path) -> Path | None:
    if df_merge.empty:
        return None

    out_path = Path(out_dir) / "correlation_matrix_enhanced.png"
    corr = df_merge.drop(columns=["year"]).corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(image, ax=ax, fraction=0.046, label="Pearson correlation")

    labels = corr.columns.tolist()
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i, j]
            color = "white" if abs(value) > 0.6 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.5, color=color, fontweight="bold")

    ax.set_title("Correlations between official energy indicators (France)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    return out_path


def build_cost_context(monthly_energy_kwh: pd.Series, df_es_annual: pd.DataFrame) -> pd.DataFrame:
    if df_es_annual.empty:
        return pd.DataFrame()

    df_cost = monthly_energy_kwh.to_frame().copy()
    df_cost["year"] = df_cost.index.year
    df_cost = df_cost.join(df_es_annual.set_index("year"), on="year")
    df_cost = df_cost.dropna(subset=["price_eur_per_kwh"]).copy()
    df_cost["estimated_cost_eur"] = df_cost["monthly_energy_kwh"] * df_cost["price_eur_per_kwh"]
    return df_cost


def plot_cost_context(df_cost: pd.DataFrame, out_dir: str | Path) -> Path | None:
    if df_cost.empty:
        return None

    out_path = Path(out_dir) / "monthly_cost_context_enhanced.png"
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].bar(df_cost.index, df_cost["monthly_energy_kwh"], color="steelblue", alpha=0.85)
    axes[0].set_title("Estimated monthly household energy", fontsize=13)
    axes[0].set_ylabel("kWh")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].plot(df_cost.index, df_cost["estimated_cost_eur"], color="firebrick", linewidth=2, marker="o", markersize=4)
    axes[1].set_title("Estimated monthly electricity cost using Eurostat annual prices", fontsize=13)
    axes[1].set_ylabel("EUR")
    axes[1].grid(True, alpha=0.3)

    price_axis = axes[1].twinx()
    price_axis.plot(df_cost.index, df_cost["price_eur_per_kwh"], color="gray", linewidth=1.5, linestyle="--")
    price_axis.set_ylabel("EUR / kWh")
    price_axis.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.2f}"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    return out_path
