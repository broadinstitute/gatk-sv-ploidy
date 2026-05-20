"""Aggregate one or more gatk-sv-ploidy runs into a PDF report."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from gatk_sv_ploidy._logging import log_output_artifacts, tool_logging_context


_SEX_CHROMS = frozenset({"chrX", "chrY"})
_NORMAL_SEX_LABELS = frozenset({"MALE", "FEMALE"})
_NO_ANEUPLOIDY_LABELS = frozenset({"NONE", "NORMAL", "", "nan", "NaN"})
_CASE_SECTIONS = (
    ("confident_sex_aneuploidy", "Confident Sex Aneuploidies"),
    ("confident_polyploidy", "Confident Polyploidy"),
    ("confident_autosomal_aneuploidy", "Confident Autosomal Aneuploidy"),
    ("low_confidence_aneuploidy", "Low-confidence Aneuploidies"),
)
_REQUIRED_PREDICTION_COLUMNS = {"sample", "sex", "predicted_aneuploidy_type"}
_REQUIRED_CHROM_COLUMNS = {
    "sample",
    "chromosome",
    "copy_number",
    "mean_cn_probability",
    "plq",
    "is_aneuploid",
}
_EVENT_COLUMNS = [
    "batch_id",
    "batch_label",
    "sample",
    "sample_key",
    "category",
    "confidence",
    "chromosome",
    "copy_number",
    "expected_copy_number",
    "copy_number_delta",
    "mean_cn_probability",
    "plq",
    "n_bins",
    "frac_bins_retained",
    "median_depth",
    "mean_depth",
    "sample_depth_ratio",
    "sample_depth_percentile",
    "sample_overdispersion_map",
    "sample_overdispersion_percentile",
    "sample_score",
    "predicted_aneuploidy_type",
]
_CASE_COLUMNS = [
    "category",
    "batch_id",
    "batch_label",
    "sample",
    "sample_key",
    "sex",
    "predicted_aneuploidy_type",
    "autosomal_aneuploidy_type",
    "allosomal_aneuploidy_type",
    "baseline_ploidy_type",
    "autosomal_baseline_cn",
    "score",
    "sample_depth_ratio",
    "sample_depth_percentile",
    "sample_overdispersion_map",
    "sample_overdispersion_percentile",
    "median_frac_bins_retained",
    "used_filtered_chrom_stats",
    "true_aneuploidy_type",
    "anomalous_contigs",
    "n_anomalous_contigs",
]


def _report_figure_size(height_mm: float = 150.0) -> tuple[float, float]:
    """Return report figure size, using plot style helpers when available."""
    try:
        from gatk_sv_ploidy._plot_style import double_column_size

        return double_column_size(height_mm)
    except ModuleNotFoundError:
        return 183.0 / 25.4, min(float(height_mm), 170.0) / 25.4


def _apply_report_theme() -> None:
    """Apply shared plot theme when optional style dependencies are present."""
    try:
        from gatk_sv_ploidy._plot_style import apply_theme

        apply_theme()
    except ModuleNotFoundError:
        return None


def _expected_allosome_copy_number_pairs(
    autosomal_baseline_cn: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return female-like and male-like chrX/chrY CN pairs for a baseline CN."""
    baseline = int(autosomal_baseline_cn)
    female_like = (baseline, 0)
    male_y_cn = max(1, baseline // 2)
    male_y_cn = min(male_y_cn, baseline)
    male_like = (baseline - male_y_cn, male_y_cn)
    return female_like, male_like


@dataclass
class RunData:
    """Loaded artifacts for one run directory."""

    batch_id: int
    batch_label: str
    work_dir: Path
    pred_df: pd.DataFrame
    chrom_df: pd.DataFrame
    chrom_stats_source: Path
    used_filtered_chrom_stats: bool
    baseline_df: pd.DataFrame | None
    bin_df: pd.DataFrame | None
    site_data: dict[str, np.ndarray] | None
    missing_artifacts: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the aggregate subcommand."""
    p = argparse.ArgumentParser(
        description="Aggregate one or more gatk-sv-ploidy run directories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "work_dirs",
        nargs="+",
        help="One or more work directories produced by run_ploidy.sh",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory for the aggregate report and sidecar tables",
    )
    p.add_argument(
        "--output-name",
        default="aggregate_report.pdf",
        help="Filename for the aggregate PDF report",
    )
    p.add_argument(
        "--batch-label",
        action="append",
        default=None,
        help="Optional batch label. Repeat once per work directory.",
    )
    p.add_argument(
        "--prob-threshold",
        type=float,
        default=0.5,
        help="Mean CN probability threshold for confident aneuploidy calls",
    )
    p.add_argument(
        "--binq-field",
        default="auto",
        help="PPD quality field to annotate sample plots with when available",
    )
    p.add_argument(
        "--min-het-alt",
        type=int,
        default=3,
        help="Minimum alternate reads for site allele-fraction points in sample plots",
    )
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Validate aggregate arguments before doing filesystem work."""
    output_name = Path(args.output_name)
    if output_name.name != args.output_name or output_name.suffix.lower() != ".pdf":
        raise ValueError("--output-name must be a PDF filename, not a path")
    if args.batch_label is not None and len(args.batch_label) != len(args.work_dirs):
        raise ValueError("--batch-label must be provided once per work directory")
    if not 0.0 <= float(args.prob_threshold) <= 1.0:
        raise ValueError("--prob-threshold must be between 0 and 1")
    if int(args.min_het_alt) < 0:
        raise ValueError("--min-het-alt must be non-negative")


def _default_batch_labels(work_dirs: Iterable[str]) -> list[str]:
    """Return deterministic default batch labels from work directory names."""
    labels: list[str] = []
    seen: dict[str, int] = {}
    for index, work_dir in enumerate(work_dirs, start=1):
        name = Path(work_dir).name or f"batch_{index}"
        count = seen.get(name, 0) + 1
        seen[name] = count
        labels.append(name if count == 1 else f"{name}_{count}")
    return labels


def _validate_columns(df: pd.DataFrame, required: set[str], path: Path) -> None:
    """Fail with a clear message when a required input table is malformed."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )


def _read_tsv(path: Path, *, compression: str | None = "infer") -> pd.DataFrame:
    """Read a TSV with a normalized error message."""
    try:
        return pd.read_csv(path, sep="\t", compression=compression)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read TSV {path}: {exc}") from exc


def _optional_missing(
    batch_label: str,
    artifact: str,
    path: Path,
    reason: str = "not found",
) -> dict[str, str]:
    """Return a row for the missing-artifact sidecar."""
    return {
        "batch_label": batch_label,
        "artifact": artifact,
        "path": str(path),
        "reason": reason,
    }


def _load_optional_baseline(
    work_dir: Path,
    batch_label: str,
    missing: list[dict[str, str]],
) -> pd.DataFrame | None:
    path = work_dir / "polyploidy" / "sample_autosomal_baseline_cn.tsv"
    if not path.exists():
        missing.append(_optional_missing(batch_label, "polyploidy_manifest", path))
        return None
    baseline_df = _read_tsv(path)
    if "sample" not in baseline_df.columns:
        missing.append(
            _optional_missing(
                batch_label,
                "polyploidy_manifest",
                path,
                "missing sample column",
            )
        )
        return None
    return baseline_df


def _load_optional_bin_stats(
    work_dir: Path,
    batch_label: str,
    missing: list[dict[str, str]],
    binq_field: str,
) -> pd.DataFrame | None:
    path = work_dir / "infer" / "bin_stats.tsv.gz"
    if not path.exists():
        missing.append(_optional_missing(batch_label, "bin_stats", path))
        return None

    from gatk_sv_ploidy.plot import (
        _annotate_binq_values,
        _annotate_ignored_bins,
        _apply_plot_depth_bin_columns,
    )

    bin_df = _apply_plot_depth_bin_columns(_read_tsv(path, compression="gzip"))
    ignored_path = work_dir / "call" / "ignored_bins.tsv.gz"
    if ignored_path.exists():
        bin_df = _annotate_ignored_bins(bin_df, _read_tsv(ignored_path, compression="gzip"))
    else:
        missing.append(_optional_missing(batch_label, "ignored_bins", ignored_path))

    ppd_quality_path = work_dir / "ppd" / "ppd_bin_quality.tsv"
    if ppd_quality_path.exists():
        bin_df = _annotate_binq_values(bin_df, _read_tsv(ppd_quality_path), binq_field)
    else:
        missing.append(_optional_missing(batch_label, "ppd_bin_quality", ppd_quality_path))
    return bin_df


def _load_optional_site_data(
    work_dir: Path,
    batch_label: str,
    missing: list[dict[str, str]],
) -> dict[str, np.ndarray] | None:
    path = work_dir / "preprocess" / "site_data.npz"
    if not path.exists():
        missing.append(_optional_missing(batch_label, "site_data", path))
        return None
    try:
        from gatk_sv_ploidy.data import load_site_data

        return load_site_data(str(path))
    except Exception as exc:
        missing.append(_optional_missing(batch_label, "site_data", path, str(exc)))
        return None


def _load_run_data(
    work_dir: str | Path,
    *,
    batch_id: int,
    batch_label: str,
    binq_field: str,
) -> RunData:
    """Load required and optional artifacts from one run directory."""
    root = Path(work_dir)
    pred_path = root / "call" / "aneuploidy_type_predictions.tsv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing required predictions file: {pred_path}")
    pred_df = _read_tsv(pred_path)
    _validate_columns(pred_df, _REQUIRED_PREDICTION_COLUMNS, pred_path)

    filtered_path = root / "call" / "chromosome_stats.filtered.tsv"
    infer_path = root / "infer" / "chromosome_stats.tsv"
    chrom_path = filtered_path if filtered_path.exists() else infer_path
    if not chrom_path.exists():
        raise FileNotFoundError(
            f"Missing required chromosome stats file: {filtered_path} or {infer_path}"
        )
    chrom_df = _read_tsv(chrom_path)
    _validate_columns(chrom_df, _REQUIRED_CHROM_COLUMNS, chrom_path)

    missing: list[dict[str, str]] = []
    baseline_df = _load_optional_baseline(root, batch_label, missing)
    bin_df = _load_optional_bin_stats(root, batch_label, missing, binq_field)
    site_data = _load_optional_site_data(root, batch_label, missing)

    return RunData(
        batch_id=batch_id,
        batch_label=batch_label,
        work_dir=root,
        pred_df=pred_df,
        chrom_df=chrom_df,
        chrom_stats_source=chrom_path,
        used_filtered_chrom_stats=chrom_path == filtered_path,
        baseline_df=baseline_df,
        bin_df=bin_df,
        site_data=site_data,
        missing_artifacts=missing,
    )


def _add_batch_columns(df: pd.DataFrame, run: RunData) -> pd.DataFrame:
    """Attach batch/run identifiers to a DataFrame."""
    out = df.copy()
    out.insert(0, "batch_id", run.batch_id)
    out.insert(1, "batch_label", run.batch_label)
    out.insert(2, "work_dir", str(run.work_dir))
    if "sample" in out.columns:
        out.insert(3, "sample_key", out["sample"].map(lambda s: f"{run.batch_label}/{s}"))
    return out


def _first_non_null(series: pd.Series, default: Any = np.nan) -> Any:
    values = series.dropna()
    if values.empty:
        return default
    return values.iloc[0]


def _batch_percentile(series: pd.Series) -> pd.Series:
    """Return 0-100 within-batch percentiles for a numeric series."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return numeric.rank(method="average", pct=True) * 100.0


def _build_sample_table(runs: list[RunData]) -> pd.DataFrame:
    """Build one row per sample per batch with batch-context metrics."""
    pred_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    for run in runs:
        pred_raw = run.pred_df.copy()
        if run.baseline_df is not None and "sample" in run.baseline_df.columns:
            baseline_cols = [
                col for col in [
                    "sample",
                    "autosomal_baseline_cn",
                    "baseline_cn_call",
                    "baseline_cn_reason",
                    "include_in_infer",
                ]
                if col in run.baseline_df.columns
            ]
            if len(baseline_cols) > 1:
                baseline_view = run.baseline_df[baseline_cols].drop_duplicates(
                    subset=["sample"],
                    keep="last",
                )
                pred_raw = pred_raw.merge(
                    baseline_view,
                    on="sample",
                    how="left",
                    suffixes=("", "__manifest"),
                )
                for col in baseline_cols:
                    if col == "sample":
                        continue
                    manifest_col = f"{col}__manifest"
                    if manifest_col not in pred_raw.columns:
                        continue
                    if col in pred_raw.columns:
                        pred_raw[col] = pred_raw[col].where(
                            pred_raw[col].notna(),
                            pred_raw[manifest_col],
                        )
                    else:
                        pred_raw[col] = pred_raw[manifest_col]
                    pred_raw = pred_raw.drop(columns=[manifest_col])
        pred = _add_batch_columns(pred_raw, run)
        pred["chrom_stats_source"] = str(run.chrom_stats_source)
        pred["used_filtered_chrom_stats"] = run.used_filtered_chrom_stats
        pred_frames.append(pred)

        chrom = run.chrom_df.copy()
        metrics = chrom.groupby("sample", sort=False).agg(
            sample_overdispersion_map=("sample_overdispersion_map", _first_non_null)
            if "sample_overdispersion_map" in chrom.columns else ("copy_number", lambda _: np.nan),
            sample_depth_map=("sample_depth_map", _first_non_null)
            if "sample_depth_map" in chrom.columns else ("copy_number", lambda _: np.nan),
            median_frac_bins_retained=("frac_bins_retained", "median")
            if "frac_bins_retained" in chrom.columns else ("copy_number", lambda _: np.nan),
        ).reset_index()
        metric_frames.append(_add_batch_columns(metrics, run))

    sample_df = pd.concat(pred_frames, ignore_index=True, sort=False)
    metrics_df = pd.concat(metric_frames, ignore_index=True, sort=False)
    drop_cols = [
        col for col in ["batch_id", "batch_label", "work_dir", "sample"]
        if col in metrics_df.columns
    ]
    sample_df = sample_df.merge(
        metrics_df.drop(columns=drop_cols),
        on="sample_key",
        how="left",
    )

    if "autosomal_baseline_cn" not in sample_df.columns:
        sample_df["autosomal_baseline_cn"] = 2
    sample_df["autosomal_baseline_cn"] = pd.to_numeric(
        sample_df["autosomal_baseline_cn"],
        errors="coerce",
    ).fillna(2).astype(int)
    if "baseline_ploidy_type" not in sample_df.columns:
        sample_df["baseline_ploidy_type"] = sample_df["autosomal_baseline_cn"].map(
            {1: "HAPLOID", 2: "DIPLOID", 3: "TRIPLOID", 4: "TETRAPLOID"}
        ).fillna("DIPLOID")
    if "baseline_cn_call" in sample_df.columns:
        has_manifest_call = sample_df["baseline_cn_call"].notna()
        sample_df.loc[has_manifest_call, "baseline_ploidy_type"] = (
            sample_df.loc[has_manifest_call, "baseline_cn_call"].astype(str)
        )

    for col in ("sample_depth_ratio", "sample_overdispersion_map", "sample_depth_map"):
        if col not in sample_df.columns:
            sample_df[col] = np.nan
        sample_df[col] = pd.to_numeric(sample_df[col], errors="coerce")

    sample_df["sample_depth_percentile"] = sample_df.groupby("batch_label")[
        "sample_depth_ratio"
    ].transform(_batch_percentile)
    missing_depth_ratio = sample_df["sample_depth_percentile"].isna()
    if missing_depth_ratio.any():
        sample_df.loc[missing_depth_ratio, "sample_depth_percentile"] = (
            sample_df.loc[missing_depth_ratio]
            .groupby("batch_label")["sample_depth_map"]
            .transform(_batch_percentile)
        )
    sample_df["sample_overdispersion_percentile"] = sample_df.groupby("batch_label")[
        "sample_overdispersion_map"
    ].transform(_batch_percentile)
    return sample_df


def _build_chrom_table(runs: list[RunData]) -> pd.DataFrame:
    """Build one row per sample-contig per batch."""
    frames: list[pd.DataFrame] = []
    for run in runs:
        chrom = _add_batch_columns(run.chrom_df, run)
        chrom["chrom_stats_source"] = str(run.chrom_stats_source)
        chrom["used_filtered_chrom_stats"] = run.used_filtered_chrom_stats
        frames.append(chrom)
    chrom_df = pd.concat(frames, ignore_index=True, sort=False)
    if "autosomal_baseline_cn" not in chrom_df.columns:
        chrom_df["autosomal_baseline_cn"] = 2
    chrom_df["autosomal_baseline_cn"] = pd.to_numeric(
        chrom_df["autosomal_baseline_cn"],
        errors="coerce",
    ).fillna(2).astype(int)
    chrom_df["copy_number"] = pd.to_numeric(chrom_df["copy_number"], errors="coerce")
    chrom_df["mean_cn_probability"] = pd.to_numeric(
        chrom_df["mean_cn_probability"],
        errors="coerce",
    )
    chrom_df["plq"] = pd.to_numeric(chrom_df["plq"], errors="coerce")
    if chrom_df["is_aneuploid"].dtype == object:
        chrom_df["is_aneuploid"] = (
            chrom_df["is_aneuploid"].astype(str).str.lower().isin(["true", "1", "yes"])
        )
    else:
        chrom_df["is_aneuploid"] = chrom_df["is_aneuploid"].fillna(False).astype(bool)
    return chrom_df


def _is_no_aneuploidy_label(value: Any) -> bool:
    if pd.isna(value):
        return True
    return str(value) in _NO_ANEUPLOIDY_LABELS


def _is_confident_sex_aneuploidy(row: pd.Series) -> bool:
    if not _is_no_aneuploidy_label(row.get("allosomal_aneuploidy_type")):
        return True
    sex = str(row.get("sex", ""))
    baseline = int(row.get("autosomal_baseline_cn", 2))
    baseline_type = str(row.get("baseline_ploidy_type", "DIPLOID"))
    return (
        baseline == 2 and
        baseline_type == "DIPLOID" and
        sex not in {"", "nan", "NaN"} and
        sex not in _NORMAL_SEX_LABELS
    )


def _is_confident_polyploidy(row: pd.Series) -> bool:
    baseline = int(row.get("autosomal_baseline_cn", 2))
    baseline_type = str(row.get("baseline_ploidy_type", "DIPLOID"))
    return baseline in {1, 3, 4} or baseline_type in {"HAPLOID", "TRIPLOID", "TETRAPLOID"}


def _is_confident_autosomal_aneuploidy(row: pd.Series) -> bool:
    return not _is_no_aneuploidy_label(row.get("autosomal_aneuploidy_type"))


def _expected_allosome_cn(row: pd.Series, chrom: str) -> int | None:
    """Infer expected chrX/chrY CN for normal sex labels."""
    sex = str(row.get("sex", ""))
    baseline = int(row.get("autosomal_baseline_cn", 2))
    female_like, male_like = _expected_allosome_copy_number_pairs(baseline)
    if sex == "FEMALE" or sex.endswith("_FEMALE"):
        return female_like[0] if chrom == "chrX" else female_like[1]
    if sex == "MALE" or sex.endswith("_MALE"):
        return male_like[0] if chrom == "chrX" else male_like[1]
    return None


def _annotate_expected_copy_number(
    chrom_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach expected CN and deviation flags to chromosome rows."""
    sample_cols = [
        "sample_key",
        "sex",
        "autosomal_baseline_cn",
        "baseline_ploidy_type",
        "sample_depth_ratio",
        "sample_depth_percentile",
        "sample_overdispersion_map",
        "sample_overdispersion_percentile",
        "score",
        "predicted_aneuploidy_type",
    ]
    sample_cols = [col for col in sample_cols if col in sample_df.columns]
    out = chrom_df.merge(
        sample_df[sample_cols].drop_duplicates("sample_key"),
        on="sample_key",
        how="left",
        suffixes=("", "__sample"),
    )
    expected: list[float] = []
    for _, row in out.iterrows():
        chrom = str(row["chromosome"])
        if chrom in _SEX_CHROMS:
            sex_expected = _expected_allosome_cn(row, chrom)
            expected.append(float("nan") if sex_expected is None else float(sex_expected))
        else:
            expected.append(float(row.get("autosomal_baseline_cn", 2)))
    out["expected_copy_number"] = expected
    out["copy_number_delta"] = out["copy_number"] - out["expected_copy_number"]
    out["copy_number_differs_from_expected"] = (
        out["expected_copy_number"].notna() &
        out["copy_number"].notna() &
        (out["copy_number"] != out["expected_copy_number"])
    )
    return out


def _format_contigs(events: pd.DataFrame) -> str:
    """Return compact chromosome/CN/probability text for case tables."""
    if events.empty:
        return ""
    parts: list[str] = []
    for _, event in events.sort_values(["chromosome", "copy_number"]).iterrows():
        prob = event.get("mean_cn_probability")
        prob_text = "nan" if pd.isna(prob) else f"{float(prob):.3f}"
        parts.append(f"{event['chromosome']}:CN{int(event['copy_number'])} p={prob_text}")
    return "; ".join(parts)


def _build_event_table(
    annotated_chrom_df: pd.DataFrame,
    prob_threshold: float,
) -> pd.DataFrame:
    """Build one row per confident or low-confidence contig event."""
    rows: list[dict[str, Any]] = []
    for _, row in annotated_chrom_df.iterrows():
        chrom = str(row["chromosome"])
        is_autosome = chrom not in _SEX_CHROMS
        is_confident = bool(row.get("is_aneuploid", False))
        is_low_conf = bool(
            row.get("copy_number_differs_from_expected", False) and
            not is_confident and
            pd.notna(row.get("mean_cn_probability")) and
            float(row["mean_cn_probability"]) <= prob_threshold
        )
        if not is_confident and not is_low_conf:
            continue
        if is_low_conf:
            category = "low_confidence_aneuploidy"
            confidence = "low_confidence"
        elif is_autosome:
            category = "confident_autosomal_aneuploidy"
            confidence = "confident"
        else:
            category = "confident_sex_aneuploidy"
            confidence = "confident"
        rows.append(
            {
                "batch_id": row["batch_id"],
                "batch_label": row["batch_label"],
                "sample": row["sample"],
                "sample_key": row["sample_key"],
                "category": category,
                "confidence": confidence,
                "chromosome": chrom,
                "copy_number": row.get("copy_number"),
                "expected_copy_number": row.get("expected_copy_number"),
                "copy_number_delta": row.get("copy_number_delta"),
                "mean_cn_probability": row.get("mean_cn_probability"),
                "plq": row.get("plq"),
                "n_bins": row.get("n_bins", np.nan),
                "frac_bins_retained": row.get("frac_bins_retained", np.nan),
                "median_depth": row.get("median_depth", np.nan),
                "mean_depth": row.get("mean_depth", np.nan),
                "sample_depth_ratio": row.get("sample_depth_ratio", np.nan),
                "sample_depth_percentile": row.get("sample_depth_percentile", np.nan),
                "sample_overdispersion_map": row.get("sample_overdispersion_map", np.nan),
                "sample_overdispersion_percentile": row.get(
                    "sample_overdispersion_percentile",
                    np.nan,
                ),
                "sample_score": row.get("score", np.nan),
                "predicted_aneuploidy_type": row.get("predicted_aneuploidy_type", ""),
            }
        )
    return pd.DataFrame(rows, columns=_EVENT_COLUMNS)


def _build_case_table(sample_df: pd.DataFrame, event_df: pd.DataFrame) -> pd.DataFrame:
    """Build one row per sample section entry."""
    rows: list[dict[str, Any]] = []
    event_groups = {
        (sample_key, category): group
        for (sample_key, category), group in event_df.groupby(["sample_key", "category"])
    } if not event_df.empty else {}
    for _, sample in sample_df.iterrows():
        categories: list[str] = []
        sample_key = str(sample["sample_key"])
        if (
            _is_confident_sex_aneuploidy(sample) or
            (sample_key, "confident_sex_aneuploidy") in event_groups
        ):
            categories.append("confident_sex_aneuploidy")
        if _is_confident_polyploidy(sample):
            categories.append("confident_polyploidy")
        if (
            _is_confident_autosomal_aneuploidy(sample) or
            (sample_key, "confident_autosomal_aneuploidy") in event_groups
        ):
            categories.append("confident_autosomal_aneuploidy")
        if (sample_key, "low_confidence_aneuploidy") in event_groups:
            categories.append("low_confidence_aneuploidy")

        for category in categories:
            events = event_groups.get((str(sample["sample_key"]), category), pd.DataFrame())
            rows.append(
                {
                    "category": category,
                    "batch_id": sample["batch_id"],
                    "batch_label": sample["batch_label"],
                    "sample": sample["sample"],
                    "sample_key": sample["sample_key"],
                    "sex": sample.get("sex", ""),
                    "predicted_aneuploidy_type": sample.get("predicted_aneuploidy_type", ""),
                    "autosomal_aneuploidy_type": sample.get("autosomal_aneuploidy_type", ""),
                    "allosomal_aneuploidy_type": sample.get("allosomal_aneuploidy_type", ""),
                    "baseline_ploidy_type": sample.get("baseline_ploidy_type", ""),
                    "autosomal_baseline_cn": sample.get("autosomal_baseline_cn", np.nan),
                    "score": sample.get("score", np.nan),
                    "sample_depth_ratio": sample.get("sample_depth_ratio", np.nan),
                    "sample_depth_percentile": sample.get("sample_depth_percentile", np.nan),
                    "sample_overdispersion_map": sample.get("sample_overdispersion_map", np.nan),
                    "sample_overdispersion_percentile": sample.get(
                        "sample_overdispersion_percentile",
                        np.nan,
                    ),
                    "median_frac_bins_retained": sample.get("median_frac_bins_retained", np.nan),
                    "used_filtered_chrom_stats": sample.get("used_filtered_chrom_stats", False),
                    "true_aneuploidy_type": sample.get("true_aneuploidy_type", ""),
                    "anomalous_contigs": _format_contigs(events),
                    "n_anomalous_contigs": int(len(events)),
                }
            )
    return pd.DataFrame(rows, columns=_CASE_COLUMNS)


def _build_summary_table(
    sample_df: pd.DataFrame,
    case_df: pd.DataFrame,
    event_df: pd.DataFrame,
    runs: list[RunData],
) -> pd.DataFrame:
    """Build a compact key/value summary table."""
    confident_auto = event_df[event_df["category"] == "confident_autosomal_aneuploidy"]
    low_conf = event_df[event_df["category"] == "low_confidence_aneuploidy"]
    rows: list[dict[str, Any]] = [
        {"metric": "n_samples", "value": int(sample_df["sample_key"].nunique())},
        {"metric": "n_batches", "value": len(runs)},
        {"metric": "n_male", "value": int((sample_df["sex"] == "MALE").sum())},
        {"metric": "n_female", "value": int((sample_df["sex"] == "FEMALE").sum())},
        {
            "metric": "n_confident_sex_aneuploidy_samples",
            "value": int((case_df["category"] == "confident_sex_aneuploidy").sum())
            if not case_df.empty else 0,
        },
        {
            "metric": "n_confident_polyploidy_samples",
            "value": int((case_df["category"] == "confident_polyploidy").sum())
            if not case_df.empty else 0,
        },
        {
            "metric": "n_confident_autosomal_aneuploidy_samples",
            "value": int((case_df["category"] == "confident_autosomal_aneuploidy").sum())
            if not case_df.empty else 0,
        },
        {
            "metric": "n_low_confidence_aneuploidy_samples",
            "value": int((case_df["category"] == "low_confidence_aneuploidy").sum())
            if not case_df.empty else 0,
        },
        {"metric": "n_confident_autosomal_aneuploidy_events", "value": int(len(confident_auto))},
        {"metric": "n_low_confidence_aneuploidy_events", "value": int(len(low_conf))},
    ]
    for chromosome, count in confident_auto.groupby("chromosome").size().items():
        rows.append({"metric": f"n_confident_autosomal_events_{chromosome}", "value": int(count)})
    return pd.DataFrame(rows)


def _load_runs(args: argparse.Namespace) -> list[RunData]:
    labels = args.batch_label or _default_batch_labels(args.work_dirs)
    return [
        _load_run_data(
            work_dir,
            batch_id=index,
            batch_label=label,
            binq_field=args.binq_field,
        )
        for index, (work_dir, label) in enumerate(zip(args.work_dirs, labels), start=1)
    ]


def _build_report_tables(
    runs: list[RunData],
    prob_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return summary, case, event, missing-artifact, and chromosome tables."""
    sample_df = _build_sample_table(runs)
    chrom_df = _build_chrom_table(runs)
    annotated_chrom_df = _annotate_expected_copy_number(chrom_df, sample_df)
    event_df = _build_event_table(annotated_chrom_df, prob_threshold)
    case_df = _build_case_table(sample_df, event_df)
    summary_df = _build_summary_table(sample_df, case_df, event_df, runs)
    missing_rows = [row for run in runs for row in run.missing_artifacts]
    missing_df = pd.DataFrame(
        missing_rows,
        columns=["batch_label", "artifact", "path", "reason"],
    )
    return summary_df, case_df, event_df, missing_df, annotated_chrom_df


def _write_sidecars(
    output_dir: Path,
    summary_df: pd.DataFrame,
    case_df: pd.DataFrame,
    event_df: pd.DataFrame,
    missing_df: pd.DataFrame,
) -> list[Path]:
    """Write machine-readable aggregate sidecar tables."""
    outputs = {
        "aggregate_summary.tsv": summary_df,
        "aggregate_cases.tsv": case_df,
        "aggregate_contig_events.tsv": event_df,
        "aggregate_missing_artifacts.tsv": missing_df,
    }
    paths: list[Path] = []
    for name, df in outputs.items():
        path = output_dir / name
        df.to_csv(path, sep="\t", index=False)
        paths.append(path)
    return paths


def _format_value(value: Any) -> str:
    """Format a scalar for PDF text/table display."""
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=_report_figure_size(150))
    fig.text(0.06, 0.94, title, fontsize=13, fontweight="bold", ha="left", va="top")
    y = 0.88
    for line in lines:
        fig.text(0.06, y, line, fontsize=8, ha="left", va="top", wrap=True)
        y -= 0.035
        if y < 0.06:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=_report_figure_size(150))
            y = 0.94
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_table_page(
    pdf: PdfPages,
    title: str,
    df: pd.DataFrame,
    columns: list[str],
    *,
    max_rows: int = 24,
) -> None:
    fig, ax = plt.subplots(figsize=_report_figure_size(150))
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=12)
    display = df.loc[:, [col for col in columns if col in df.columns]].head(max_rows).copy()
    if display.empty:
        ax.text(0.02, 0.92, "No rows", fontsize=9, ha="left", va="top")
    else:
        display = display.applymap(_format_value)
        table = ax.table(
            cellText=display.to_numpy().tolist(),
            colLabels=display.columns.tolist(),
            loc="upper left",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1.0, 1.2)
        if len(df) > max_rows:
            ax.text(
                0.02,
                0.03,
                f"Showing {max_rows} of {len(df)} rows. See sidecar TSV for full table.",
                fontsize=7,
                ha="left",
                va="bottom",
            )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _run_lookup(runs: list[RunData]) -> dict[tuple[int, str], RunData]:
    return {(run.batch_id, run.batch_label): run for run in runs}


def _sample_plot_to_pdf(
    pdf: PdfPages,
    run: RunData,
    sample_row: pd.Series,
    event_rows: pd.DataFrame,
    min_het_alt: int,
) -> bool:
    """Append the existing per-sample diagnostic plot to the aggregate PDF."""
    if run.bin_df is None or run.bin_df.empty:
        return False
    sample_data = run.bin_df[run.bin_df["sample"].astype(str) == str(sample_row["sample"])]
    if sample_data.empty:
        return False

    from gatk_sv_ploidy._plot_detail import plot_sample_with_variance

    all_vars = None
    if "sample_var" in run.bin_df.columns:
        all_vars = pd.to_numeric(run.bin_df["sample_var"], errors="coerce").dropna().unique()
    sample_idx_map = None
    if run.site_data is not None and "sample_ids" in run.site_data:
        sample_idx_map = {str(sample): idx for idx, sample in enumerate(run.site_data["sample_ids"])}
    chromosome_plq_map = {
        str(row["chromosome"]): float(row["plq"])
        for _, row in run.chrom_df[run.chrom_df["sample"].astype(str) == str(sample_row["sample"])].iterrows()
        if "plq" in row.index and pd.notna(row["plq"])
    }
    aneuploid_chrs = [
        (str(row["chromosome"]), int(row["copy_number"]), float(row["mean_cn_probability"]))
        for _, row in event_rows.iterrows()
        if pd.notna(row.get("copy_number")) and pd.notna(row.get("mean_cn_probability"))
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_sample_with_variance(
            sample_data,
            all_vars,
            tmpdir,
            aneuploid_chrs=aneuploid_chrs,
            baseline_ploidy_type=str(sample_row.get("baseline_ploidy_type", "DIPLOID")),
            autosomal_baseline_cn=int(sample_row.get("autosomal_baseline_cn", 2)),
            site_data=run.site_data,
            sample_idx_map=sample_idx_map,
            chromosome_plq_map=chromosome_plq_map,
            min_het_alt=min_het_alt,
            detail_note=str(sample_row.get("category", "")).replace("_", " "),
        )
        safe = str(sample_row["sample"]).replace("/", "_").replace(" ", "_")
        image_path = Path(tmpdir) / "sample_plots" / f"{safe}.png"
        if not image_path.exists():
            return False
        image = plt.imread(image_path)
        fig, ax = plt.subplots(figsize=_report_figure_size(150))
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"{sample_row['sample_key']} diagnostic plot",
            fontsize=10,
            fontweight="bold",
            loc="left",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    return True


def _add_case_pages(
    pdf: PdfPages,
    case_df: pd.DataFrame,
    event_df: pd.DataFrame,
    runs: list[RunData],
    min_het_alt: int,
) -> None:
    run_by_key = _run_lookup(runs)
    for category, title in _CASE_SECTIONS:
        section_cases = case_df[case_df["category"] == category] if not case_df.empty else pd.DataFrame()
        _add_text_page(
            pdf,
            title,
            [f"Samples in section: {len(section_cases)}"],
        )
        for _, case in section_cases.sort_values(["batch_label", "sample"]).iterrows():
            events = event_df[
                (event_df["sample_key"] == case["sample_key"]) &
                (event_df["category"] == category)
            ] if not event_df.empty else pd.DataFrame()
            lines = [
                f"Batch: {case['batch_label']}",
                f"Sample: {case['sample']}",
                f"Predicted type: {_format_value(case.get('predicted_aneuploidy_type'))}",
                f"Sex: {_format_value(case.get('sex'))}",
                f"Baseline: {_format_value(case.get('baseline_ploidy_type'))} (CN={_format_value(case.get('autosomal_baseline_cn'))})",
                f"Score: {_format_value(case.get('score'))}",
                f"Sample depth ratio: {_format_value(case.get('sample_depth_ratio'))} (batch percentile {_format_value(case.get('sample_depth_percentile'))})",
                f"Sample overdispersion: {_format_value(case.get('sample_overdispersion_map'))} (batch percentile {_format_value(case.get('sample_overdispersion_percentile'))})",
                f"Median retained-bin fraction: {_format_value(case.get('median_frac_bins_retained'))}",
                f"Anomalous contigs: {_format_value(case.get('anomalous_contigs'))}",
                f"Truth label: {_format_value(case.get('true_aneuploidy_type'))}",
            ]
            _add_text_page(pdf, f"{title}: {case['sample_key']}", lines)
            if not events.empty:
                _add_table_page(
                    pdf,
                    f"Contig metrics: {case['sample_key']}",
                    events,
                    [
                        "chromosome",
                        "copy_number",
                        "expected_copy_number",
                        "mean_cn_probability",
                        "plq",
                        "frac_bins_retained",
                        "median_depth",
                    ],
                    max_rows=20,
                )
            run = run_by_key.get((int(case["batch_id"]), str(case["batch_label"])))
            if run is None:
                continue
            plotted = _sample_plot_to_pdf(pdf, run, case, events, min_het_alt)
            if not plotted:
                _add_text_page(
                    pdf,
                    f"Diagnostic plot unavailable: {case['sample_key']}",
                    ["Per-bin statistics were not available for this run/sample."],
                )


def _write_pdf_report(
    report_path: Path,
    runs: list[RunData],
    summary_df: pd.DataFrame,
    case_df: pd.DataFrame,
    event_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    prob_threshold: float,
    min_het_alt: int,
) -> None:
    """Write the aggregate multi-page PDF report."""
    _apply_report_theme()
    with PdfPages(report_path) as pdf:
        lines = [
            "gatk-sv-ploidy aggregate report",
            f"Runs: {len(runs)}",
            f"Low-confidence threshold: mean_cn_probability <= {prob_threshold:.3f}",
            "Each input work directory is treated as one batch.",
            "Filtered chromosome stats are used when call/chromosome_stats.filtered.tsv is present.",
        ]
        for run in runs:
            source = "filtered" if run.used_filtered_chrom_stats else "infer"
            lines.append(f"{run.batch_label}: {source} chromosome stats, {run.work_dir}")
        _add_text_page(pdf, "Aggregate Report", lines)
        _add_table_page(pdf, "Summary", summary_df, ["metric", "value"], max_rows=80)
        _add_table_page(
            pdf,
            "Case Index",
            case_df,
            [
                "category",
                "batch_label",
                "sample",
                "predicted_aneuploidy_type",
                "anomalous_contigs",
                "score",
            ],
            max_rows=30,
        )
        if not missing_df.empty:
            _add_table_page(
                pdf,
                "Missing Optional Artifacts",
                missing_df,
                ["batch_label", "artifact", "reason"],
                max_rows=30,
            )
        _add_case_pages(pdf, case_df, event_df, runs, min_het_alt)


def _run_aggregate(args: argparse.Namespace, logger) -> None:
    """Run aggregation after CLI logging is configured."""
    _validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / args.output_name
    runs = _load_runs(args)
    summary_df, case_df, event_df, missing_df, _ = _build_report_tables(
        runs,
        prob_threshold=float(args.prob_threshold),
    )
    sidecar_paths = _write_sidecars(output_dir, summary_df, case_df, event_df, missing_df)
    _write_pdf_report(
        report_path,
        runs,
        summary_df,
        case_df,
        event_df,
        missing_df,
        prob_threshold=float(args.prob_threshold),
        min_het_alt=int(args.min_het_alt),
    )
    logger.info(
        "Aggregate report complete: runs=%d cases=%d contig_events=%d prob_threshold=%.3f",
        len(args.work_dirs),
        len(case_df),
        len(event_df),
        float(args.prob_threshold),
    )
    log_output_artifacts(logger, [report_path, *sidecar_paths])


def main() -> None:
    """Entry point for ``gatk-sv-ploidy aggregate``."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with tool_logging_context(
        tool_name="aggregate",
        output_dir=args.output_dir,
        args=args,
    ) as logger:
        _run_aggregate(args, logger)


if __name__ == "__main__":
    main()