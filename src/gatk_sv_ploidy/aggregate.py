"""Aggregate one or more gatk-sv-ploidy runs into a PDF report."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from gatk_sv_ploidy._logging import log_output_artifacts, tool_logging_context


_SEX_CHROMS = frozenset({"chrX", "chrY"})
_NORMAL_SEX_LABELS = frozenset({"MALE", "FEMALE"})
_NO_ANEUPLOIDY_LABELS = frozenset({"NONE", "NORMAL", "", "nan", "NaN"})
_COMPACT_TABLE_ROW_HEIGHT = 0.72
_TABLE_HORIZONTAL_PAD = 0.01
_TABLE_VERTICAL_PAD = 0.10
_APPENDIX_TOC_LABEL = "Appendix: Report Field Guide"
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
    "coverage_score",
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
    "coverage_score",
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
    call_prob_threshold: float | None
    truth_labels_provided: bool
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
        help=(
            "Fallback coverage score threshold used when a batch's call "
            "cutoff cannot be recovered from call.log"
        ),
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


def _load_optional_call_prob_threshold(
    work_dir: Path,
    batch_label: str,
    missing: list[dict[str, str]],
) -> float | None:
    """Recover the per-batch call probability cutoff from call.log when present."""
    path = work_dir / "call" / "call.log"
    if not path.exists():
        return None

    # TODO: replace ad-hoc log parsing with a machine-readable call metadata artifact.
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                marker = "Command arguments:"
                if marker not in line:
                    continue
                payload = line.split(marker, 1)[1].strip()
                args = json.loads(payload)
                value = args.get("prob_threshold")
                if value is None:
                    continue
                threshold = float(value)
                if 0.0 <= threshold <= 1.0:
                    return threshold
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        missing.append(_optional_missing(batch_label, "call_prob_threshold", path, str(exc)))
        return None

    missing.append(
        _optional_missing(
            batch_label,
            "call_prob_threshold",
            path,
            "prob_threshold not found in log",
        )
    )
    return None


def _load_truth_labels_provided(work_dir: Path) -> bool:
    """Return whether call.log records a non-empty truth_json argument."""
    path = work_dir / "call" / "call.log"
    if not path.exists():
        return False

    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                marker = "Command arguments:"
                if marker not in line:
                    continue
                payload = line.split(marker, 1)[1].strip()
                args = json.loads(payload)
                return bool(args.get("truth_json"))
    except (OSError, TypeError, json.JSONDecodeError):
        return False

    return False


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

    called_path = root / "call" / "chromosome_stats.tsv"
    filtered_path = root / "call" / "chromosome_stats.filtered.tsv"
    chrom_path = called_path if called_path.exists() else filtered_path
    if not chrom_path.exists():
        raise FileNotFoundError(
            f"Missing required called chromosome stats file: {called_path} or {filtered_path}"
        )
    chrom_df = _read_tsv(chrom_path)
    _validate_columns(chrom_df, _REQUIRED_CHROM_COLUMNS, chrom_path)

    missing: list[dict[str, str]] = []
    baseline_df = _load_optional_baseline(root, batch_label, missing)
    bin_df = _load_optional_bin_stats(root, batch_label, missing, binq_field)
    site_data = _load_optional_site_data(root, batch_label, missing)
    call_prob_threshold = _load_optional_call_prob_threshold(root, batch_label, missing)
    truth_labels_provided = _load_truth_labels_provided(root)

    return RunData(
        batch_id=batch_id,
        batch_label=batch_label,
        work_dir=root,
        pred_df=pred_df,
        chrom_df=chrom_df,
        chrom_stats_source=chrom_path,
        used_filtered_chrom_stats=filtered_path.exists(),
        baseline_df=baseline_df,
        bin_df=bin_df,
        site_data=site_data,
        call_prob_threshold=call_prob_threshold,
        truth_labels_provided=truth_labels_provided,
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
        if "true_aneuploidy_type" in pred.columns and not run.truth_labels_provided:
            pred["true_aneuploidy_type"] = ""
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
    chrom_df["coverage_score"] = pd.to_numeric(
        chrom_df["coverage_score"],
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
    """Return compact chromosome/CN/score text for case tables."""
    if events.empty:
        return ""
    parts: list[str] = []
    for _, event in events.sort_values(["chromosome", "copy_number"]).iterrows():
        prob = event.get("coverage_score")
        prob_text = "nan" if pd.isna(prob) else f"{float(prob):.3f}"
        parts.append(f"{event['chromosome']}:CN{int(event['copy_number'])} score={prob_text}")
    return "; ".join(parts)


def _resolved_call_prob_threshold(run: RunData, fallback_prob_threshold: float) -> float:
    """Return the batch-specific call cutoff, falling back to aggregate input."""
    if run.call_prob_threshold is not None:
        return float(run.call_prob_threshold)
    return float(fallback_prob_threshold)


def _build_event_table(
    annotated_chrom_df: pd.DataFrame,
    threshold_by_batch: dict[tuple[int, str], float],
    fallback_prob_threshold: float,
) -> pd.DataFrame:
    """Build one row per confident or low-confidence contig event."""
    rows: list[dict[str, Any]] = []
    for _, row in annotated_chrom_df.iterrows():
        chrom = str(row["chromosome"])
        batch_key = (int(row["batch_id"]), str(row["batch_label"]))
        prob_threshold = threshold_by_batch.get(batch_key, fallback_prob_threshold)
        is_autosome = chrom not in _SEX_CHROMS
        prob = row.get("coverage_score")
        is_above_threshold = pd.notna(prob) and float(prob) > prob_threshold
        is_confident = bool(row.get("is_aneuploid", False)) and is_above_threshold
        is_low_conf = bool(
            row.get("copy_number_differs_from_expected", False) and
            not is_confident and
            pd.notna(prob) and
            float(prob) <= prob_threshold
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
                "coverage_score": row.get("coverage_score"),
                "plq": row.get("plq"),
                "n_bins": row.get("n_bins", np.nan),
                "frac_bins_retained": row.get("frac_bins_retained", np.nan),
                "median_depth": row.get("plot_median_depth", row.get("median_depth", np.nan)),
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
        if (sample_key, "confident_sex_aneuploidy") in event_groups:
            categories.append("confident_sex_aneuploidy")
        if _is_confident_polyploidy(sample):
            categories.append("confident_polyploidy")
        if (sample_key, "confident_autosomal_aneuploidy") in event_groups:
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
    threshold_by_batch = {
        (run.batch_id, run.batch_label): _resolved_call_prob_threshold(run, prob_threshold)
        for run in runs
    }
    event_df = _build_event_table(
        annotated_chrom_df,
        threshold_by_batch,
        fallback_prob_threshold=float(prob_threshold),
    )
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


# -----------------------------------------------------------------------------
# PDF report rendering
#
# The pages below are laid out to resemble a clinical-style genome report:
# US-Letter portrait pages with thin top/bottom rules, a running header, a
# centered title block on the cover page, section bands with hairline rules,
# and lightly striped tables with no vertical gridlines.
# -----------------------------------------------------------------------------

_PAGE_SIZE_IN = (8.5, 11.0)  # US Letter portrait
_MARGIN_L = 0.75 / 8.5      # ~0.75 inch
_MARGIN_R = 1.0 - 0.75 / 8.5
_HEADER_RULE_Y = 1.0 - 0.65 / 11.0
_HEADER_TEXT_Y = 1.0 - 0.45 / 11.0
_FOOTER_RULE_Y = 0.55 / 11.0
_FOOTER_TEXT_Y = 0.35 / 11.0
_BODY_TOP = 1.0 - 0.85 / 11.0
_BODY_BOTTOM = 0.70 / 11.0

_INK = "#222222"
_RULE = "#808080"
_MUTED = "#5A5A5A"
_BAND = "#ECECEC"
_STRIPE = "#F5F5F5"

_REPORT_TITLE = "GATK-SV Ploidy Aggregate Report"


def _format_value(value: Any) -> str:
    """Format a scalar for PDF text/table display."""
    if pd.isna(value):
        return "\u2014"  # em-dash for missing values
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _new_page(
    pdf_state: dict[str, Any],
    *,
    header: str | None = None,
) -> plt.Figure:
    """Create a fresh letter-size figure with running header and footer."""
    fig = plt.figure(figsize=_PAGE_SIZE_IN)
    fig.patch.set_facecolor("white")
    pdf_state["page"] += 1
    page_num = pdf_state["page"]

    # Header
    if header:
        fig.text(
            _MARGIN_L, _HEADER_TEXT_Y, header,
            fontsize=8, color=_MUTED, ha="left", va="center",
            family="sans-serif",
        )
    fig.text(
        _MARGIN_R, _HEADER_TEXT_Y, _REPORT_TITLE,
        fontsize=8, color=_MUTED, ha="right", va="center",
        family="sans-serif",
    )
    fig.add_artist(Line2D(
        [_MARGIN_L, _MARGIN_R], [_HEADER_RULE_Y, _HEADER_RULE_Y],
        color=_RULE, linewidth=0.6, transform=fig.transFigure,
    ))

    # Footer
    fig.add_artist(Line2D(
        [_MARGIN_L, _MARGIN_R], [_FOOTER_RULE_Y, _FOOTER_RULE_Y],
        color=_RULE, linewidth=0.6, transform=fig.transFigure,
    ))
    fig.text(
        _MARGIN_L, _FOOTER_TEXT_Y, pdf_state.get("footer_left", ""),
        fontsize=7, color=_MUTED, ha="left", va="center",
    )
    fig.text(
        _MARGIN_R, _FOOTER_TEXT_Y, f"Page {page_num}",
        fontsize=7, color=_MUTED, ha="right", va="center",
    )
    return fig


def _save_page(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig)
    plt.close(fig)


def _section_band(fig: plt.Figure, y: float, title: str, *, eyebrow: str | None = None) -> float:
    """Draw a section header with eyebrow label, bold title, and a hairline rule.

    Returns the y-coordinate just below the band (in figure fraction).
    """
    width = _MARGIN_R - _MARGIN_L
    if eyebrow:
        fig.text(
            _MARGIN_L, y, eyebrow.upper(),
            fontsize=7, color=_MUTED, ha="left", va="top",
            family="sans-serif",
        )
        y -= 0.012
    fig.text(
        _MARGIN_L, y, title,
        fontsize=12, fontweight="bold", color=_INK, ha="left", va="top",
        family="sans-serif",
    )
    y -= 0.018
    fig.add_artist(Line2D(
        [_MARGIN_L, _MARGIN_L + width], [y, y],
        color=_INK, linewidth=0.8, transform=fig.transFigure,
    ))
    return y - 0.012


def _draw_kv_block(
    fig: plt.Figure,
    items: list[tuple[str, str]],
    *,
    start_y: float,
    columns: int = 2,
    line_height: float = 0.020,
    label_fraction: float = 0.55,
    col_widths: list[float] | None = None,
) -> float:
    """Render labelled key/value pairs in a clean multi-column layout."""
    if not items:
        return start_y
    total_width = _MARGIN_R - _MARGIN_L
    gutter = 0.015
    usable_width = total_width - gutter * (columns - 1)
    if col_widths is None:
        normalized_col_widths = [1.0 / columns] * columns
    else:
        total = sum(col_widths)
        normalized_col_widths = [width / total for width in col_widths]
    column_widths = [usable_width * width for width in normalized_col_widths]
    column_starts: list[float] = []
    x = _MARGIN_L
    for width in column_widths:
        column_starts.append(x)
        x += width + gutter
    label_widths = [width * label_fraction for width in column_widths]
    value_widths = [width - label_width for width, label_width in zip(column_widths, label_widths)]
    n = len(items)
    rows = (n + columns - 1) // columns
    label_chars_by_col = [_estimate_wrap_chars(fig, width, fontsize=8) for width in label_widths]
    value_chars_by_col = [_estimate_wrap_chars(fig, width, fontsize=8) for width in value_widths]
    wrapped_items: list[tuple[str, str]] = []
    row_line_counts = [1] * rows
    for idx, (label, value) in enumerate(items):
        col = idx // rows
        row = idx % rows
        wrapped_label = _wrap_table_text(label, label_chars_by_col[col])
        wrapped_value = _wrap_table_text(value, value_chars_by_col[col])
        row_line_counts[row] = max(
            row_line_counts[row],
            _wrapped_line_count(wrapped_label),
            _wrapped_line_count(wrapped_value),
        )
        wrapped_items.append((wrapped_label, wrapped_value))

    row_offsets: list[float] = []
    used_height = 0.0
    for line_count in row_line_counts:
        row_offsets.append(used_height)
        used_height += line_count * line_height

    for idx, (label, value) in enumerate(wrapped_items):
        col = idx // rows
        row = idx % rows
        x_label = column_starts[col]
        x_value = x_label + label_widths[col]
        y = start_y - row_offsets[row]
        row_height = row_line_counts[row] * line_height
        _draw_clipped_text_box(
            fig,
            x_label,
            y,
            label_widths[col],
            row_height,
            label,
            fontsize=8,
            color=_MUTED,
        )
        _draw_clipped_text_box(
            fig,
            x_value,
            y,
            value_widths[col],
            row_height,
            value,
            fontsize=8,
            color=_INK,
            fontweight="semibold",
        )
    return start_y - used_height - 0.004


def _draw_clipped_text_box(
    fig: plt.Figure,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    fontsize: int,
    color: str,
    fontweight: str | None = None,
) -> None:
    """Draw text clipped to a figure-coordinate box."""
    ax = fig.add_axes([x, max(0.0, y - height), width, height], frameon=False)
    ax.set_axis_off()
    artist = ax.text(
        0.0, 1.0, text,
        fontsize=fontsize, color=color, ha="left", va="top",
        family="sans-serif", fontweight=fontweight, transform=ax.transAxes,
        clip_on=True,
    )
    artist.set_clip_path(ax.patch)


def _estimate_wrap_chars(fig: plt.Figure, width_fraction: float, *, fontsize: float) -> int:
    """Estimate how many monospace-ish characters fit in a figure-width fraction."""
    available_points = max(fig.get_figwidth() * 72.0 * float(width_fraction), 1.0)
    approx_char_width = max(float(fontsize) * 0.72, 1.0)
    return max(1, int(available_points / approx_char_width))


def _draw_paragraph(
    fig: plt.Figure,
    text: str,
    *,
    start_y: float,
    fontsize: int = 8,
    color: str = _INK,
    line_height: float = 0.018,
) -> float:
    for line in text.split("\n"):
        fig.text(
            _MARGIN_L, start_y, line,
            fontsize=fontsize, color=color, ha="left", va="top",
            family="sans-serif", wrap=True,
        )
        start_y -= line_height
    return start_y


def _set_top_aligned_cell_text_position(cell, renderer) -> None:
    """Position wrapped table text from the top-left of each cell."""
    bbox = cell.get_window_extent(renderer)
    vertical_pad = getattr(cell, "_vertical_pad", cell.PAD)
    y = bbox.y1 - bbox.height * vertical_pad
    loc = cell._text.get_horizontalalignment()
    if loc == "center":
        x = bbox.x0 + bbox.width / 2
    elif loc == "left":
        x = bbox.x0 + bbox.width * cell.PAD
    else:
        x = bbox.x0 + bbox.width * (1 - cell.PAD)
    cell._text.set_position((x, y))


def _style_table(table, *, header_bg: str = _BAND, stripe_bg: str = _STRIPE) -> None:
    """Apply a clean clinical-report style to a matplotlib Table object."""
    cells = table.get_celld()
    if not cells:
        return
    row_indices = {key[0] for key in cells.keys()}
    col_indices = {key[1] for key in cells.keys()}
    n_cols = max(col_indices) + 1 if col_indices else 0
    for (row, col), cell in cells.items():
        cell.set_linewidth(0)
        cell.PAD = _TABLE_HORIZONTAL_PAD
        cell._vertical_pad = _TABLE_VERTICAL_PAD
        cell._set_text_position = MethodType(_set_top_aligned_cell_text_position, cell)
        text = cell.get_text()
        text.set_color(_INK)
        text.set_ha("left")
        text.set_va("top")
        text.set_wrap(True)
        if row == 0:
            cell.set_facecolor(header_bg)
            text.set_fontweight("bold")
            text.set_color(_INK)
        else:
            if (row - 1) % 2 == 1:
                cell.set_facecolor(stripe_bg)
            else:
                cell.set_facecolor("white")
        # Hairline top rule below header
        if row == 0:
            cell.visible_edges = "B"
            cell.set_edgecolor(_INK)
            cell.set_linewidth(0.6)
        elif row == max(row_indices):
            cell.visible_edges = "B"
            cell.set_edgecolor(_RULE)
            cell.set_linewidth(0.4)
        else:
            cell.visible_edges = ""
        _ = n_cols  # silence linter; reserved for future column-specific styling


def _wrap_table_text(text: str, max_chars: int) -> str:
    """Wrap ``text`` for narrow table cells, breaking long tokens when needed."""
    if max_chars <= 1:
        return text

    wrapped_lines: list[str] = []
    for line in str(text).splitlines() or [""]:
        parts = textwrap.wrap(
            line,
            width=max_chars,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=True,
        )
        if parts:
            wrapped_lines.extend(parts)
        else:
            wrapped_lines.append("")
    return "\n".join(wrapped_lines)


def _wrapped_line_count(text: str) -> int:
    return max(1, str(text).count("\n") + 1)


@dataclass
class TableRenderResult:
    bottom: float
    rows_rendered: int


def _draw_table(
    fig: plt.Figure,
    df: pd.DataFrame,
    columns: list[str],
    *,
    start_y: float,
    max_rows: int | None = 30,
    start_row: int = 0,
    col_widths: list[float] | None = None,
    font_size: int = 7,
    row_height: float = 0.92,
    note: str | None = None,
    column_labels: dict[str, str] | None = None,
) -> TableRenderResult:
    """Render a styled DataFrame table starting at ``start_y`` in figure coords.

    Returns the bottom y-coordinate and the number of rows rendered.
    """
    available_cols = [c for c in columns if c in df.columns]
    width = _MARGIN_R - _MARGIN_L
    min_bottom = _BODY_BOTTOM + 0.03
    row_unit_height = 0.018 * row_height
    if df.empty or not available_cols:
        fig.text(
            _MARGIN_L, start_y - 0.005, "No rows.",
            fontsize=8, color=_MUTED, ha="left", va="top", style="italic",
        )
        return TableRenderResult(start_y - 0.030, 0)

    if start_row >= len(df):
        return TableRenderResult(start_y, 0)

    usable_height = start_y - min_bottom
    max_rows_by_height = max(0, int(usable_height / row_unit_height) - 1)
    if max_rows is not None:
        max_rows_by_height = min(max_rows_by_height, max_rows)
    if max_rows_by_height <= 0:
        return TableRenderResult(start_y, 0)

    display = df.loc[:, available_cols].iloc[start_row:].copy()
    if max_rows is not None:
        display = display.head(max_rows)
    display = display.apply(lambda col: col.map(_format_value))

    if col_widths is None:
        col_widths = [1.0 / len(available_cols)] * len(available_cols)
    else:
        total = sum(col_widths)
        col_widths = [w / total for w in col_widths]

    if column_labels:
        display = display.rename(columns=column_labels)

    # Wrap cell contents to fit allocated column width (avoid neighbor overflow).
    # Empirical: DejaVu Sans table text at fontsize 7 fits closer to 55-60
    # characters in a half-page cell; keep a small margin for padding.
    char_per_unit = 0.0078 * (font_size / 7.0)
    max_chars_by_col: dict[str, int] = {}
    for col_name, frac in zip(display.columns, col_widths):
        col_fig_width = frac * width
        max_chars_by_col[col_name] = max(4, int(col_fig_width / char_per_unit))

    header_labels = [
        _wrap_table_text(col_name, max_chars_by_col[col_name]) for col_name in display.columns
    ]
    header_units = max(_wrapped_line_count(label) for label in header_labels)

    wrapped_rows: list[list[str]] = []
    row_units: list[int] = []
    consumed_units = header_units
    for _, row in display.iterrows():
        wrapped_row: list[str] = []
        current_units = 1
        for col_name in display.columns:
            wrapped_value = _wrap_table_text(row[col_name], max_chars_by_col[col_name])
            wrapped_row.append(wrapped_value)
            current_units = max(current_units, _wrapped_line_count(wrapped_value))
        if consumed_units + current_units > max(1, int(usable_height / row_unit_height)):
            break
        wrapped_rows.append(wrapped_row)
        row_units.append(current_units)
        consumed_units += current_units

    display_rows = len(wrapped_rows)
    if display_rows <= 0:
        return TableRenderResult(start_y, 0)

    total_row_units = header_units + sum(row_units)
    approx_height = row_unit_height * total_row_units
    bottom = max(start_y - approx_height, min_bottom)
    ax = fig.add_axes([_MARGIN_L, bottom, width, start_y - bottom])
    ax.axis("off")

    table = ax.table(
        cellText=wrapped_rows,
        colLabels=header_labels,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.0, 1.0, 1.0],
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    _style_table(table)

    total_units = header_units + sum(row_units)
    for row_index, units in enumerate([header_units, *row_units]):
        height = units / total_units
        for col_index in range(len(header_labels)):
            table[(row_index, col_index)].set_height(height)

    note_text = note
    if note_text:
        fig.text(
            _MARGIN_L, bottom - 0.009, note_text,
            fontsize=7, color=_MUTED, ha="left", va="top", style="italic",
        )
        bottom -= 0.014

    return TableRenderResult(bottom - 0.010, display_rows)


def _draw_table_across_pages(
    pdf: PdfPages,
    pdf_state: dict[str, Any],
    df: pd.DataFrame,
    columns: list[str],
    *,
    page_header: str | None,
    section_title: str,
    eyebrow: str | None,
    start_fig: plt.Figure,
    start_y: float,
    continuation_title: str | None = None,
    max_rows: int | None = 30,
    col_widths: list[float] | None = None,
    font_size: int = 7,
    row_height: float = 0.92,
    note: str | None = None,
    column_labels: dict[str, str] | None = None,
) -> tuple[plt.Figure, float]:
    """Render a table across as many report pages as needed."""
    fig = start_fig
    y = start_y
    row_start = 0
    continued_title = continuation_title or f"{section_title} (continued)"

    while row_start < len(df):
        result = _draw_table(
            fig,
            df,
            columns,
            start_y=y,
            max_rows=max_rows,
            start_row=row_start,
            col_widths=col_widths,
            font_size=font_size,
            row_height=row_height,
            note=note,
            column_labels=column_labels,
        )
        if result.rows_rendered == 0:
            _save_page(pdf, fig)
            fig = _new_page(pdf_state, header=page_header)
            y = _section_band(fig, _BODY_TOP, continued_title, eyebrow=eyebrow)
            continue
        row_start += result.rows_rendered
        y = result.bottom
        if row_start < len(df):
            _save_page(pdf, fig)
            fig = _new_page(pdf_state, header=page_header)
            y = _section_band(fig, _BODY_TOP, continued_title, eyebrow=eyebrow)

    return fig, y


def _plan_table_across_pages(
    pdf_state: dict[str, Any],
    df: pd.DataFrame,
    columns: list[str],
    *,
    page_header: str | None,
    section_title: str,
    eyebrow: str | None,
    start_fig: plt.Figure,
    start_y: float,
    continuation_title: str | None = None,
    max_rows: int | None = 30,
    col_widths: list[float] | None = None,
    font_size: int = 7,
    row_height: float = 0.92,
    note: str | None = None,
    column_labels: dict[str, str] | None = None,
) -> tuple[plt.Figure, float]:
    """Plan table pagination using the same layout rules without writing pages."""
    fig = start_fig
    y = start_y
    row_start = 0
    continued_title = continuation_title or f"{section_title} (continued)"

    while row_start < len(df):
        result = _draw_table(
            fig,
            df,
            columns,
            start_y=y,
            max_rows=max_rows,
            start_row=row_start,
            col_widths=col_widths,
            font_size=font_size,
            row_height=row_height,
            note=note,
            column_labels=column_labels,
        )
        if result.rows_rendered == 0:
            plt.close(fig)
            fig = _new_page(pdf_state, header=page_header)
            y = _section_band(fig, _BODY_TOP, continued_title, eyebrow=eyebrow)
            continue
        row_start += result.rows_rendered
        y = result.bottom
        if row_start < len(df):
            plt.close(fig)
            fig = _new_page(pdf_state, header=page_header)
            y = _section_band(fig, _BODY_TOP, continued_title, eyebrow=eyebrow)

    return fig, y


def _build_batch_inventory_rows(
    runs: list[RunData],
    prob_threshold: float,
) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "batch_id": run.batch_id,
            "batch_label": run.batch_label,
            "sample_count": int(run.pred_df["sample"].nunique()) if "sample" in run.pred_df.columns else int(len(run.pred_df)),
            "call_prob_threshold": _resolved_call_prob_threshold(run, prob_threshold),
            "work_dir": str(run.work_dir),
        }
        for run in runs
    ])


def _build_appendix_field_guide() -> pd.DataFrame:
    rows = [
        {
            "display_element": "Cover page - Summary",
            "displayed_label": "Purpose",
            "definition": "Fixed front-page key/value rows summarizing when the report was created and its batch/sample scope.",
        },
        {
            "display_element": "Cover page - Summary",
            "displayed_label": "Report generated",
            "definition": "Local timestamp when the aggregate PDF report was rendered.",
        },
        {
            "display_element": "Cover page - Summary",
            "displayed_label": "Batches analysed",
            "definition": "Number of batch work directories included in the aggregate report.",
        },
        {
            "display_element": "Cover page - Summary",
            "displayed_label": "Samples included",
            "definition": "Number of distinct sample_key values included in the aggregate report.",
        },
        {
            "display_element": "Table of Contents",
            "displayed_label": "Purpose",
            "definition": "Front-page linked index listing major report sections and their starting PDF page numbers.",
        },
        {
            "display_element": "Table of Contents",
            "displayed_label": "Section title",
            "definition": "Linked name of a report section or appendix.",
        },
        {
            "display_element": "Table of Contents",
            "displayed_label": "Page",
            "definition": "Starting PDF page number for the linked section.",
        },
        {
            "display_element": "Batch Inventory",
            "displayed_label": "Purpose",
            "definition": (
                "Front-matter table listing each aggregated run directory, its batch "
                "label, the number of loaded samples, the effective low-confidence "
                "call cutoff, and the source work directory."
            ),
        },
        {
            "display_element": "Batch Inventory",
            "displayed_label": "ID",
            "definition": "Integer batch identifier assigned by the aggregate report in load order.",
        },
        {
            "display_element": "Batch Inventory",
            "displayed_label": "Batch",
            "definition": (
                "User-visible batch label supplied on the command line or derived "
                "from the work directory name."
            ),
        },
        {
            "display_element": "Batch Inventory",
            "displayed_label": "Sample count",
            "definition": (
                "Number of distinct samples loaded from aneuploidy_type_predictions.tsv "
                "for the batch."
            ),
        },
        {
            "display_element": "Batch Inventory",
            "displayed_label": "Call cutoff",
            "definition": (
                "Effective per-batch probability threshold used to flag low-confidence "
                "contigs when coverage_score is at or below the cutoff."
            ),
        },
        {
            "display_element": "Batch Inventory",
            "displayed_label": "Work directory",
            "definition": (
                "Absolute path to the run_ploidy.sh work directory used as the source "
                "for this batch."
            ),
        },
        {
            "display_element": "Cohort Summary",
            "displayed_label": "Purpose",
            "definition": "Section 1 table of cohort-wide counts aggregated across all loaded batches.",
        },
        {
            "display_element": "Cohort Summary",
            "displayed_label": "Metric",
            "definition": "Name of the summarized cohort metric.",
        },
        {
            "display_element": "Cohort Summary",
            "displayed_label": "Value",
            "definition": "Count or scalar value for the corresponding metric.",
        },
        {
            "display_element": "Cohort Summary",
            "displayed_label": "Samples",
            "definition": "Number of distinct sample_key values included in the report.",
        },
        {
            "display_element": "Cohort Summary",
            "displayed_label": "Batches",
            "definition": "Number of aggregated batch work directories.",
        },
        {
            "display_element": "Cohort Summary",
            "displayed_label": "Reported male",
            "definition": "Count of samples whose input sex label is MALE.",
        },
        {
            "display_element": "Cohort Summary",
            "displayed_label": "Reported female",
            "definition": "Count of samples whose input sex label is FEMALE.",
        },
        {
            "display_element": "Cohort Summary",
            "displayed_label": "Confident sex aneuploidy (samples)",
            "definition": "Count of case samples categorized as confident sex-chromosome aneuploidy.",
        },
        {
            "display_element": "Cohort Summary",
            "displayed_label": "Confident polyploidy (samples)",
            "definition": "Count of case samples categorized as confident polyploidy.",
        },
        {
            "display_element": "Cohort Summary",
            "displayed_label": "Confident autosomal aneuploidy (samples)",
            "definition": "Count of case samples categorized as confident autosomal aneuploidy.",
        },
        {
            "display_element": "Cohort Summary",
            "displayed_label": "Low-confidence aneuploidy (samples)",
            "definition": "Count of case samples with at least one low-confidence contig event.",
        },
        {
            "display_element": "Cohort Summary",
            "displayed_label": "Confident autosomal events",
            "definition": "Total number of contig-level confident autosomal aneuploidy events across all samples.",
        },
        {
            "display_element": "Cohort Summary",
            "displayed_label": "Low-confidence events",
            "definition": (
                "Total number of contig-level low-confidence events, where copy number "
                "differs from expected and coverage_score is at or below the call cutoff."
            ),
        },
        {
            "display_element": "Cohort Summary",
            "displayed_label": "Confident autosomal events on <chromosome>",
            "definition": "Optional row pattern. Per-chromosome count of confident autosomal events for the named chromosome.",
        },
        {
            "display_element": "Case Index",
            "displayed_label": "Purpose",
            "definition": (
                "Section 2 table listing every reportable case once, with its review "
                "category and compact contig summary."
            ),
        },
        {
            "display_element": "Case Index",
            "displayed_label": "Category",
            "definition": (
                "Case grouping used for report sections: Sex aneuploidy, Polyploidy, "
                "Autosomal aneuploidy, or Low-confidence."
            ),
        },
        {
            "display_element": "Case Index",
            "displayed_label": "Batch",
            "definition": "Batch label for the case sample.",
        },
        {
            "display_element": "Case Index",
            "displayed_label": "Sample",
            "definition": "Sample identifier.",
        },
        {
            "display_element": "Case Index",
            "displayed_label": "Predicted",
            "definition": "Sample-level predicted_aneuploidy_type label.",
        },
        {
            "display_element": "Case Index",
            "displayed_label": "Anomalous contigs",
            "definition": (
                "Compact per-contig summary formatted as <chromosome>:CN<copy_number> "
                "score=<coverage_score> for reportable events in that case."
            ),
        },
        {
            "display_element": "Case detail - Identifiers",
            "displayed_label": "Purpose",
            "definition": "Fixed key/value rows shown at the top of every case page.",
        },
        {
            "display_element": "Case detail - Identifiers",
            "displayed_label": "Sample",
            "definition": "Sample identifier for the case.",
        },
        {
            "display_element": "Case detail - Identifiers",
            "displayed_label": "Batch",
            "definition": "Batch label for the case.",
        },
        {
            "display_element": "Case detail - Identifiers",
            "displayed_label": "Predicted sex",
            "definition": "Predicted sex label carried from the sample prediction table.",
        },
        {
            "display_element": "Case detail - Identifiers",
            "displayed_label": "Predicted type",
            "definition": "Sample-level predicted_aneuploidy_type label.",
        },
        {
            "display_element": "Case detail - Identifiers",
            "displayed_label": "Truth label",
            "definition": "Reference truth label when provided; blank otherwise.",
        },
        {
            "display_element": "Case detail - Ploidy and Call Metrics",
            "displayed_label": "Purpose",
            "definition": "Fixed key/value rows summarizing the sample-level ploidy interpretation and call outputs.",
        },
        {
            "display_element": "Case detail - Ploidy and Call Metrics",
            "displayed_label": "Baseline ploidy",
            "definition": "Autosomal baseline ploidy label together with the baseline autosomal copy number.",
        },
        {
            "display_element": "Case detail - Ploidy and Call Metrics",
            "displayed_label": "Score",
            "definition": (
                "Sample-level score, computed as the minimum contig-level "
                "coverage_score across the sample's chromosome summaries."
            ),
        },
        {
            "display_element": "Case detail - Ploidy and Call Metrics",
            "displayed_label": "Autosomal call",
            "definition": "Sample-level autosomal_aneuploidy_type label.",
        },
        {
            "display_element": "Case detail - Ploidy and Call Metrics",
            "displayed_label": "Allosomal call",
            "definition": "Sample-level allosomal_aneuploidy_type label.",
        },
        {
            "display_element": "Case detail - Ploidy and Call Metrics",
            "displayed_label": "Anomalous contigs",
            "definition": "Number of reportable contig events shown for the case.",
        },
        {
            "display_element": "Case detail - Sample QC",
            "displayed_label": "Purpose",
            "definition": "Fixed key/value rows summarizing sample-level depth and overdispersion quality-control metrics.",
        },
        {
            "display_element": "Case detail - Sample QC",
            "displayed_label": "Sample depth ratio",
            "definition": "Sample_depth_map divided by the cohort median sample_depth_map across finite positive samples.",
        },
        {
            "display_element": "Case detail - Sample QC",
            "displayed_label": "Depth percentile (batch)",
            "definition": "Within-batch percentile rank of Sample depth ratio.",
        },
        {
            "display_element": "Case detail - Sample QC",
            "displayed_label": "Sample overdispersion",
            "definition": (
                "Sample-level overdispersion MAP estimate carried from infer output "
                "(sample_var / sample_overdispersion_map)."
            ),
        },
        {
            "display_element": "Case detail - Sample QC",
            "displayed_label": "Overdispersion percentile",
            "definition": "Within-batch percentile rank of Sample overdispersion.",
        },
        {
            "display_element": "Anomalous Contig Evidence",
            "displayed_label": "Purpose",
            "definition": "Per-case table listing each reportable contig event on its own row.",
        },
        {
            "display_element": "Anomalous Contig Evidence",
            "displayed_label": "Chrom",
            "definition": "Chromosome name for the event.",
        },
        {
            "display_element": "Anomalous Contig Evidence",
            "displayed_label": "CN",
            "definition": "Called absolute copy number for the contig.",
        },
        {
            "display_element": "Anomalous Contig Evidence",
            "displayed_label": "Score",
            "definition": "Contig-level coverage_score for the displayed copy-number call.",
        },
        {
            "display_element": "Anomalous Contig Evidence",
            "displayed_label": "PLQ",
            "definition": (
                "Phred-scaled log-likelihood ratio between the most likely and second "
                "most likely ploidy states; larger values indicate clearer support."
            ),
        },
        {
            "display_element": "Anomalous Contig Evidence",
            "displayed_label": "Median normalized depth",
            "definition": (
                "Median plot-normalized per-bin depth across the contig; raw median depth "
                "is used only when normalized depth is unavailable."
            ),
        },
        {
            "display_element": "Report Field Guide",
            "displayed_label": "Purpose",
            "definition": "Appendix table defining each displayed report table, fixed row group, row label, and column label.",
        },
        {
            "display_element": "Report Field Guide",
            "displayed_label": "Table or row group",
            "definition": "Name of the report table, fixed key/value row group, or appendix section being defined.",
        },
        {
            "display_element": "Report Field Guide",
            "displayed_label": "Displayed row/column",
            "definition": "Exact visible row label, column label, or Purpose marker being defined.",
        },
        {
            "display_element": "Report Field Guide",
            "displayed_label": "Definition",
            "definition": "Precise meaning of the displayed row or column.",
        },
    ]
    return pd.DataFrame(rows, columns=["display_element", "displayed_label", "definition"])


@dataclass
class TocLinkSpec:
    target_page: int
    x0: float
    y0: float
    x1: float
    y1: float


def _draw_toc(
    fig: plt.Figure,
    toc_entries: list[tuple[str, int]],
    *,
    start_y: float,
    report_path: Path,
) -> tuple[float, list[TocLinkSpec]]:
    """Render a simple linked table of contents."""
    if not toc_entries:
        return start_y, []
    y = _section_band(fig, start_y, "Contents", eyebrow="Front matter")
    _ = report_path
    link_specs: list[TocLinkSpec] = []
    for label, page in toc_entries:
        fig.text(
            _MARGIN_L, y, label,
            fontsize=9, color="#0B57D0", ha="left", va="top",
            family="sans-serif",
        )
        fig.text(
            _MARGIN_R, y, str(page),
            fontsize=9, color="#0B57D0", ha="right", va="top",
            family="sans-serif",
        )
        underline_y = y - 0.014
        fig.add_artist(Line2D(
            [_MARGIN_L, _MARGIN_R], [underline_y, underline_y],
            color="#0B57D0", linewidth=0.6, transform=fig.transFigure,
        ))
        link_specs.append(
            TocLinkSpec(
                target_page=int(page),
                x0=_MARGIN_L,
                y0=y - 0.020,
                x1=_MARGIN_R,
                y1=y + 0.002,
            )
        )
        y -= 0.022
    return y - 0.006, link_specs


def _add_pdf_internal_links(
    pdf: PdfPages,
    source_page: int,
    link_specs: list[TocLinkSpec],
) -> None:
    """Attach internal PDF GoTo links to an existing page."""
    if not link_specs:
        return

    from matplotlib.backends.backend_pdf import Name

    pdf_file = pdf._ensure_file()
    if source_page < 1 or source_page > len(pdf_file._annotations):
        return

    page_annotations = pdf_file._annotations[source_page - 1][1]
    page_width = 72.0 * float(_PAGE_SIZE_IN[0])
    page_height = 72.0 * float(_PAGE_SIZE_IN[1])

    for spec in link_specs:
        if spec.target_page < 1 or spec.target_page > len(pdf_file.pageList):
            continue
        page_annotations.append(
            {
                "Type": Name("Annot"),
                "Subtype": Name("Link"),
                "Rect": [
                    page_width * spec.x0,
                    page_height * spec.y0,
                    page_width * spec.x1,
                    page_height * spec.y1,
                ],
                "Border": [0, 0, 0],
                "A": {
                    "S": Name("GoTo"),
                    "D": [pdf_file.pageList[spec.target_page - 1], Name("Fit")],
                },
            }
        )


def _plan_inventory_page_count(runs: list[RunData], prob_threshold: float) -> int:
    pdf_state: dict[str, Any] = {"page": 0, "footer_left": ""}
    fig = _new_page(pdf_state, header="Inputs  \u2022  Batch Inventory")
    y = _section_band(fig, _BODY_TOP, "Batch Inventory", eyebrow="Inputs")
    run_rows = _build_batch_inventory_rows(runs, prob_threshold)
    if not run_rows.empty:
        fig, _ = _plan_table_across_pages(
            pdf_state,
            run_rows,
            ["batch_id", "batch_label", "sample_count", "call_prob_threshold", "work_dir"],
            page_header="Inputs  \u2022  Batch Inventory",
            section_title="Batch Inventory",
            eyebrow="Inputs",
            start_fig=fig,
            start_y=y,
            max_rows=15,
            col_widths=[0.08, 0.36, 0.10, 0.10, 0.36],
            font_size=7,
            row_height=_COMPACT_TABLE_ROW_HEIGHT,
            column_labels={
                "batch_id": "ID",
                "batch_label": "Batch",
                "sample_count": "Sample count",
                "call_prob_threshold": "Call cutoff",
                "work_dir": "Work directory",
            },
        )
    plt.close(fig)
    return max(1, int(pdf_state["page"]))


def _plan_summary_section_pages(
    summary_df: pd.DataFrame,
    case_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    *,
    start_page: int,
) -> dict[str, int]:
    pdf_state: dict[str, Any] = {"page": start_page - 1, "footer_left": ""}
    pages: dict[str, int] = {}
    fig = _new_page(pdf_state, header="Section 1  \u2022  Cohort Summary")
    pages["Cohort Summary"] = int(pdf_state["page"])
    y = _section_band(fig, _BODY_TOP, "Cohort Summary", eyebrow="Section 1")

    if not summary_df.empty:
        display_summary = summary_df.copy()
        display_summary["metric"] = display_summary["metric"].map(_humanize_metric)
        fig, y = _plan_table_across_pages(
            pdf_state,
            display_summary,
            ["metric", "value"],
            page_header="Section 1  \u2022  Cohort Summary",
            section_title="Cohort Summary",
            eyebrow="Section 1",
            start_fig=fig,
            start_y=y,
            max_rows=30,
            col_widths=[0.72, 0.28],
            font_size=8,
            column_labels={"metric": "Metric", "value": "Value"},
        )

    pages["Case Index"] = int(pdf_state["page"])
    y = _section_band(fig, y - 0.010, "Case Index", eyebrow="Section 2")
    case_display = case_df.copy()
    if not case_display.empty and "category" in case_display.columns:
        case_display["category"] = case_display["category"].map(_humanize_category)
    if not case_display.empty:
        fig, y = _plan_table_across_pages(
            pdf_state,
            case_display,
            ["category", "batch_label", "sample", "predicted_aneuploidy_type",
             "anomalous_contigs"],
            page_header="Section 2  \u2022  Case Index",
            section_title="Case Index",
            eyebrow="Section 2",
            start_fig=fig,
            start_y=y,
            max_rows=22,
            col_widths=[0.194, 0.215, 0.141, 0.170, 0.280],
            font_size=7,
            row_height=_COMPACT_TABLE_ROW_HEIGHT,
            column_labels={
                "category": "Category",
                "batch_label": "Batch",
                "sample": "Sample",
                "predicted_aneuploidy_type": "Predicted",
                "anomalous_contigs": "Anomalous contigs",
            },
        )

    plt.close(fig)
    return pages


def _plan_case_page_count(
    case: pd.Series,
    events: pd.DataFrame,
    run: RunData | None,
    *,
    section_title: str,
    section_number: int,
) -> int:
    pdf_state: dict[str, Any] = {"page": 0, "footer_left": ""}
    fig = _new_page(
        pdf_state,
        header=f"Section {section_number}  \u2022  {section_title}",
    )

    y = _BODY_TOP
    y -= 0.014
    y -= 0.022
    y -= 0.015
    y = _draw_kv_block(
        fig,
        [
            ("Sample", str(case["sample"])),
            ("Batch", str(case["batch_label"])),
            ("Predicted sex", _format_value(case.get("sex"))),
            ("Predicted type", _format_value(case.get("predicted_aneuploidy_type"))),
            ("Truth label", _format_value(case.get("true_aneuploidy_type"))),
        ],
        start_y=y,
        columns=1,
        line_height=0.020,
        label_fraction=0.30,
    )
    baseline = (
        f"{_format_value(case.get('baseline_ploidy_type'))}"
        f" (CN={_format_value(case.get('autosomal_baseline_cn'))})"
    )
    y = _section_band(fig, y - 0.008, "Ploidy & Call Metrics", eyebrow="Findings")
    y = _draw_kv_block(
        fig,
        [
            ("Baseline ploidy", baseline),
            ("Score", _format_value(case.get("score"))),
            ("Autosomal call", _format_value(case.get("autosomal_aneuploidy_type"))),
            ("Allosomal call", _format_value(case.get("allosomal_aneuploidy_type"))),
            ("Anomalous contigs", str(int(case.get("n_anomalous_contigs", 0) or 0))),
        ],
        start_y=y,
        columns=1,
        line_height=0.020,
        label_fraction=0.30,
    )
    y = _section_band(fig, y - 0.008, "Sample QC", eyebrow="Quality")
    y = _draw_kv_block(
        fig,
        [
            ("Sample depth ratio", _format_value(case.get("sample_depth_ratio"))),
            ("Depth percentile (batch)", _format_value(case.get("sample_depth_percentile"))),
            ("Sample overdispersion", _format_value(case.get("sample_overdispersion_map"))),
            ("Overdispersion percentile", _format_value(case.get("sample_overdispersion_percentile"))),
        ],
        start_y=y,
        columns=1,
        line_height=0.020,
        label_fraction=0.30,
    )
    y = _section_band(fig, y - 0.008, "Anomalous Contig Evidence", eyebrow="Events")
    if not events.empty:
        fig, _ = _plan_table_across_pages(
            pdf_state,
            events,
            ["chromosome", "copy_number", "coverage_score", "plq", "median_depth"],
            page_header=f"Section {section_number}  \u2022  {section_title}",
            section_title=f"{case['sample']} \u2014 Anomalous Contig Evidence",
            eyebrow="Events",
            start_fig=fig,
            start_y=y,
            max_rows=18,
            col_widths=[0.18, 0.14, 0.24, 0.14, 0.30],
            font_size=7,
            row_height=_COMPACT_TABLE_ROW_HEIGHT,
            column_labels={
                "chromosome": "Chrom",
                "copy_number": "CN",
                "coverage_score": "Score",
                "plq": "PLQ",
                "median_depth": "Median normalized depth",
            },
        )
    plt.close(fig)
    if run is not None:
        plot_fig = _new_page(
            pdf_state,
            header=f"Section {section_number}  \u2022  {section_title} \u2014 Diagnostic",
        )
        plt.close(plot_fig)
    return int(pdf_state["page"])


def _plan_appendix_page_count() -> int:
    pdf_state: dict[str, Any] = {"page": 0, "footer_left": ""}
    fig = _new_page(pdf_state, header="Appendix  \u2022  Report Field Guide")
    y = _section_band(fig, _BODY_TOP, "Report Field Guide", eyebrow="Appendix")
    appendix_df = _build_appendix_field_guide()
    if not appendix_df.empty:
        fig, _ = _plan_table_across_pages(
            pdf_state,
            appendix_df,
            ["display_element", "displayed_label", "definition"],
            page_header="Appendix  \u2022  Report Field Guide",
            section_title="Report Field Guide",
            eyebrow="Appendix",
            start_fig=fig,
            start_y=y,
            max_rows=18,
            col_widths=[0.22, 0.20, 0.58],
            font_size=7,
            row_height=_COMPACT_TABLE_ROW_HEIGHT,
            column_labels={
                "display_element": "Table or row group",
                "displayed_label": "Displayed row/column",
                "definition": "Definition",
            },
        )
    plt.close(fig)
    return max(1, int(pdf_state["page"]))


def _build_report_toc_entries(
    runs: list[RunData],
    summary_df: pd.DataFrame,
    case_df: pd.DataFrame,
    event_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    prob_threshold: float,
) -> list[tuple[str, int]]:
    """Return front-matter TOC entries with planned start pages."""
    entries: list[tuple[str, int]] = [("Batch Inventory", 2)]
    inventory_pages = _plan_inventory_page_count(runs, prob_threshold)
    summary_start_page = 1 + inventory_pages + 1
    summary_pages = _plan_summary_section_pages(
        summary_df,
        case_df,
        missing_df,
        start_page=summary_start_page,
    )
    entries.append(("Cohort Summary", summary_pages["Cohort Summary"]))
    entries.append(("Case Index", summary_pages["Case Index"]))

    current_page = max(summary_pages.values())
    section_pages = _plan_summary_section_pages(
        summary_df,
        case_df,
        missing_df,
        start_page=summary_start_page,
    )
    current_page = max(section_pages.values())
    # Summary pages may span additional continuation pages beyond the section starts.
    summary_end_page = current_page
    current_page = max(summary_end_page, summary_start_page)

    # Re-plan precisely by simulating the entire summary block's final page number.
    summary_pdf_state: dict[str, Any] = {"page": summary_start_page - 1, "footer_left": ""}
    summary_fig = _new_page(summary_pdf_state, header="Section 1  \u2022  Cohort Summary")
    summary_y = _section_band(summary_fig, _BODY_TOP, "Cohort Summary", eyebrow="Section 1")
    if not summary_df.empty:
        display_summary = summary_df.copy()
        display_summary["metric"] = display_summary["metric"].map(_humanize_metric)
        summary_fig, summary_y = _plan_table_across_pages(
            summary_pdf_state,
            display_summary,
            ["metric", "value"],
            page_header="Section 1  \u2022  Cohort Summary",
            section_title="Cohort Summary",
            eyebrow="Section 1",
            start_fig=summary_fig,
            start_y=summary_y,
            max_rows=30,
            col_widths=[0.72, 0.28],
            font_size=8,
            column_labels={"metric": "Metric", "value": "Value"},
        )
    summary_y = _section_band(summary_fig, summary_y - 0.010, "Case Index", eyebrow="Section 2")
    case_display = case_df.copy()
    if not case_display.empty and "category" in case_display.columns:
        case_display["category"] = case_display["category"].map(_humanize_category)
    if not case_display.empty:
        summary_fig, summary_y = _plan_table_across_pages(
            summary_pdf_state,
            case_display,
            ["category", "batch_label", "sample", "predicted_aneuploidy_type", "anomalous_contigs"],
            page_header="Section 2  \u2022  Case Index",
            section_title="Case Index",
            eyebrow="Section 2",
            start_fig=summary_fig,
            start_y=summary_y,
            max_rows=22,
            col_widths=[0.194, 0.215, 0.141, 0.170, 0.280],
            font_size=7,
            row_height=_COMPACT_TABLE_ROW_HEIGHT,
            column_labels={
                "category": "Category",
                "batch_label": "Batch",
                "sample": "Sample",
                "predicted_aneuploidy_type": "Predicted",
                "anomalous_contigs": "Anomalous contigs",
            },
        )
    plt.close(summary_fig)
    current_page = int(summary_pdf_state["page"]) + 1

    run_by_key = _run_lookup(runs)
    section_number = 3
    for category, title in _CASE_SECTIONS:
        section_cases = (
            case_df[case_df["category"] == category]
            if not case_df.empty else pd.DataFrame()
        )
        if section_cases.empty:
            continue
        entries.append((title, current_page))
        current_page += 1
        for _, case in section_cases.sort_values(["batch_label", "sample"]).iterrows():
            events = (
                event_df[
                    (event_df["sample_key"] == case["sample_key"]) &
                    (event_df["category"] == category)
                ]
                if not event_df.empty else pd.DataFrame()
            )
            run = run_by_key.get((int(case["batch_id"]), str(case["batch_label"])))
            current_page += _plan_case_page_count(
                case,
                events,
                run,
                section_title=title,
                section_number=section_number,
            )
        section_number += 1

    entries.append((_APPENDIX_TOC_LABEL, current_page))

    return entries


def _add_cover_page(
    pdf: PdfPages,
    pdf_state: dict[str, Any],
    runs: list[RunData],
    summary_df: pd.DataFrame,
    toc_entries: list[tuple[str, int]],
    report_path: Path,
) -> list[TocLinkSpec]:
    """Render the cover page: title block, key counts, and table of contents."""
    fig = _new_page(pdf_state, header=None)

    # Title block (centered)
    fig.text(
        0.5, 0.83, _REPORT_TITLE,
        fontsize=22, fontweight="bold", color=_INK, ha="center", va="center",
        family="sans-serif",
    )
    fig.text(
        0.5, 0.79, "Whole-Chromosome Copy-Number & Aneuploidy Summary",
        fontsize=11, color=_MUTED, ha="center", va="center", style="italic",
    )
    # Decorative double rule
    for offset, lw in ((0.770, 1.2), (0.766, 0.4)):
        fig.add_artist(Line2D(
            [_MARGIN_L + 0.08, _MARGIN_R - 0.08], [offset, offset],
            color=_INK, linewidth=lw, transform=fig.transFigure,
        ))

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_samples_row = summary_df.loc[summary_df["metric"] == "n_samples", "value"]
    n_samples = int(n_samples_row.iloc[0]) if not n_samples_row.empty else 0

    y = _draw_kv_block(
        fig,
        [
            ("Report generated", generated),
            ("Batches analysed", str(len(runs))),
            ("Samples included", str(n_samples)),
        ],
        start_y=0.71,
        columns=2,
        line_height=0.024,
    )

    _, toc_link_specs = _draw_toc(
        fig,
        toc_entries,
        start_y=y - 0.020,
        report_path=report_path,
    )

    _save_page(pdf, fig)
    return toc_link_specs


def _add_inventory_pages(
    pdf: PdfPages,
    pdf_state: dict[str, Any],
    runs: list[RunData],
    prob_threshold: float,
) -> None:
    """Render the batch inventory on its own front-matter page(s)."""
    fig = _new_page(pdf_state, header="Inputs  \u2022  Batch Inventory")
    y = _section_band(fig, _BODY_TOP, "Batch Inventory", eyebrow="Inputs")
    run_rows = _build_batch_inventory_rows(runs, prob_threshold)
    if not run_rows.empty:
        fig, _ = _draw_table_across_pages(
            pdf,
            pdf_state,
            run_rows,
            ["batch_id", "batch_label", "sample_count", "call_prob_threshold", "work_dir"],
            page_header="Inputs  \u2022  Batch Inventory",
            section_title="Batch Inventory",
            eyebrow="Inputs",
            start_fig=fig,
            start_y=y,
            max_rows=15,
            col_widths=[0.08, 0.36, 0.10, 0.10, 0.36],
            font_size=7,
            row_height=_COMPACT_TABLE_ROW_HEIGHT,
            column_labels={
                "batch_id": "ID",
                "batch_label": "Batch",
                "sample_count": "Sample count",
                "call_prob_threshold": "Call cutoff",
                "work_dir": "Work directory",
            },
        )

    _save_page(pdf, fig)


_CATEGORY_DISPLAY = {
    "confident_sex_aneuploidy": "Sex aneuploidy",
    "confident_polyploidy": "Polyploidy",
    "confident_autosomal_aneuploidy": "Autosomal aneuploidy",
    "low_confidence_aneuploidy": "Low-confidence",
}


def _humanize_category(category: str) -> str:
    return _CATEGORY_DISPLAY.get(str(category), str(category).replace("_", " "))


def _humanize_metric(name: str) -> str:
    """Render a snake_case metric key as a clinical-report-friendly label."""
    overrides = {
        "n_samples": "Samples",
        "n_batches": "Batches",
        "n_male": "Reported male",
        "n_female": "Reported female",
        "n_confident_sex_aneuploidy_samples": "Confident sex aneuploidy (samples)",
        "n_confident_polyploidy_samples": "Confident polyploidy (samples)",
        "n_confident_autosomal_aneuploidy_samples": "Confident autosomal aneuploidy (samples)",
        "n_low_confidence_aneuploidy_samples": "Low-confidence aneuploidy (samples)",
        "n_confident_autosomal_aneuploidy_events": "Confident autosomal events",
        "n_low_confidence_aneuploidy_events": "Low-confidence events",
    }
    if name in overrides:
        return overrides[name]
    if name.startswith("n_confident_autosomal_events_"):
        chrom = name.replace("n_confident_autosomal_events_", "")
        return f"Confident autosomal events on {chrom}"
    return name.replace("_", " ").capitalize()


def _add_summary_page(
    pdf: PdfPages,
    pdf_state: dict[str, Any],
    summary_df: pd.DataFrame,
    case_df: pd.DataFrame,
    missing_df: pd.DataFrame,
) -> None:
    """Cohort-level summary, case index, and missing-artifact notice."""
    fig = _new_page(pdf_state, header="Section 1  \u2022  Cohort Summary")
    y = _section_band(fig, _BODY_TOP, "Cohort Summary", eyebrow="Section 1")

    if not summary_df.empty:
        display_summary = summary_df.copy()
        display_summary["metric"] = display_summary["metric"].map(_humanize_metric)
        fig, y = _draw_table_across_pages(
            pdf,
            pdf_state,
            display_summary,
            ["metric", "value"],
            page_header="Section 1  \u2022  Cohort Summary",
            section_title="Cohort Summary",
            eyebrow="Section 1",
            start_fig=fig,
            start_y=y,
            max_rows=30,
            col_widths=[0.72, 0.28],
            font_size=8,
            column_labels={"metric": "Metric", "value": "Value"},
        )

    # Case index
    y = _section_band(fig, y - 0.010, "Case Index", eyebrow="Section 2")
    case_display = case_df.copy()
    if not case_display.empty and "category" in case_display.columns:
        case_display["category"] = case_display["category"].map(_humanize_category)
    if not case_display.empty:
        fig, y = _draw_table_across_pages(
            pdf,
            pdf_state,
            case_display,
            ["category", "batch_label", "sample", "predicted_aneuploidy_type",
             "anomalous_contigs"],
            page_header="Section 2  \u2022  Case Index",
            section_title="Case Index",
            eyebrow="Section 2",
            start_fig=fig,
            start_y=y,
            max_rows=22,
            col_widths=[0.194, 0.215, 0.141, 0.170, 0.280],
            font_size=7,
            row_height=_COMPACT_TABLE_ROW_HEIGHT,
            column_labels={
                "category": "Category",
                "batch_label": "Batch",
                "sample": "Sample",
                "predicted_aneuploidy_type": "Predicted",
                "anomalous_contigs": "Anomalous contigs",
            },
        )
    _save_page(pdf, fig)


def _add_appendix_pages(pdf: PdfPages, pdf_state: dict[str, Any]) -> None:
    """Render the report field guide appendix."""
    fig = _new_page(pdf_state, header="Appendix  \u2022  Report Field Guide")
    y = _section_band(fig, _BODY_TOP, "Report Field Guide", eyebrow="Appendix")
    appendix_df = _build_appendix_field_guide()
    if not appendix_df.empty:
        fig, _ = _draw_table_across_pages(
            pdf,
            pdf_state,
            appendix_df,
            ["display_element", "displayed_label", "definition"],
            page_header="Appendix  \u2022  Report Field Guide",
            section_title="Report Field Guide",
            eyebrow="Appendix",
            start_fig=fig,
            start_y=y,
            max_rows=18,
            col_widths=[0.22, 0.20, 0.58],
            font_size=7,
            row_height=_COMPACT_TABLE_ROW_HEIGHT,
            column_labels={
                "display_element": "Table or row group",
                "displayed_label": "Displayed row/column",
                "definition": "Definition",
            },
        )
    _save_page(pdf, fig)


def _run_lookup(runs: list[RunData]) -> dict[tuple[int, str], RunData]:
    return {(run.batch_id, run.batch_label): run for run in runs}


def _category_title(category: str) -> str:
    for key, label in _CASE_SECTIONS:
        if key == category:
            return label
    return category.replace("_", " ").title()


def _sample_plot_safe_name(sample: Any) -> str:
    return str(sample).replace("/", "_").replace(" ", "_")


def _existing_sample_plot_image(run: RunData, sample: Any) -> Path | None:
    """Return an existing per-sample plot image from the run directory, if present."""
    safe_name = _sample_plot_safe_name(sample)
    candidate_dirs = [
        run.work_dir / "plot" / "sample_plots",
        run.work_dir / "sample_plots",
    ]
    for suffix in (".png", ".jpg", ".jpeg"):
        for sample_plot_dir in candidate_dirs:
            image_path = sample_plot_dir / f"{safe_name}{suffix}"
            if image_path.exists():
                return image_path
    return None


def _case_detail_note(case: pd.Series) -> str | None:
    """Return a short case note to display above the diagnostic plot."""
    category = str(case.get("category", "")).strip()
    if not category or category in _NO_ANEUPLOIDY_LABELS:
        return None
    return _humanize_category(category)


def _generate_sample_plot_image(
    run: RunData,
    sample_row: pd.Series,
    event_rows: pd.DataFrame,
    min_het_alt: int,
    tmpdir: str,
    logger: Any | None = None,
) -> Path | None:
    """Render the per-sample diagnostic image (returns path or None)."""
    sample_label = str(sample_row.get("sample_key", sample_row.get("sample", "unknown")))
    existing_image = _existing_sample_plot_image(run, sample_row["sample"])
    if existing_image is not None:
        if logger is not None:
            logger.info(
                "Case diagnostic plot: sample=%s batch=%s generate=no reason=reuse_existing_plot",
                sample_label,
                run.batch_label,
            )
        return existing_image

    if run.bin_df is None or run.bin_df.empty:
        if logger is not None:
            logger.info(
                "Case diagnostic plot: sample=%s batch=%s generate=no reason=missing_bin_stats",
                sample_label,
                run.batch_label,
            )
        return None
    sample_data = run.bin_df[run.bin_df["sample"].astype(str) == str(sample_row["sample"])]
    if sample_data.empty:
        if logger is not None:
            logger.info(
                "Case diagnostic plot: sample=%s batch=%s generate=no reason=sample_missing_from_bin_stats",
                sample_label,
                run.batch_label,
            )
        return None

    from gatk_sv_ploidy._plot_detail import plot_sample_with_variance

    all_vars = None
    if "sample_var" in run.bin_df.columns:
        all_vars = pd.to_numeric(run.bin_df["sample_var"], errors="coerce").dropna().unique()
    sample_idx_map = None
    if run.site_data is not None and "sample_ids" in run.site_data:
        sample_idx_map = {
            str(sample): idx for idx, sample in enumerate(run.site_data["sample_ids"])
        }
    chromosome_plq_map = {
        str(row["chromosome"]): float(row["plq"])
        for _, row in run.chrom_df[
            run.chrom_df["sample"].astype(str) == str(sample_row["sample"])
        ].iterrows()
        if "plq" in row.index and pd.notna(row["plq"])
    }
    aneuploid_chrs = [
        (str(row["chromosome"]), int(row["copy_number"]), float(row["coverage_score"]))
        for _, row in event_rows.iterrows()
        if pd.notna(row.get("copy_number")) and pd.notna(row.get("coverage_score"))
    ]
    if logger is not None:
        logger.info(
            "Case diagnostic plot: sample=%s batch=%s generate=yes reason=missing_existing_plot",
            sample_label,
            run.batch_label,
        )
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
    )
    safe = _sample_plot_safe_name(sample_row["sample"])
    image_path = Path(tmpdir) / "sample_plots" / f"{safe}.png"
    if logger is not None and not image_path.exists():
        logger.info(
            "Case diagnostic plot: sample=%s batch=%s generate=no reason=generated_image_missing",
            sample_label,
            run.batch_label,
        )
    return image_path if image_path.exists() else None


def _add_case_page(
    pdf: PdfPages,
    pdf_state: dict[str, Any],
    section_title: str,
    section_number: int,
    case: pd.Series,
    events: pd.DataFrame,
    run: RunData | None,
    min_het_alt: int,
    logger: Any | None = None,
) -> None:
    """Render a one-page (or two-page) case report for a single sample."""
    fig = _new_page(
        pdf_state,
        header=f"Section {section_number}  \u2022  {section_title}",
    )

    # Case header band
    y = _BODY_TOP
    fig.text(
        _MARGIN_L, y, "CASE",
        fontsize=7, color=_MUTED, ha="left", va="top",
    )
    y -= 0.014
    fig.text(
        _MARGIN_L, y, str(case["sample"]),
        fontsize=16, fontweight="bold", color=_INK, ha="left", va="top",
        family="sans-serif",
    )
    fig.text(
        _MARGIN_R, y, section_title,
        fontsize=10, color=_MUTED, ha="right", va="top", style="italic",
    )
    y -= 0.022
    fig.add_artist(Line2D(
        [_MARGIN_L, _MARGIN_R], [y, y],
        color=_INK, linewidth=0.8, transform=fig.transFigure,
    ))
    y -= 0.015

    # Identifiers block
    y = _draw_kv_block(
        fig,
        [
            ("Sample", str(case["sample"])),
            ("Batch", str(case["batch_label"])),
            ("Predicted sex", _format_value(case.get("sex"))),
            ("Predicted type", _format_value(case.get("predicted_aneuploidy_type"))),
            ("Truth label", _format_value(case.get("true_aneuploidy_type"))),
        ],
        start_y=y,
        columns=1,
        line_height=0.020,
        label_fraction=0.30,
    )

    # Ploidy / call metrics
    y = _section_band(fig, y - 0.008, "Ploidy & Call Metrics", eyebrow="Findings")
    baseline = (
        f"{_format_value(case.get('baseline_ploidy_type'))}"
        f" (CN={_format_value(case.get('autosomal_baseline_cn'))})"
    )
    y = _draw_kv_block(
        fig,
        [
            ("Baseline ploidy", baseline),
            ("Score", _format_value(case.get("score"))),
            ("Autosomal call", _format_value(case.get("autosomal_aneuploidy_type"))),
            ("Allosomal call", _format_value(case.get("allosomal_aneuploidy_type"))),
            ("Anomalous contigs", str(int(case.get("n_anomalous_contigs", 0) or 0))),
        ],
        start_y=y,
        columns=1,
        line_height=0.020,
        label_fraction=0.30,
    )

    # QC metrics
    y = _section_band(fig, y - 0.008, "Sample QC", eyebrow="Quality")
    depth_pct = _format_value(case.get("sample_depth_percentile"))
    od_pct = _format_value(case.get("sample_overdispersion_percentile"))
    y = _draw_kv_block(
        fig,
        [
            ("Sample depth ratio", _format_value(case.get("sample_depth_ratio"))),
            ("Depth percentile (batch)", depth_pct),
            ("Sample overdispersion", _format_value(case.get("sample_overdispersion_map"))),
            ("Overdispersion percentile", od_pct),
        ],
        start_y=y,
        columns=1,
        line_height=0.020,
        label_fraction=0.30,
    )

    # Contig events
    y = _section_band(fig, y - 0.008, "Anomalous Contig Evidence", eyebrow="Events")
    if events.empty:
        _draw_paragraph(
            fig,
            "No per-contig events were emitted at the current confidence threshold.",
            start_y=y, fontsize=8, color=_MUTED,
        )
    else:
        fig, y = _draw_table_across_pages(
            pdf,
            pdf_state,
            events,
            ["chromosome", "copy_number", "coverage_score", "plq", "median_depth"],
            page_header=f"Section {section_number}  \u2022  {section_title}",
            section_title=f"{case['sample']} \u2014 Anomalous Contig Evidence",
            eyebrow="Events",
            start_fig=fig,
            start_y=y,
            max_rows=18,
            col_widths=[0.18, 0.14, 0.24, 0.14, 0.30],
            font_size=7,
            row_height=_COMPACT_TABLE_ROW_HEIGHT,
            column_labels={
                "chromosome": "Chrom",
                "copy_number": "CN",
                "coverage_score": "Score",
                "plq": "PLQ",
                "median_depth": "Median normalized depth",
            },
        )

    _save_page(pdf, fig)

    # Optional diagnostic plot on a follow-up page
    if run is None:
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = _generate_sample_plot_image(
            run, case, events, min_het_alt, tmpdir, logger,
        )
        plot_fig = _new_page(
            pdf_state,
            header=f"Section {section_number}  \u2022  {section_title} \u2014 Diagnostic",
        )
        y = _section_band(
            plot_fig,
            _BODY_TOP,
            f"{case['sample']} \u2014 Diagnostic Plot",
            eyebrow=f"Batch {case['batch_label']}",
        )
        detail_note = _case_detail_note(case)
        if detail_note:
            y = _draw_paragraph(
                plot_fig,
                f"Note: {detail_note}",
                start_y=y,
                fontsize=8,
                color=_MUTED,
                line_height=0.016,
            ) - 0.004
        if image_path is None:
            _draw_paragraph(
                plot_fig,
                "Per-bin statistics were not available for this run/sample, so the\n"
                "diagnostic plot could not be generated.",
                start_y=y, fontsize=9, color=_MUTED,
            )
        else:
            image = plt.imread(image_path)
            # Fit image into body box preserving aspect ratio
            body_h = y - (_BODY_BOTTOM + 0.02)
            body_w = _MARGIN_R - _MARGIN_L
            img_aspect = image.shape[0] / image.shape[1]
            box_aspect = body_h / body_w
            if img_aspect > box_aspect:
                draw_h = body_h
                draw_w = draw_h / img_aspect
            else:
                draw_w = body_w
                draw_h = draw_w * img_aspect
            left = _MARGIN_L + (body_w - draw_w) / 2
            bottom = y - draw_h
            ax = plot_fig.add_axes([left, bottom, draw_w, draw_h])
            ax.imshow(image)
            ax.axis("off")
        _save_page(pdf, plot_fig)


def _add_section_divider(
    pdf: PdfPages,
    pdf_state: dict[str, Any],
    section_number: int,
    title: str,
    n_cases: int,
) -> None:
    """Insert a divider page introducing a case section."""
    fig = _new_page(
        pdf_state,
        header=f"Section {section_number}",
    )
    fig.text(
        0.5, 0.58, f"Section {section_number}",
        fontsize=11, color=_MUTED, ha="center", va="center", family="sans-serif",
    )
    fig.text(
        0.5, 0.52, title,
        fontsize=24, fontweight="bold", color=_INK, ha="center", va="center",
        family="sans-serif",
    )
    fig.add_artist(Line2D(
        [0.30, 0.70], [0.495, 0.495],
        color=_INK, linewidth=0.8, transform=fig.transFigure,
    ))
    fig.text(
        0.5, 0.47,
        f"{n_cases} case{'s' if n_cases != 1 else ''} in this section",
        fontsize=10, color=_MUTED, ha="center", va="top", style="italic",
    )
    _save_page(pdf, fig)


def _add_case_pages(
    pdf: PdfPages,
    pdf_state: dict[str, Any],
    case_df: pd.DataFrame,
    event_df: pd.DataFrame,
    runs: list[RunData],
    min_het_alt: int,
    *,
    starting_section_number: int = 3,
    logger: Any | None = None,
) -> None:
    run_by_key = _run_lookup(runs)
    section_number = starting_section_number
    for category, title in _CASE_SECTIONS:
        section_cases = (
            case_df[case_df["category"] == category]
            if not case_df.empty else pd.DataFrame()
        )
        if section_cases.empty:
            continue
        _add_section_divider(pdf, pdf_state, section_number, title, len(section_cases))
        for _, case in section_cases.sort_values(["batch_label", "sample"]).iterrows():
            events = (
                event_df[
                    (event_df["sample_key"] == case["sample_key"]) &
                    (event_df["category"] == category)
                ]
                if not event_df.empty else pd.DataFrame()
            )
            run = run_by_key.get(
                (int(case["batch_id"]), str(case["batch_label"]))
            )
            _add_case_page(
                pdf, pdf_state, title, section_number, case, events, run, min_het_alt, logger,
            )
        section_number += 1


def _write_pdf_report(
    report_path: Path,
    runs: list[RunData],
    summary_df: pd.DataFrame,
    case_df: pd.DataFrame,
    event_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    prob_threshold: float,
    min_het_alt: int,
    logger: Any | None = None,
) -> None:
    """Write the aggregate multi-page PDF report."""
    _apply_report_theme()
    pdf_state: dict[str, Any] = {
        "page": 0,
        "footer_left": (
            f"Generated {datetime.now().strftime('%Y-%m-%d')}"
            f"  \u2022  {len(runs)} batch{'es' if len(runs) != 1 else ''}"
        ),
    }
    # Use neutral typography overrides scoped to the PDF (do not pollute global state).
    rc_overrides = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "axes.edgecolor": _INK,
        "text.color": _INK,
        "axes.labelcolor": _INK,
        "xtick.color": _INK,
        "ytick.color": _INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    toc_entries = _build_report_toc_entries(
        runs,
        summary_df,
        case_df,
        event_df,
        missing_df,
        prob_threshold,
    )
    with plt.rc_context(rc_overrides), PdfPages(report_path) as pdf:
        toc_link_specs = _add_cover_page(pdf, pdf_state, runs, summary_df, toc_entries, report_path)
        _add_inventory_pages(pdf, pdf_state, runs, prob_threshold)
        _add_summary_page(pdf, pdf_state, summary_df, case_df, missing_df)
        _add_case_pages(pdf, pdf_state, case_df, event_df, runs, min_het_alt, logger=logger)
        _add_appendix_pages(pdf, pdf_state)
        _add_pdf_internal_links(pdf, 1, toc_link_specs)


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
        logger=logger,
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