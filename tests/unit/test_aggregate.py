from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from gatk_sv_ploidy import aggregate


def _prediction_row(
    sample: str,
    *,
    sex: str = "FEMALE",
    predicted_type: str = "NORMAL",
    autosomal_type: str = "NONE",
    allosomal_type: str = "NONE",
    baseline_type: str = "DIPLOID",
    baseline_cn: int = 2,
    score: float = 0.95,
    depth_ratio: float = 1.0,
) -> dict[str, object]:
    return {
        "sample": sample,
        "sex": sex,
        "predicted_aneuploidy_type": predicted_type,
        "autosomal_aneuploidy_type": autosomal_type,
        "allosomal_aneuploidy_type": allosomal_type,
        "baseline_ploidy_type": baseline_type,
        "autosomal_baseline_cn": baseline_cn,
        "score": score,
        "sample_depth_ratio": depth_ratio,
        "true_aneuploidy_type": "NORMAL",
    }


def _chrom_row(
    sample: str,
    chromosome: str,
    copy_number: int,
    prob: float,
    *,
    is_aneuploid: bool = False,
    plq: int = 40,
    overdispersion: float = 0.1,
    baseline_cn: int = 2,
) -> dict[str, object]:
    return {
        "sample": sample,
        "chromosome": chromosome,
        "copy_number": copy_number,
        "mean_cn_probability": prob,
        "plq": plq,
        "is_aneuploid": is_aneuploid,
        "n_bins": 10,
        "median_depth": float(copy_number),
        "mean_depth": float(copy_number),
        "sample_overdispersion_map": overdispersion,
        "autosomal_baseline_cn": baseline_cn,
    }


def _write_run(
    root: Path,
    predictions: list[dict[str, object]],
    chrom_rows: list[dict[str, object]],
    *,
    filtered_rows: list[dict[str, object]] | None = None,
) -> None:
    (root / "call").mkdir(parents=True)
    (root / "infer").mkdir(parents=True)
    pd.DataFrame(predictions).to_csv(
        root / "call" / "aneuploidy_type_predictions.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(chrom_rows).to_csv(
        root / "infer" / "chromosome_stats.tsv",
        sep="\t",
        index=False,
    )
    if filtered_rows is not None:
        pd.DataFrame(filtered_rows).to_csv(
            root / "call" / "chromosome_stats.filtered.tsv",
            sep="\t",
            index=False,
        )


def _basic_chrom_rows(sample: str, *, overdispersion: float = 0.1) -> list[dict[str, object]]:
    return [
        _chrom_row(sample, "chr21", 2, 0.95, overdispersion=overdispersion),
        _chrom_row(sample, "chrX", 2, 0.95, overdispersion=overdispersion),
        _chrom_row(sample, "chrY", 0, 0.95, overdispersion=overdispersion),
    ]


def test_default_batch_labels_are_stable_for_duplicate_names() -> None:
    labels = aggregate._default_batch_labels(["/tmp/run", "/other/run", "/tmp/next"])
    assert labels == ["run", "run_2", "next"]


def test_load_run_prefers_filtered_chromosome_stats(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_prediction_row("S1")],
        _basic_chrom_rows("S1"),
        filtered_rows=[
            *_basic_chrom_rows("S1")[:2],
            _chrom_row("S1", "chrY", 1, 0.40, is_aneuploid=False),
        ],
    )

    run = aggregate._load_run_data(
        run_dir,
        batch_id=1,
        batch_label="batch",
        binq_field="auto",
    )

    assert run.used_filtered_chrom_stats is True
    assert run.chrom_stats_source.name == "chromosome_stats.filtered.tsv"
    assert run.chrom_df.loc[run.chrom_df["chromosome"] == "chrY", "copy_number"].iloc[0] == 1


def test_build_report_tables_classifies_requested_sections(tmp_path) -> None:
    run_a = tmp_path / "run_a"
    _write_run(
        run_a,
        [
            _prediction_row("S1", depth_ratio=1.0),
            _prediction_row(
                "S2",
                sex="TRIPLE_X",
                predicted_type="TRIPLE_X",
                allosomal_type="TRIPLE_X",
                score=0.91,
                depth_ratio=1.2,
            ),
        ],
        [
            *_basic_chrom_rows("S1", overdispersion=0.1),
            _chrom_row("S2", "chr21", 2, 0.91, overdispersion=0.2),
            _chrom_row("S2", "chrX", 3, 0.91, is_aneuploid=True, overdispersion=0.2),
            _chrom_row("S2", "chrY", 0, 0.91, overdispersion=0.2),
        ],
    )
    run_b = tmp_path / "run_b"
    _write_run(
        run_b,
        [
            _prediction_row("S3", score=0.42, depth_ratio=0.85),
            _prediction_row(
                "S4",
                sex="TRIPLOID_FEMALE",
                predicted_type="TRIPLOID",
                baseline_type="TRIPLOID",
                baseline_cn=3,
                score=0.88,
                depth_ratio=1.4,
            ),
            _prediction_row(
                "S5",
                predicted_type="TRISOMY_21",
                autosomal_type="TRISOMY_21",
                score=0.93,
                depth_ratio=1.1,
            ),
        ],
        [
            _chrom_row("S3", "chr21", 3, 0.42, is_aneuploid=False, plq=8, overdispersion=0.15),
            _chrom_row("S3", "chrX", 2, 0.90, overdispersion=0.15),
            _chrom_row("S3", "chrY", 0, 0.90, overdispersion=0.15),
            _chrom_row("S4", "chr21", 3, 0.88, baseline_cn=3, overdispersion=0.3),
            _chrom_row("S4", "chrX", 3, 0.88, baseline_cn=3, overdispersion=0.3),
            _chrom_row("S4", "chrY", 0, 0.88, baseline_cn=3, overdispersion=0.3),
            _chrom_row("S5", "chr21", 3, 0.93, is_aneuploid=True, overdispersion=0.18),
            _chrom_row("S5", "chrX", 2, 0.93, overdispersion=0.18),
            _chrom_row("S5", "chrY", 0, 0.93, overdispersion=0.18),
        ],
    )
    runs = [
        aggregate._load_run_data(run_a, batch_id=1, batch_label="run_a", binq_field="auto"),
        aggregate._load_run_data(run_b, batch_id=2, batch_label="run_b", binq_field="auto"),
    ]

    summary_df, case_df, event_df, missing_df, _ = aggregate._build_report_tables(
        runs,
        prob_threshold=0.5,
    )

    summary = summary_df.set_index("metric")["value"].to_dict()
    assert summary["n_samples"] == 5
    assert summary["n_batches"] == 2
    assert summary["n_confident_sex_aneuploidy_samples"] == 1
    assert summary["n_confident_polyploidy_samples"] == 1
    assert summary["n_confident_autosomal_aneuploidy_samples"] == 1
    assert summary["n_low_confidence_aneuploidy_samples"] == 1
    assert summary["n_confident_autosomal_events_chr21"] == 1
    assert set(case_df["category"]) == {
        "confident_sex_aneuploidy",
        "confident_polyploidy",
        "confident_autosomal_aneuploidy",
        "low_confidence_aneuploidy",
    }
    low_event = event_df[event_df["category"] == "low_confidence_aneuploidy"].iloc[0]
    assert low_event["sample"] == "S3"
    assert low_event["chromosome"] == "chr21"
    assert low_event["mean_cn_probability"] == pytest.approx(0.42)
    assert missing_df["artifact"].isin(["bin_stats", "site_data", "polyploidy_manifest"]).any()


def test_main_writes_pdf_and_sidecars(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_prediction_row("S1", score=0.42)],
        [
            _chrom_row("S1", "chr21", 3, 0.42, is_aneuploid=False, plq=8),
            _chrom_row("S1", "chrX", 2, 0.90),
            _chrom_row("S1", "chrY", 0, 0.90),
        ],
    )
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["gatk-sv-ploidy aggregate", str(run_dir), "-o", str(output_dir)],
    )

    aggregate.main()

    report_path = output_dir / "aggregate_report.pdf"
    assert report_path.exists()
    assert report_path.stat().st_size > 0
    for name in [
        "aggregate_summary.tsv",
        "aggregate_cases.tsv",
        "aggregate_contig_events.tsv",
        "aggregate_missing_artifacts.tsv",
    ]:
        assert (output_dir / name).exists()
    case_df = pd.read_csv(output_dir / "aggregate_cases.tsv", sep="\t")
    assert case_df["category"].tolist() == ["low_confidence_aneuploidy"]


def test_validate_args_requires_matching_batch_labels(tmp_path) -> None:
    namespace = type(
        "Args",
        (),
        {
            "output_name": "aggregate_report.pdf",
            "batch_label": ["one"],
            "work_dirs": ["a", "b"],
            "prob_threshold": 0.5,
            "min_het_alt": 3,
        },
    )()
    with pytest.raises(ValueError, match="once per work directory"):
        aggregate._validate_args(namespace)