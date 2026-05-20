from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    plot_median_depth: float | None = None,
) -> dict[str, object]:
    row = {
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
    if plot_median_depth is not None:
        row["plot_median_depth"] = plot_median_depth
    return row


def _write_run(
    root: Path,
    predictions: list[dict[str, object]],
    chrom_rows: list[dict[str, object]],
    *,
    filtered_rows: list[dict[str, object]] | None = None,
    call_prob_threshold: float | None = None,
    truth_json: str | None = None,
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
    if call_prob_threshold is not None or truth_json is not None:
        args: dict[str, object] = {}
        if call_prob_threshold is not None:
            args["prob_threshold"] = call_prob_threshold
        if truth_json is not None:
            args["truth_json"] = truth_json
        (root / "call" / "call.log").write_text(
            "Command arguments: "
            + json.dumps(args)
            + "\n",
            encoding="utf-8",
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


def test_build_batch_inventory_rows_uses_sample_count(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_prediction_row("S1"), _prediction_row("S2"), _prediction_row("S2")],
        _basic_chrom_rows("S1"),
    )
    run = aggregate._load_run_data(
        run_dir,
        batch_id=7,
        batch_label="batch_alpha",
        binq_field="auto",
    )

    rows = aggregate._build_batch_inventory_rows([run], prob_threshold=0.9)

    assert list(rows.columns) == [
        "batch_id",
        "batch_label",
        "sample_count",
        "call_prob_threshold",
        "work_dir",
    ]
    assert rows.loc[0, "sample_count"] == 2


def test_build_report_tables_blanks_default_truth_without_truth_json(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_prediction_row("S1", predicted_type="TRISOMY_21", autosomal_type="TRISOMY_21")],
        [
            _chrom_row("S1", "chr21", 3, 0.95, is_aneuploid=True),
            _chrom_row("S1", "chrX", 2, 0.95),
            _chrom_row("S1", "chrY", 0, 0.95),
        ],
    )
    run = aggregate._load_run_data(
        run_dir,
        batch_id=1,
        batch_label="batch",
        binq_field="auto",
    )

    _, case_df, _, _, _ = aggregate._build_report_tables([run], prob_threshold=0.9)

    assert case_df.loc[0, "true_aneuploidy_type"] == ""


def test_build_report_tables_keeps_truth_when_truth_json_was_provided(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_prediction_row("S1", predicted_type="TRISOMY_21", autosomal_type="TRISOMY_21")],
        [
            _chrom_row("S1", "chr21", 3, 0.95, is_aneuploid=True),
            _chrom_row("S1", "chrX", 2, 0.95),
            _chrom_row("S1", "chrY", 0, 0.95),
        ],
        truth_json="truth.json",
    )
    run = aggregate._load_run_data(
        run_dir,
        batch_id=1,
        batch_label="batch",
        binq_field="auto",
    )

    _, case_df, _, _, _ = aggregate._build_report_tables([run], prob_threshold=0.9)

    assert case_df.loc[0, "true_aneuploidy_type"] == "NORMAL"


def test_build_report_tables_prefers_normalized_event_depth(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_prediction_row("S1", predicted_type="TRISOMY_21", autosomal_type="TRISOMY_21")],
        [
            _chrom_row("S1", "chr21", 3, 0.95, is_aneuploid=True, plot_median_depth=1.5),
            _chrom_row("S1", "chrX", 2, 0.95),
            _chrom_row("S1", "chrY", 0, 0.95),
        ],
    )
    run = aggregate._load_run_data(
        run_dir,
        batch_id=1,
        batch_label="batch",
        binq_field="auto",
    )

    _, _, event_df, _, _ = aggregate._build_report_tables([run], prob_threshold=0.9)

    chr21_event = event_df.loc[event_df["chromosome"] == "chr21"].iloc[0]
    assert chr21_event["median_depth"] == pytest.approx(1.5)


def test_load_run_parses_call_prob_threshold_from_call_log(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_prediction_row("S1")],
        _basic_chrom_rows("S1"),
        call_prob_threshold=0.73,
    )

    run = aggregate._load_run_data(
        run_dir,
        batch_id=1,
        batch_label="batch",
        binq_field="auto",
    )

    assert run.call_prob_threshold == pytest.approx(0.73)


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


def test_build_report_tables_uses_per_batch_call_thresholds(tmp_path) -> None:
    run_a = tmp_path / "run_a"
    _write_run(
        run_a,
        [_prediction_row("S1", score=0.55)],
        [
            _chrom_row("S1", "chr21", 3, 0.55, is_aneuploid=False, plq=10),
            _chrom_row("S1", "chrX", 2, 0.95),
            _chrom_row("S1", "chrY", 0, 0.95),
        ],
        call_prob_threshold=0.60,
    )
    run_b = tmp_path / "run_b"
    _write_run(
        run_b,
        [_prediction_row("S2", score=0.55)],
        [
            _chrom_row("S2", "chr21", 3, 0.55, is_aneuploid=False, plq=10),
            _chrom_row("S2", "chrX", 2, 0.95),
            _chrom_row("S2", "chrY", 0, 0.95),
        ],
        call_prob_threshold=0.40,
    )
    runs = [
        aggregate._load_run_data(run_a, batch_id=1, batch_label="run_a", binq_field="auto"),
        aggregate._load_run_data(run_b, batch_id=2, batch_label="run_b", binq_field="auto"),
    ]

    _, case_df, event_df, _, _ = aggregate._build_report_tables(
        runs,
        prob_threshold=0.50,
    )

    assert event_df["sample"].tolist() == ["S1"]
    assert case_df["sample"].tolist() == ["S1"]
    assert event_df.iloc[0]["category"] == "low_confidence_aneuploidy"


def test_build_report_tables_demotes_subthreshold_aneuploid_flags(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [
            _prediction_row(
                "S1",
                predicted_type="TRISOMY_21",
                autosomal_type="TRISOMY_21",
                score=0.55,
            )
        ],
        [
            _chrom_row("S1", "chr21", 3, 0.55, is_aneuploid=True, plq=10),
            _chrom_row("S1", "chrX", 2, 0.95),
            _chrom_row("S1", "chrY", 0, 0.95),
        ],
        call_prob_threshold=0.60,
    )
    run = aggregate._load_run_data(
        run_dir,
        batch_id=1,
        batch_label="run",
        binq_field="auto",
    )

    _, case_df, event_df, _, _ = aggregate._build_report_tables(
        [run],
        prob_threshold=0.50,
    )

    assert event_df["category"].tolist() == ["low_confidence_aneuploidy"]
    assert case_df["category"].tolist() == ["low_confidence_aneuploidy"]


def test_build_report_toc_entries_starts_inventory_on_page_two(tmp_path) -> None:
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
    runs = [
        aggregate._load_run_data(run_dir, batch_id=1, batch_label="run", binq_field="auto"),
    ]
    summary_df, case_df, event_df, missing_df, _ = aggregate._build_report_tables(
        runs,
        prob_threshold=0.5,
    )

    toc_entries = aggregate._build_report_toc_entries(
        runs,
        summary_df,
        case_df,
        event_df,
        missing_df,
        prob_threshold=0.5,
    )

    toc_pages = dict(toc_entries)
    assert toc_pages["Batch Inventory"] == 2
    assert toc_pages["Cohort Summary"] == 3
    assert toc_pages["Case Index"] == 3
    assert toc_pages["Low-confidence Aneuploidies"] == 4
    assert aggregate._APPENDIX_TOC_LABEL in toc_pages
    assert toc_pages[aggregate._APPENDIX_TOC_LABEL] > toc_pages["Low-confidence Aneuploidies"]


def test_build_appendix_field_guide_documents_displayed_tables() -> None:
    appendix_df = aggregate._build_appendix_field_guide()

    display_elements = set(appendix_df["display_element"])
    displayed_labels = set(appendix_df["displayed_label"])
    definitions = "\n".join(appendix_df["definition"].astype(str))

    assert "Cover page - Summary" in display_elements
    assert "Table of Contents" in display_elements
    assert "Batch Inventory" in display_elements
    assert "Cohort Summary" in display_elements
    assert "Case Index" in display_elements
    assert "Case detail - Identifiers" in display_elements
    assert "Case detail - Ploidy and Call Metrics" in display_elements
    assert "Case detail - Sample QC" in display_elements
    assert "Anomalous Contig Evidence" in display_elements
    assert "Report Field Guide" in display_elements
    assert "Sample count" in displayed_labels
    assert "Median normalized depth" in displayed_labels
    assert "Stats source" not in displayed_labels
    assert "mean_cn_probability" in definitions


def test_add_pdf_internal_links_appends_goto_annotations() -> None:
    class DummyPdfFile:
        def __init__(self) -> None:
            self.pageList = ["page1", "page2", "page3"]
            self._annotations = [(None, []), (None, []), (None, [])]

    class DummyPdf:
        def __init__(self) -> None:
            self.file = DummyPdfFile()

        def _ensure_file(self):
            return self.file

    pdf = DummyPdf()
    specs = [aggregate.TocLinkSpec(target_page=3, x0=0.1, y0=0.2, x1=0.9, y1=0.3)]

    aggregate._add_pdf_internal_links(pdf, 1, specs)

    annotations = pdf.file._annotations[0][1]
    assert len(annotations) == 1
    assert annotations[0]["A"]["D"][0] == "page3"
    assert annotations[0]["Rect"][0] < annotations[0]["Rect"][2]
    assert annotations[0]["Rect"][1] < annotations[0]["Rect"][3]


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


def test_draw_kv_block_wraps_long_case_identifiers() -> None:
    fig = plt.figure(figsize=(8.5, 11))
    try:
        end_y = aggregate._draw_kv_block(
            fig,
            [
                ("Sample", "S1"),
                (
                    "Batch",
                    "batch-label-with-an-overly-long-identifier-that-needs-wrapping-"
                    "and-keeps-going-long-enough-to-exercise-bounded-wrapping",
                ),
                ("Predicted sex", "FEMALE"),
                ("Predicted type", "NORMAL"),
                ("Truth label", "NORMAL"),
            ],
            start_y=0.8,
            columns=2,
            line_height=0.02,
            label_fraction=0.20,
            col_widths=[0.74, 0.26],
        )
        text_artists = [text for ax in fig.axes for text in ax.texts]
        batch_axis = next(
            ax for ax in fig.axes
            if ax.texts and "batch-label" in ax.texts[0].get_text()
        )
        right_group_x0 = min(
            ax.get_position().x0
            for ax in fig.axes
            if ax.get_position().x0 > batch_axis.get_position().x0
        )

        assert sum(
            1
            for text in text_artists
            if "batch-label" in text.get_text() and "\n" in text.get_text()
        ) >= 1
        assert batch_axis.get_position().x1 < right_group_x0
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        text_bbox = batch_axis.texts[0].get_window_extent(renderer=renderer)
        axis_bbox = batch_axis.get_window_extent(renderer=renderer)
        assert text_bbox.x1 <= axis_bbox.x1
        assert end_y < 0.8 - (3 * 0.02) - 0.004
    finally:
        plt.close(fig)


def test_format_contigs_uses_score_label() -> None:
    events = pd.DataFrame(
        [
            {"chromosome": "chr21", "copy_number": 3, "mean_cn_probability": 0.42},
            {"chromosome": "chrX", "copy_number": 1, "mean_cn_probability": 0.97},
        ]
    )

    text = aggregate._format_contigs(events)

    assert "score=0.420" in text
    assert "score=0.970" in text
    assert " p=" not in text


def test_draw_table_across_pages_paginates_without_sidecar_note() -> None:
    class DummyPdf:
        def __init__(self) -> None:
            self.saved = 0

        def savefig(self, fig) -> None:
            self.saved += 1

    pdf = DummyPdf()
    pdf_state: dict[str, object] = {"page": 0, "footer_left": ""}
    fig = aggregate._new_page(pdf_state, header="Section 2")
    y = aggregate._section_band(fig, aggregate._BODY_TOP, "Case Index", eyebrow="Section 2")
    df = pd.DataFrame(
        [
            {
                "category": "Low-confidence",
                "batch_label": "batch",
                "sample": f"S{i}",
                "predicted_aneuploidy_type": "NORMAL",
                "anomalous_contigs": "chr21:CN3 score=0.420",
                "score": 0.42,
            }
            for i in range(9)
        ]
    )

    try:
        fig, y = aggregate._draw_table_across_pages(
            pdf,
            pdf_state,
            df,
            ["category", "batch_label", "sample", "predicted_aneuploidy_type",
             "anomalous_contigs", "score"],
            page_header="Section 2",
            section_title="Case Index",
            eyebrow="Section 2",
            start_fig=fig,
            start_y=y,
            max_rows=3,
            col_widths=[0.18, 0.10, 0.14, 0.18, 0.31, 0.09],
            font_size=7,
            column_labels={
                "category": "Category",
                "batch_label": "Batch",
                "sample": "Sample",
                "predicted_aneuploidy_type": "Predicted",
                "anomalous_contigs": "Anomalous contigs",
                "score": "Score",
            },
        )

        assert pdf.saved == 2
        assert pdf_state["page"] == 3
        assert all("sidecar TSV" not in text.get_text() for text in fig.texts)
        assert y < aggregate._BODY_TOP
    finally:
        plt.close(fig)


def test_generate_sample_plot_image_reuses_existing_plot_when_present(tmp_path, caplog) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_prediction_row("S1")],
        _basic_chrom_rows("S1"),
    )
    existing_plot = run_dir / "plot" / "sample_plots" / "S1.png"
    existing_plot.parent.mkdir(parents=True)
    existing_plot.write_bytes(b"png")

    run = aggregate._load_run_data(
        run_dir,
        batch_id=1,
        batch_label="batch",
        binq_field="auto",
    )

    with caplog.at_level(logging.INFO):
        image_path = aggregate._generate_sample_plot_image(
            run,
            pd.Series({"sample": "S1", "category": "low_confidence_aneuploidy"}),
            pd.DataFrame(),
            3,
            str(tmp_path / "scratch"),
            logger=logging.getLogger("test.aggregate"),
        )

    assert image_path == existing_plot
    assert "generate=no reason=reuse_existing_plot" in caplog.text


def test_generate_sample_plot_image_regenerates_when_original_missing(tmp_path, monkeypatch, caplog) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [_prediction_row("S1")],
        _basic_chrom_rows("S1"),
    )
    run = aggregate._load_run_data(
        run_dir,
        batch_id=1,
        batch_label="batch",
        binq_field="auto",
    )
    run.bin_df = pd.DataFrame({"sample": ["S1"]})

    from gatk_sv_ploidy import _plot_detail

    def fake_plot_sample_with_variance(sample_data, all_vars, output_dir, **kwargs):
        plot_path = Path(output_dir) / "sample_plots" / "S1.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_bytes(b"png")

    monkeypatch.setattr(_plot_detail, "plot_sample_with_variance", fake_plot_sample_with_variance)

    with caplog.at_level(logging.INFO):
        image_path = aggregate._generate_sample_plot_image(
            run,
            pd.Series({"sample": "S1", "category": "low_confidence_aneuploidy"}),
            pd.DataFrame(),
            3,
            str(tmp_path / "scratch"),
            logger=logging.getLogger("test.aggregate"),
        )

    assert image_path == tmp_path / "scratch" / "sample_plots" / "S1.png"
    assert "generate=yes reason=missing_existing_plot" in caplog.text


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