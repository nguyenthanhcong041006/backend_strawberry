from pathlib import Path
from typing import Optional

import pandas as pd

from .io import configure_paths, load_csv_if_exists, load_summary_json, save_report
from .plots import (
    plot_counts_by_column,
    plot_eol_anchor_status,
    plot_eol_timestamp_distribution,
    plot_firmness_trend,
    plot_frame_counts_by_date,
    plot_frame_counts_by_date_and_fruit,
    plot_histogram,
    plot_mapping_delta_histogram,
    plot_mapping_method_counts,
    plot_mask_valid_over_time,
    plot_mask_valid_ratio_over_time,
    plot_numeric_mapping_status_counts,
    plot_roi_layout_comparison,
    plot_roi_position_distribution,
    plot_rul_distribution_by_fruit,
    plot_scatter,
    plot_temporal_image_strip,
    plot_time_series,
    plot_color_trends,
)
from .report import build_basic_report
from .validate import collect_validation_issues


def create_graphs(
    frame_manifest: pd.DataFrame,
    labels: pd.DataFrame,
    excluded_frames: pd.DataFrame,
    eol_anchors: pd.DataFrame,
    numeric_mapping: pd.DataFrame,
    graph_dir: Path,
) -> None:
    if not len(frame_manifest) and not len(labels) and not len(excluded_frames) and not len(eol_anchors) and not len(numeric_mapping):
        return
    graph_dir.mkdir(parents=True, exist_ok=True)
    plot_frame_counts_by_date(frame_manifest, graph_dir / 'frame_count_by_date.png')
    plot_frame_counts_by_date_and_fruit(frame_manifest, graph_dir / 'frame_count_by_date_and_fruit.png')
    plot_roi_layout_comparison(frame_manifest, graph_dir / 'roi_layout_comparison.png')

    if len(frame_manifest):
        plot_counts_by_column(frame_manifest, 'fruit_id', graph_dir / 'frame_count_by_fruit.png', 'Frame count by fruit ID')
        plot_counts_by_column(frame_manifest, 'experiment_id', graph_dir / 'frame_count_by_experiment.png', 'Frame count by experiment ID')
        if 'fruit_type' in frame_manifest.columns:
            plot_counts_by_column(frame_manifest, 'fruit_type', graph_dir / 'frame_count_by_fruit_type.png', 'Frame count by fruit type')
        if 'mask_valid' in frame_manifest.columns:
            plot_mask_valid_over_time(frame_manifest, graph_dir / 'mask_valid_over_time.png')
            plot_mask_valid_ratio_over_time(frame_manifest, graph_dir / 'mask_valid_ratio_over_time.png')

    if len(excluded_frames):
        if 'reason' in excluded_frames.columns:
            plot_counts_by_column(excluded_frames, 'reason', graph_dir / 'excluded_frame_reasons.png', 'Excluded frame reasons')
        if 'mask_reason' in excluded_frames.columns:
            plot_counts_by_column(excluded_frames, 'mask_reason', graph_dir / 'excluded_frame_mask_reasons.png', 'Excluded frame mask reasons')
        if 'roi_position' in excluded_frames.columns:
            plot_counts_by_column(excluded_frames, 'roi_position', graph_dir / 'excluded_frames_by_roi.png', 'Excluded frames by ROI position')
            plot_roi_position_distribution(excluded_frames, graph_dir / 'excluded_frames_roi_position_distribution.png')
        if 'fruit_id' in excluded_frames.columns:
            plot_counts_by_column(excluded_frames, 'fruit_id', graph_dir / 'excluded_frames_by_fruit.png', 'Excluded frames by fruit ID')

    if len(labels):
        if 'fruit_id' in labels.columns:
            plot_counts_by_column(labels, 'fruit_id', graph_dir / 'label_count_by_fruit.png', 'Label count by fruit ID')
        if 'rul_hours' in labels.columns:
            plot_rul_distribution_by_fruit(labels, graph_dir / 'rul_distribution_by_fruit.png')
            plot_time_series(labels, 'timestamp', 'rul_hours', graph_dir / 'rul_over_time.png', 'RUL over time')
            plot_histogram(labels, 'rul_hours', graph_dir / 'rul_distribution.png', 'RUL distribution')
        if 'temperature_c' in labels.columns:
            plot_time_series(labels, 'timestamp', 'temperature_c', graph_dir / 'temperature_over_time.png', 'Temperature over time')
            plot_histogram(labels, 'temperature_c', graph_dir / 'temperature_distribution.png', 'Temperature distribution')
            plot_scatter(labels, 'temperature_c', 'rul_hours', graph_dir / 'temperature_vs_rul.png', 'Temperature vs RUL')
        if 'humidity_pct' in labels.columns:
            plot_time_series(labels, 'timestamp', 'humidity_pct', graph_dir / 'humidity_over_time.png', 'Humidity over time')
            plot_histogram(labels, 'humidity_pct', graph_dir / 'humidity_distribution.png', 'Humidity distribution')
            plot_scatter(labels, 'humidity_pct', 'rul_hours', graph_dir / 'humidity_vs_rul.png', 'Humidity vs RUL')
        if 'firmness_avg' in labels.columns:
            plot_firmness_trend(labels, graph_dir / 'firmness_trend_over_time.png')
        plot_color_trends(labels, graph_dir)
        if 'fruit_id' in labels.columns:
            fruit_ids = labels['fruit_id'].astype(str).unique()[:3]
            for fruit_id in fruit_ids:
                subset = labels[labels['fruit_id'].astype(str) == str(fruit_id)]
                plot_temporal_image_strip(subset, str(fruit_id), graph_dir / f'temporal_image_strip_{fruit_id}.png')
    if len(eol_anchors):
        plot_eol_anchor_status(eol_anchors, graph_dir / 'eol_anchor_status.png')
        plot_eol_timestamp_distribution(eol_anchors, graph_dir / 'eol_timestamp_distribution.png')
    if len(numeric_mapping):
        plot_mapping_delta_histogram(numeric_mapping, graph_dir / 'numeric_mapping_delta_histogram.png')
        plot_mapping_method_counts(numeric_mapping, graph_dir / 'numeric_mapping_method_counts.png')
        plot_numeric_mapping_status_counts(numeric_mapping, graph_dir / 'numeric_mapping_status_counts.png')


def run_from_manifests(root: Optional[Path] = None) -> None:
    _, manifest_dir, report_dir, graph_dir = configure_paths(root)

    required = [
        manifest_dir / 'frame_manifest.csv',
        manifest_dir / 'labels.csv',
        manifest_dir / 'eol_anchors.csv',
        manifest_dir / 'numeric_mapping.csv',
        manifest_dir / 'excluded_frames.csv',
    ]
    for p in required:
        if not p.exists():
            print(f'Warning: missing {p}')

    frame_manifest = load_csv_if_exists(manifest_dir / 'frame_manifest.csv')
    labels = load_csv_if_exists(manifest_dir / 'labels.csv')
    eol_anchors = load_csv_if_exists(manifest_dir / 'eol_anchors.csv')
    numeric_mapping = load_csv_if_exists(manifest_dir / 'numeric_mapping.csv')
    excluded_frames = load_csv_if_exists(manifest_dir / 'excluded_frames.csv')
    summary = load_summary_json(manifest_dir / 'preprocessing_summary.json')
    validation_issues = collect_validation_issues(frame_manifest, labels, eol_anchors)

    lines = build_basic_report(
        frame_manifest,
        labels,
        eol_anchors,
        numeric_mapping,
        excluded_frames,
        summary,
        report_dir,
        manifest_dir,
        graph_dir,
        validation_issues,
    )
    save_report(lines, report_dir)
    create_graphs(frame_manifest, labels, excluded_frames, eol_anchors, numeric_mapping, graph_dir)
