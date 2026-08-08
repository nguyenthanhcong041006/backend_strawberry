from pathlib import Path

import pandas as pd

from .io import display_path


def timestamp_coverage_summary(df: pd.DataFrame, timestamp_column: str) -> list[str]:
    lines: list[str] = []
    if timestamp_column not in df.columns:
        lines.append(f'- `{timestamp_column}`: not available')
        return lines
    timestamps = pd.to_datetime(df[timestamp_column], errors='coerce')
    parsed = timestamps.dropna()
    if parsed.empty:
        lines.append(f'- `{timestamp_column}` exists but cannot be parsed')
        return lines
    lines.append(f'- `{timestamp_column}` range: {parsed.min()} to {parsed.max()}')
    lines.append(f'- `{timestamp_column}` parsed rows: {len(parsed)} / {len(df)}')
    lines.append(f'- unique dates: {parsed.dt.date.nunique()}')
    diffs = parsed.sort_values().diff().dt.total_seconds().dropna()
    if diffs.empty:
        lines.append('- no interval data available for timestamp coverage')
        return lines
    lines.append(
        f'- timestamp interval min/median/max (seconds): {int(diffs.min())}/{int(diffs.median())}/{int(diffs.max())}'
    )
    large_gaps = diffs[diffs > max(diffs.median() * 4, 3600)]
    if len(large_gaps):
        lines.append(f'- large gaps (>4x median or >1 hour): {len(large_gaps)}')
    return lines


def numeric_range_summary(df: pd.DataFrame, column: str) -> list[str]:
    lines: list[str] = []
    if column not in df.columns:
        lines.append(f'- `{column}`: missing from manifest')
        return lines
    values = pd.to_numeric(df[column], errors='coerce').dropna()
    if values.empty:
        lines.append(f'- `{column}`: all values missing or non-numeric')
        return lines
    stats = values.describe()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = values[(values < lower) | (values > upper)]
    lines.append(
        f'- `{column}`: count={int(stats["count"])}, mean={stats["mean"]:.2f}, min={stats["min"]}, max={stats["max"]}, missing={int(df[column].isna().sum())}'
    )
    lines.append(f'- `{column}` outliers (1.5*IQR): {len(outliers)}')
    return lines


def missing_value_summary(df: pd.DataFrame, columns: list[str], label: str) -> list[str]:
    lines: list[str] = []
    lines.append(f'- {label}')
    if df.empty:
        lines.append('  - no rows available for this table')
        return lines
    total = len(df)
    for column in columns:
        if column not in df.columns:
            lines.append(f'  - `{column}`: missing from data')
            continue
        missing = int(df[column].isna().sum())
        pct = missing / total * 100 if total else 0.0
        lines.append(f'  - `{column}`: {missing}/{total} missing ({pct:.1f}%)')
    return lines


def sequence_length_summary(frame_manifest: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if frame_manifest.empty or 'fruit_id' not in frame_manifest.columns:
        lines.append('- sequence length summary: unavailable because frame manifest is missing or lacks `fruit_id`')
        return lines
    counts = frame_manifest.groupby('fruit_id').size()
    lines.append(f'- fruit count sequences: {len(counts)} fruits')
    lines.append(f'- frame count per fruit: min={int(counts.min())}, median={int(counts.median())}, max={int(counts.max())}')
    lines.append(f'- top 10 fruits by frame count: {dict(counts.sort_values(ascending=False).head(10).to_dict())}')
    if 'sequence_index' in frame_manifest.columns:
        span = frame_manifest.groupby('fruit_id')['sequence_index'].agg(['min', 'max'])
        span_diff = (span['max'] - span['min']).describe()
        lines.append(
            f'- sequence index span per fruit: min={int(span_diff["min"])}, median={int(span_diff["50%"])}' \
            f', max={int(span_diff["max"])}'
        )
    return lines


def sensor_anomaly_summary(df: pd.DataFrame, label: str) -> list[str]:
    lines: list[str] = []
    if df.empty:
        lines.append(f'- {label}: no rows available')
        return lines
    needs = ['temperature_c', 'humidity_pct', 'temperature_anomaly', 'humidity_anomaly', 'sensor_status']
    available = [c for c in needs if c in df.columns]
    if not available:
        lines.append(f'- {label}: no sensor columns available in this dataset')
        return lines
    if 'temperature_c' in df.columns:
        temp_values = pd.to_numeric(df['temperature_c'], errors='coerce').dropna()
        if not temp_values.empty:
            lines.append(f'- `{label}` temperature_c range: {float(temp_values.min()):.1f} to {float(temp_values.max()):.1f}')
        else:
            lines.append(f'- `{label}` temperature_c: no valid values')
    if 'humidity_pct' in df.columns:
        hum_values = pd.to_numeric(df['humidity_pct'], errors='coerce').dropna()
        if not hum_values.empty:
            lines.append(f'- `{label}` humidity_pct range: {float(hum_values.min()):.1f} to {float(hum_values.max()):.1f}')
        else:
            lines.append(f'- `{label}` humidity_pct: no valid values')
    if 'temperature_anomaly' in df.columns:
        count = int(df['temperature_anomaly'].astype(bool).sum())
        lines.append(f'- `{label}` temperature_anomaly count: {count}')
    if 'humidity_anomaly' in df.columns:
        count = int(df['humidity_anomaly'].astype(bool).sum())
        lines.append(f'- `{label}` humidity_anomaly count: {count}')
    if 'sensor_status' in df.columns:
        status_counts = df['sensor_status'].astype(str).value_counts().to_dict()
        lines.append(f'- `{label}` sensor_status counts: {status_counts}')
    return lines


def build_basic_report(
    frame_manifest: pd.DataFrame,
    labels: pd.DataFrame,
    eol_anchors: pd.DataFrame,
    numeric_mapping: pd.DataFrame,
    excluded_frames: pd.DataFrame,
    summary_json: dict,
    report_dir: Path,
    manifest_dir: Path | None = None,
    graph_dir: Path | None = None,
    validation_issues: dict[str, list[str]] | None = None,
) -> list[str]:
    project_root = report_dir.parent.parent.parent
    manifest_dir = manifest_dir or project_root / 'data' / '02_processed' / 'strawberry' / 'manifests_mock'
    graph_dir = graph_dir or project_root / 'output' / 'graphs' / 'eda'

    lines: list[str] = []
    lines.append('# EDA Report')
    lines.append('')
    lines.append('')
    lines.append('## Paths')
    lines.append('')
    lines.append(f'- `MANIFEST_DIR`: `{display_path(manifest_dir, project_root)}`')
    lines.append(f'- `REPORT_DIR`: `{display_path(report_dir, project_root)}`')
    lines.append(f'- `GRAPH_DIR`: `{display_path(graph_dir, project_root)}`')
    lines.append('')
    lines.append('## Available files')
    lines.append('')
    lines.append(f'- `frame_manifest.csv`: {frame_manifest.shape[0]} rows')
    lines.append(f'- `labels.csv`: {labels.shape[0]} rows')
    lines.append(f'- `eol_anchors.csv`: {eol_anchors.shape[0]} rows')
    lines.append(f'- `numeric_mapping.csv`: {numeric_mapping.shape[0]} rows')
    lines.append(f'- `excluded_frames.csv`: {excluded_frames.shape[0]} rows')
    lines.append('')
    if summary_json:
        lines.append('## Preprocessing summary')
        lines.append('')
        for key, value in summary_json.items():
            lines.append(f'- `{key}`: {value}')
        lines.append('')
    lines.append('## Validation checks')
    lines.append('')
    if validation_issues:
        any_issues = False
        for check_name, issues in validation_issues.items():
            if issues:
                any_issues = True
                lines.append(f'- `{check_name}`: {len(issues)} issue(s)')
                for issue in issues:
                    lines.append(f'  - {issue}')
            else:
                lines.append(f'- `{check_name}`: passed')
        if not any_issues:
            lines.append('- all validation checks passed')
    else:
        lines.append('- validation was not run')
    lines.append('')
    lines.append('## Frame totals')
    lines.append('')
    raw_count = (
        summary_json.get('raw_frame_count')
        or summary_json.get('raw_frames')
        or summary_json.get('raw_image_count')
        or summary_json.get('total_raw_frames')
    )
    if raw_count is None:
        lines.append('- raw frames: unknown (not available in preprocessing summary)')
    else:
        lines.append(f'- raw frames: {raw_count}')
    lines.append(f'- processed frames: {frame_manifest.shape[0]}')
    lines.append(f'- excluded frames: {excluded_frames.shape[0]}')
    lines.append('')
    lines.append('## Missing values')
    lines.append('')
    lines.extend(missing_value_summary(
        frame_manifest,
        ['timestamp', 'fruit_id', 'fruit_type', 'image_path', 'mask_path', 'temperature_c', 'humidity_pct', 'mask_valid'],
        'Frame manifest',
    ))
    lines.append('')
    lines.extend(missing_value_summary(
        labels,
        ['timestamp', 'fruit_id', 'rul_hours', 'temperature_c', 'humidity_pct', 'firmness_avg'],
        'Label manifest',
    ))
    lines.append('')
    lines.append('## Sequence lengths per fruit')
    lines.append('')
    lines.extend(sequence_length_summary(frame_manifest))
    lines.append('')
    lines.append('## Sensor anomalies and ranges')
    lines.append('')
    lines.extend(sensor_anomaly_summary(frame_manifest, 'Frame manifest'))
    lines.append('')
    lines.extend(sensor_anomaly_summary(labels, 'Label manifest'))
    lines.append('')
    lines.append('## Frame counts')
    lines.append('')
    if 'experiment_id' in frame_manifest.columns:
        experiment_counts = frame_manifest['experiment_id'].astype(str).value_counts()
        lines.append(f'- unique `experiment_id`: {experiment_counts.size}')
        lines.append(f'- frame counts by experiment_id (top 10): {dict(list(experiment_counts.items())[:10])}')
    else:
        lines.append('- `experiment_id`: not available in frame manifest')
    if 'fruit_type' in frame_manifest.columns:
        fruit_type_counts = frame_manifest['fruit_type'].astype(str).value_counts()
        lines.append(f'- unique `fruit_type`: {fruit_type_counts.size}')
        lines.append(f'- frame counts by fruit_type: {dict(fruit_type_counts.to_dict())}')
    else:
        lines.append('- `fruit_type`: not available in frame manifest')
    if 'timestamp' in frame_manifest.columns:
        timestamps = pd.to_datetime(frame_manifest['timestamp'], errors='coerce')
        if timestamps.notna().any():
            date_counts = timestamps.dt.date.value_counts().sort_index()
            lines.append(f'- frame count by date (top 10): {dict(list(date_counts.items())[:10])}')
        else:
            lines.append('- `timestamp` exists but cannot be parsed into dates')
    else:
        lines.append('- `timestamp`: not available in frame manifest')
    if 'fruit_id' in frame_manifest.columns:
        fruit_counts = frame_manifest['fruit_id'].astype(str).value_counts()
        lines.append(f'- unique `fruit_id`: {fruit_counts.size}')
        lines.append(f'- frame counts by fruit_id (top 10): {dict(list(fruit_counts.items())[:10])}')
    else:
        lines.append('- `fruit_id`: not available in frame manifest')
    lines.append('')
    if len(frame_manifest):
        lines.append('## Frame manifest overview')
        lines.append('')
        lines.append(f'- unique `fruit_id`: {frame_manifest["fruit_id"].nunique() if "fruit_id" in frame_manifest.columns else 0}')
        if 'experiment_id' in frame_manifest.columns:
            lines.append(f'- unique `experiment_id`: {frame_manifest["experiment_id"].nunique()}')
        if 'timestamp' in frame_manifest.columns:
            lines.extend(timestamp_coverage_summary(frame_manifest, 'timestamp'))
    lines.append('')
    if len(excluded_frames):
        lines.append('## Excluded frames overview')
        lines.append('')
        if 'reason' in excluded_frames.columns:
            excluded_reasons = excluded_frames['reason'].astype(str).value_counts()
            lines.append(f'- excluded frame reasons: {dict(excluded_reasons.to_dict())}')
        if 'mask_reason' in excluded_frames.columns:
            excluded_mask_reasons = excluded_frames['mask_reason'].astype(str).value_counts()
            lines.append(f'- excluded mask reasons: {dict(excluded_mask_reasons.to_dict())}')
        if 'timestamp' in excluded_frames.columns:
            lines.extend(timestamp_coverage_summary(excluded_frames, 'timestamp'))
        lines.append('')
    if len(labels):
        lines.append('## Labels overview')
        lines.append('')
        if 'rul_hours' in labels.columns:
            lines.extend(numeric_range_summary(labels, 'rul_hours'))
        if 'temperature_c' in labels.columns:
            lines.extend(numeric_range_summary(labels, 'temperature_c'))
        if 'humidity_pct' in labels.columns:
            lines.extend(numeric_range_summary(labels, 'humidity_pct'))
        if 'firmness_avg' in labels.columns:
            lines.extend(numeric_range_summary(labels, 'firmness_avg'))
    return lines
