from __future__ import annotations

import pandas as pd


def _truthy_values(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({'true', '1', 'yes', 'y'})


def validate_labels_eol_consistency(labels: pd.DataFrame, eol_anchors: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if labels.empty:
        issues.append('labels.csv is empty')
        return issues
    if eol_anchors.empty:
        issues.append('eol_anchors.csv is empty')
        return issues

    keys = ['experiment_id', 'fruit_id', 'roi_id']
    missing = [c for c in keys if c not in labels.columns or c not in eol_anchors.columns]
    if missing:
        issues.append(f'missing required columns for label/EOL consistency: {missing}')
        return issues

    merged = labels.merge(
        eol_anchors[['experiment_id', 'fruit_id', 'roi_id', 'eol_timestamp', 'status']],
        on=keys,
        how='left',
        indicator=True,
    )
    missing_anchor = merged[merged['_merge'] == 'left_only']
    if not missing_anchor.empty:
        issues.append(
            f'{len(missing_anchor)} labels rows have no matching eol_anchors row by experiment_id/fruit_id/roi_id'
        )
    if 'eol_timestamp' in merged.columns:
        matched = merged[merged['_merge'] == 'both']
        missing_ts = matched[matched['eol_timestamp'].isna()]
        if not missing_ts.empty:
            issues.append(f'{len(missing_ts)} matched labels rows have missing eol_timestamp in eol_anchors')
    if 'status' in merged.columns:
        matched = merged[merged['_merge'] == 'both']
        rejected = matched[~matched['status'].astype(str).str.lower().isin({'approved', 'ok', 'accepted'})]
        if not rejected.empty:
            issues.append(f'{len(rejected)} matched labels rows have unapproved eol status')
    return issues


def validate_frame_manifest_roi_timeline(frame_manifest: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if frame_manifest.empty:
        issues.append('frame_manifest.csv is empty')
        return issues

    required = ['fruit_id', 'timestamp']
    missing = [c for c in required if c not in frame_manifest.columns]
    if missing:
        issues.append(f'missing required frame_manifest columns: {missing}')
        return issues

    parsed = pd.to_datetime(frame_manifest['timestamp'], errors='coerce')
    bad_timestamps = parsed.isna().sum()
    if bad_timestamps:
        issues.append(f'{bad_timestamps} frame_manifest rows have invalid timestamps')

    if 'roi_id' in frame_manifest.columns and 'fruit_id' in frame_manifest.columns:
        cross = frame_manifest.groupby(['roi_id', 'fruit_id']).size().reset_index(name='count')
        ambiguous = cross.groupby('roi_id').filter(lambda df: len(df) > 1)
        if not ambiguous.empty:
            roi_ids = ambiguous['roi_id'].unique().tolist()
            issues.append(
                f'ROI IDs mapped to multiple fruit IDs: {len(roi_ids)} roi_id values ({roi_ids[:5]})'
            )

    if 'sequence_index' in frame_manifest.columns:
        frame_manifest = frame_manifest.copy()
        frame_manifest['timestamp'] = pd.to_datetime(frame_manifest['timestamp'], errors='coerce')
        for fruit_id, group in frame_manifest.groupby('fruit_id'):
            if group['sequence_index'].duplicated().any():
                issues.append(f'fruit_id {fruit_id} contains duplicate sequence_index values')
            sequence_order = group.sort_values('sequence_index')
            if not sequence_order['timestamp'].is_monotonic_increasing:
                issues.append(f'fruit_id {fruit_id} has non-monotonic timestamp order')

    return issues


def validate_firmness_available(labels: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if labels.empty:
        issues.append('labels.csv is empty')
        return issues
    if 'fruit_type' not in labels.columns or 'firmness_available' not in labels.columns:
        issues.append('labels.csv missing fruit_type and/or firmness_available columns')
        return issues

    strawberries = labels[labels['fruit_type'].astype(str).str.lower() == 'strawberry']
    if not strawberries.empty:
        bad = strawberries[_truthy_values(strawberries['firmness_available'])]
        if not bad.empty:
            issues.append(
                f'{len(bad)} strawberry rows have firmness_available=True; strawberry should normally use False or missing'
            )

    avocados = labels[labels['fruit_type'].astype(str).str.lower() == 'avocado']
    if not avocados.empty and 'firmness_avg' in avocados.columns:
        missing = avocados[~_truthy_values(avocados['firmness_available']) & avocados['firmness_avg'].notna()]
        if not missing.empty:
            issues.append(
                f'{len(missing)} avocado rows have firmness_avg but firmness_available=False'
            )
    return issues


def collect_validation_issues(
    frame_manifest: pd.DataFrame,
    labels: pd.DataFrame,
    eol_anchors: pd.DataFrame,
) -> dict[str, list[str]]:
    return {
        'labels_eol_consistency': validate_labels_eol_consistency(labels, eol_anchors),
        'frame_manifest_roi_timeline': validate_frame_manifest_roi_timeline(frame_manifest),
        'firmness_availability': validate_firmness_available(labels),
    }
