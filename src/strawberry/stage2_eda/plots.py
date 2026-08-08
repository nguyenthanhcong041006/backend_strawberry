from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from .io import display_path

Image = None
try:
    from PIL import Image
except Exception:
    Image = None


def plot_time_series(df: pd.DataFrame, timestamp_col: str, value_col: str, output_path: Path, title: str) -> None:
    if timestamp_col not in df.columns or value_col not in df.columns or df.empty:
        return
    data = df[[timestamp_col, value_col]].copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors='coerce')
    data[value_col] = pd.to_numeric(data[value_col], errors='coerce')
    data = data.dropna(subset=[timestamp_col, value_col])
    if data.empty:
        return
    data = data.sort_values(timestamp_col)
    plt.figure(figsize=(10, 5))
    plt.plot(data[timestamp_col], data[value_col], marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('timestamp')
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved graph: {display_path(output_path)}')


def plot_histogram(df: pd.DataFrame, column: str, output_path: Path, title: str, bins: int = 20) -> None:
    if column not in df.columns or df.empty:
        return
    values = pd.to_numeric(df[column], errors='coerce').dropna()
    if values.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved graph: {display_path(output_path)}')


def plot_counts_by_column(df: pd.DataFrame, column: str, output_path: Path, title: str) -> None:
    if column not in df.columns or df.empty:
        return
    counts = df[column].astype(str).value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    counts.plot(kind='bar')
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved graph: {display_path(output_path)}')


def plot_frame_counts_by_date(frame_manifest: pd.DataFrame, output_path: Path) -> None:
    if 'timestamp' not in frame_manifest.columns or frame_manifest.empty:
        return
    dates = pd.to_datetime(frame_manifest['timestamp'], errors='coerce').dropna()
    if dates.empty:
        return
    counts = dates.dt.date.value_counts().sort_index()
    plt.figure(figsize=(12, 5))
    counts.plot(kind='bar')
    plt.title('Frame count by date')
    plt.xlabel('date')
    plt.ylabel('frame count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved graph: {display_path(output_path)}')


def plot_frame_counts_by_date_and_fruit(frame_manifest: pd.DataFrame, output_path: Path) -> None:
    if 'timestamp' not in frame_manifest.columns or 'fruit_id' not in frame_manifest.columns or frame_manifest.empty:
        return
    data = frame_manifest[['timestamp', 'fruit_id']].copy()
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data = data.dropna(subset=['timestamp', 'fruit_id'])
    if data.empty:
        return
    data['date'] = data['timestamp'].dt.date
    grouped = data.groupby(['date', 'fruit_id']).size().unstack(fill_value=0)
    if grouped.empty:
        return
    plt.figure(figsize=(14, 6))
    grouped.plot(kind='bar', stacked=True, width=0.8)
    plt.title('Frame count by date and fruit ID')
    plt.xlabel('date')
    plt.ylabel('frame count')
    plt.xticks(rotation=45)
    plt.legend(title='fruit_id', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved graph: {display_path(output_path)}')


def plot_mask_valid_ratio_over_time(frame_manifest: pd.DataFrame, output_path: Path) -> None:
    if 'timestamp' not in frame_manifest.columns or 'mask_valid' not in frame_manifest.columns or frame_manifest.empty:
        return
    data = frame_manifest[['timestamp', 'mask_valid']].copy()
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data['mask_valid'] = pd.to_numeric(data['mask_valid'], errors='coerce')
    data = data.dropna(subset=['timestamp', 'mask_valid'])
    if data.empty:
        return
    ratio = data.groupby(data['timestamp'].dt.date)['mask_valid'].mean().sort_index()
    plt.figure(figsize=(12, 5))
    ratio.plot(marker='o')
    plt.title('Mask-valid ratio by date')
    plt.xlabel('date')
    plt.ylabel('mask valid ratio')
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved graph: {display_path(output_path)}')


def plot_mask_valid_over_time(frame_manifest: pd.DataFrame, output_path: Path) -> None:
    if 'timestamp' not in frame_manifest.columns or 'mask_valid' not in frame_manifest.columns or frame_manifest.empty:
        return
    data = frame_manifest[['timestamp', 'mask_valid']].copy()
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data['mask_valid'] = pd.to_numeric(data['mask_valid'], errors='coerce')
    data = data.dropna(subset=['timestamp', 'mask_valid'])
    if data.empty:
        return
    counts = data.groupby(data['timestamp'].dt.date)['mask_valid'].sum().sort_index()
    plt.figure(figsize=(12, 5))
    counts.plot(marker='o')
    plt.title('Mask-valid count by date')
    plt.xlabel('date')
    plt.ylabel('mask valid count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved graph: {display_path(output_path)}')


def plot_roi_position_distribution(excluded_frames: pd.DataFrame, output_path: Path) -> None:
    if 'roi_position' not in excluded_frames.columns or excluded_frames.empty:
        return
    counts = excluded_frames['roi_position'].astype(str).value_counts().sort_index()
    if counts.empty:
        return
    plt.figure(figsize=(10, 5))
    counts.plot(kind='bar')
    plt.title('Excluded frames by ROI position')
    plt.xlabel('ROI position')
    plt.ylabel('frame count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved graph: {display_path(output_path)}')


def plot_temporal_image_strip(df: pd.DataFrame, fruit_id: str, output_path: Path) -> None:
    if Image is None or 'timestamp' not in df.columns or 'image_path' not in df.columns or df.empty:
        return
    data = df[['timestamp', 'image_path']].copy()
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data = data.dropna(subset=['timestamp', 'image_path']).sort_values('timestamp')
    if data.empty:
        return
    image_paths = [Path(p) for p in data['image_path'].astype(str).tolist()[:6] if Path(p).exists()]
    if not image_paths:
        return
    images = []
    for image_path in image_paths:
        try:
            images.append(Image.open(image_path).convert('RGB'))
        except Exception:
            continue
    if not images:
        return
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    canvas = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        canvas.paste(img, (x_offset, 0))
        x_offset += img.width
    canvas.save(output_path)
    print(f'Saved image strip: {display_path(output_path)}')


def plot_rul_distribution_by_fruit(labels: pd.DataFrame, output_path: Path) -> None:
    if 'fruit_id' not in labels.columns or 'rul_hours' not in labels.columns or labels.empty:
        return
    plot_boxplot(labels, 'fruit_id', 'rul_hours', output_path, 'RUL distribution by fruit ID')


def plot_boxplot(df: pd.DataFrame, category_col: str, value_col: str, output_path: Path, title: str) -> None:
    if category_col not in df.columns or value_col not in df.columns or df.empty:
        return
    data = df[[category_col, value_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors='coerce')
    data = data.dropna()
    if data.empty:
        return
    plt.figure(figsize=(12, 6))
    data.boxplot(column=value_col, by=category_col, rot=45)
    plt.title(title)
    plt.suptitle('')
    plt.xlabel(category_col)
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved graph: {display_path(output_path)}')


def plot_firmness_trend(labels: pd.DataFrame, output_path: Path) -> None:
    if 'timestamp' not in labels.columns or 'firmness_avg' not in labels.columns or labels.empty:
        return
    plot_time_series(labels, 'timestamp', 'firmness_avg', output_path, 'Firmness trend over time')


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, output_path: Path, title: str) -> None:
    if x_col not in df.columns or y_col not in df.columns or df.empty:
        return
    x = pd.to_numeric(df[x_col], errors='coerce')
    y = pd.to_numeric(df[y_col], errors='coerce')
    data = pd.concat([x, y], axis=1).dropna()
    if data.empty:
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], alpha=0.6)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved graph: {display_path(output_path)}')


def plot_color_trends(labels: pd.DataFrame, output_dir: Path) -> None:
    possible = [
        'color_r_mean',
        'color_g_mean',
        'color_b_mean',
        'lab_L',
        'lab_a',
        'lab_b',
        'r_mean',
        'g_mean',
        'b_mean',
        'red_mean',
        'green_mean',
        'blue_mean',
    ]
    cols = [c for c in possible if c in labels.columns]
    if not cols or labels.empty:
        return
    for c in cols:
        plot_time_series(labels, 'timestamp', c, output_dir / f'{c}_over_time.png', f'{c} over time')


def plot_roi_layout_comparison(frame_manifest: pd.DataFrame, output_path: Path) -> None:
    if frame_manifest.empty:
        return
    if 'roi_position' in frame_manifest.columns:
        key = 'roi_position'
    elif 'roi_id' in frame_manifest.columns:
        key = 'roi_id'
    else:
        return
    if 'fruit_type' in frame_manifest.columns:
        data = frame_manifest.groupby([key, 'fruit_type']).size().unstack(fill_value=0)
        if data.empty:
            return
        ax = data.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title('ROI layout comparison by fruit type')
    else:
        data = frame_manifest[key].astype(str).value_counts().sort_index()
        if data.empty:
            return
        ax = data.plot(kind='bar', figsize=(12, 6))
        plt.title('ROI layout comparison')
    ax.set_xlabel(key)
    ax.set_ylabel('frame count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved graph: {display_path(output_path)}')


def plot_eol_anchor_status(eol_anchors: pd.DataFrame, output_path: Path) -> None:
    if eol_anchors.empty:
        return
    status_col = 'status' if 'status' in eol_anchors.columns else 'label_status' if 'label_status' in eol_anchors.columns else None
    if status_col is None:
        return
    if 'fruit_type' in eol_anchors.columns:
        data = eol_anchors.groupby([status_col, 'fruit_type']).size().unstack(fill_value=0)
        if data.empty:
            return
        ax = data.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title('EOL anchor status by fruit type')
    else:
        data = eol_anchors[status_col].astype(str).value_counts().sort_values(ascending=False)
        if data.empty:
            return
        ax = data.plot(kind='bar', figsize=(10, 5))
        plt.title('EOL anchor status')
    ax.set_xlabel(status_col)
    ax.set_ylabel('count')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved graph: {display_path(output_path)}')


def plot_eol_timestamp_distribution(eol_anchors: pd.DataFrame, output_path: Path) -> None:
    if eol_anchors.empty or 'eol_timestamp' not in eol_anchors.columns:
        return
    data = eol_anchors[['eol_timestamp', 'fruit_type']].copy() if 'fruit_type' in eol_anchors.columns else eol_anchors[['eol_timestamp']].copy()
    data['eol_timestamp'] = pd.to_datetime(data['eol_timestamp'], errors='coerce')
    data = data.dropna(subset=['eol_timestamp'])
    if data.empty:
        return
    data['date'] = data['eol_timestamp'].dt.date
    if 'fruit_type' in data.columns:
        grouped = data.groupby(['date', 'fruit_type']).size().unstack(fill_value=0)
        if grouped.empty:
            return
        ax = grouped.plot(kind='bar', stacked=True, figsize=(14, 6))
        plt.title('EOL timestamp distribution by fruit type')
    else:
        counts = data['date'].value_counts().sort_index()
        if counts.empty:
            return
        ax = counts.plot(kind='bar', figsize=(14, 6))
        plt.title('EOL timestamp distribution')
    ax.set_xlabel('date')
    ax.set_ylabel('count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved graph: {display_path(output_path)}')


def plot_mapping_delta_histogram(numeric_mapping: pd.DataFrame, output_path: Path) -> None:
    if numeric_mapping.empty or 'mapping_delta_seconds' not in numeric_mapping.columns:
        return
    values = pd.to_numeric(numeric_mapping['mapping_delta_seconds'], errors='coerce').dropna()
    if values.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=30, edgecolor='black')
    plt.title('Numeric mapping delta seconds')
    plt.xlabel('delta seconds')
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved graph: {display_path(output_path)}')


def plot_mapping_method_counts(numeric_mapping: pd.DataFrame, output_path: Path) -> None:
    if numeric_mapping.empty or 'mapping_method' not in numeric_mapping.columns:
        return
    if 'fruit_type' in numeric_mapping.columns:
        data = numeric_mapping.groupby(['mapping_method', 'fruit_type']).size().unstack(fill_value=0)
        if data.empty:
            return
        ax = data.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title('Numeric mapping method counts by fruit type')
        ax.set_xlabel('mapping_method')
    else:
        counts = numeric_mapping['mapping_method'].astype(str).value_counts().sort_values(ascending=False)
        if counts.empty:
            return
        ax = counts.plot(kind='bar', figsize=(10, 5))
        plt.title('Numeric mapping method counts')
        ax.set_xlabel('mapping_method')
    ax.set_ylabel('count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved graph: {display_path(output_path)}')


def plot_numeric_mapping_status_counts(numeric_mapping: pd.DataFrame, output_path: Path) -> None:
    if numeric_mapping.empty:
        return
    if 'sensor_status' in numeric_mapping.columns:
        counts = numeric_mapping['sensor_status'].astype(str).value_counts().sort_values(ascending=False)
        if counts.empty:
            return
        plt.figure(figsize=(10, 5))
        counts.plot(kind='bar')
        plt.title('Numeric mapping sensor_status counts')
        plt.xlabel('sensor_status')
        plt.ylabel('count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f'Saved graph: {display_path(output_path)}')
        return
    anomaly_cols = [c for c in ['temperature_anomaly', 'humidity_anomaly'] if c in numeric_mapping.columns]
    if anomaly_cols:
        summary = {c: int(numeric_mapping[c].astype(bool).sum()) for c in anomaly_cols}
        if not summary:
            return
        plt.figure(figsize=(10, 5))
        pd.Series(summary).plot(kind='bar')
        plt.title('Numeric mapping anomaly counts')
        plt.xlabel('anomaly type')
        plt.ylabel('count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f'Saved graph: {display_path(output_path)}')
