from __future__ import annotations

import csv
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
import tkinter as tk

import pandas as pd
from PIL import Image, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEGMENTED_DIR = PROJECT_ROOT / "data" / "02_processed" / "avocado" / "segmented"
DEFAULT_FIRMNESS_CSV = PROJECT_ROOT / "data" / "01_raw" / "avocado" / "hardness" / "hardness.csv"
DEFAULT_ENV_CSV = PROJECT_ROOT / "data" / "01_raw" / "avocado" / "th10s_readings.csv"
DEFAULT_FEATURES_CSV = PROJECT_ROOT / "data" / "02_processed" / "avocado" / "eda" / "features" / "avocado_features.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "02_processed" / "avocado" / "labels" / "eol_annotations.csv"
DEFAULT_EXCLUSIONS_CSV = PROJECT_ROOT / "data" / "02_processed" / "avocado" / "labels" / "frame_exclusions.csv"
TIMESTAMP_PATTERN = re.compile(r"webcam_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_fruit_(\d+)\.png$")


@dataclass(frozen=True)
class FruitFrame:
    fruit_id: str
    timestamp: pd.Timestamp
    image_path: Path


def parse_frame_timestamp(path: Path) -> pd.Timestamp:
    match = TIMESTAMP_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse timestamp from {path.name}")
    date_text, time_text, _fruit_num = match.groups()
    return pd.Timestamp(datetime.strptime(f"{date_text} {time_text}", "%Y-%m-%d %H-%M-%S"))


def load_frames(segmented_dir: Path) -> dict[str, list[FruitFrame]]:
    frames_by_fruit: dict[str, list[FruitFrame]] = {}
    for fruit_dir in sorted(segmented_dir.glob("F*")):
        if not fruit_dir.is_dir():
            continue
        frames: list[FruitFrame] = []
        for image_path in sorted(fruit_dir.glob("*.png")):
            try:
                frames.append(
                    FruitFrame(
                        fruit_id=fruit_dir.name,
                        timestamp=parse_frame_timestamp(image_path),
                        image_path=image_path,
                    )
                )
            except ValueError:
                continue
        if frames:
            frames_by_fruit[fruit_dir.name] = sorted(frames, key=lambda row: row.timestamp)
    return frames_by_fruit


def load_firmness(firmness_csv: Path) -> pd.DataFrame:
    if not firmness_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(firmness_csv)
    df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    return df


def load_environment(env_csv: Path) -> pd.DataFrame:
    if not env_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(env_csv)
    # The webcam filenames and sensor rows share the same clock text; strip timezone for UI alignment.
    df["datetime"] = pd.to_datetime(df["timestamp"].str[:19])
    return df.sort_values("datetime").reset_index(drop=True)


def load_features(features_csv: Path) -> pd.DataFrame:
    if not features_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(features_csv)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["fruit_id", "timestamp"]).reset_index(drop=True)


def firmness_column(fruit_id: str) -> str:
    return f"fruit_{int(fruit_id[1:])}"


def firmness_context(firmness_df: pd.DataFrame, fruit_id: str, timestamp: pd.Timestamp) -> dict[str, object]:
    if firmness_df.empty:
        return {
            "firmness_latest": "",
            "firmness_date": "",
            "first_zero_date": "",
            "first_le5_date": "",
        }
    col = firmness_column(fruit_id)
    if col not in firmness_df.columns:
        return {
            "firmness_latest": "",
            "firmness_date": "",
            "first_zero_date": "",
            "first_le5_date": "",
        }

    current_date = timestamp.normalize()
    previous = firmness_df[firmness_df["date"] <= current_date]
    latest_value = ""
    latest_date = ""
    if not previous.empty:
        latest_row = previous.iloc[-1]
        latest_value = latest_row[col]
        latest_date = latest_row["date"].date().isoformat()

    first_zero = firmness_df.loc[firmness_df[col] <= 0, "date"]
    first_le5 = firmness_df.loc[firmness_df[col] <= 5, "date"]
    return {
        "firmness_latest": latest_value,
        "firmness_date": latest_date,
        "first_zero_date": first_zero.iloc[0].date().isoformat() if not first_zero.empty else "",
        "first_le5_date": first_le5.iloc[0].date().isoformat() if not first_le5.empty else "",
    }


def environment_context(env_df: pd.DataFrame, timestamp: pd.Timestamp) -> dict[str, object]:
    if env_df.empty:
        return {
            "temperature_c": "",
            "humidity_rh": "",
            "env_timestamp": "",
        }
    deltas = (env_df["datetime"] - timestamp).abs()
    row = env_df.loc[deltas.idxmin()]
    return {
        "temperature_c": row["temperature_c"],
        "humidity_rh": row["humidity_rh"],
        "env_timestamp": row["datetime"],
    }


def feature_context(feature_df: pd.DataFrame, fruit_id: str, timestamp: pd.Timestamp) -> dict[str, object]:
    if feature_df.empty:
        return {}
    if "fruit_id" not in feature_df.columns or "timestamp" not in feature_df.columns:
        return {}
    fruit_df = feature_df[feature_df["fruit_id"] == fruit_id]
    if fruit_df.empty:
        return {}
    deltas = (fruit_df["timestamp"] - timestamp).abs()
    row = fruit_df.loc[deltas.idxmin()]
    return row.to_dict()


def load_existing_annotations(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["fruit_id"]: row for row in reader if row.get("fruit_id")}


def load_existing_exclusions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row.get("fruit_id")]


class EolLabelingApp:
    def __init__(
        self,
        root: tk.Tk,
        segmented_dir: Path = DEFAULT_SEGMENTED_DIR,
        firmness_csv: Path = DEFAULT_FIRMNESS_CSV,
        env_csv: Path = DEFAULT_ENV_CSV,
        features_csv: Path = DEFAULT_FEATURES_CSV,
        output_csv: Path = DEFAULT_OUTPUT_CSV,
        exclusions_csv: Path = DEFAULT_EXCLUSIONS_CSV,
    ) -> None:
        self.root = root
        self.root.title("Avocado EOL Labeling")
        self.root.geometry("1860x1080")

        self.segmented_dir = segmented_dir
        self.output_csv = output_csv
        self.exclusions_csv = exclusions_csv
        self.frames_by_fruit = load_frames(segmented_dir)
        if not self.frames_by_fruit:
            raise RuntimeError(f"No segmented fruit images found in {segmented_dir}")

        self.firmness_df = load_firmness(firmness_csv)
        self.env_df = load_environment(env_csv)
        self.feature_df = load_features(features_csv)
        self.annotations = load_existing_annotations(output_csv)
        self.exclusions = load_existing_exclusions(exclusions_csv)
        self.fruit_ids = sorted(self.frames_by_fruit)
        self.current_fruit = self.fruit_ids[0]
        self.selected_day: str = "All days"
        self.frame_idx = 0
        self.current_image: ImageTk.PhotoImage | None = None
        self.image_cache: OrderedDict[Path, ImageTk.PhotoImage] = OrderedDict()
        self.max_cached_images = 180
        self.slider_job: str | None = None
        self.suppress_slider_callback = False
        self.playing = False
        self.play_job: str | None = None

        self.basis_var = tk.StringVar(value="visible_mold_onset")
        self.confidence_var = tk.StringVar(value="medium")
        self.note_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self.timestamp_var = tk.StringVar(value="")
        self.firmness_var = tk.StringVar(value="")
        self.environment_var = tk.StringVar(value="")
        self.eol_var = tk.StringVar(value="")
        self.notation_var = tk.StringVar(value="")
        self.play_delay_var = tk.IntVar(value=350)
        self.play_skip_var = tk.IntVar(value=10)
        self.repeat_playback_var = tk.BooleanVar(value=False)
        self.shuffle_playback_var = tk.BooleanVar(value=False)
        self.start_from_eol_var = tk.BooleanVar(value=False)
        self.rgb_var = tk.StringVar(value="")
        self.lab_var = tk.StringVar(value="")
        self.day_var = tk.StringVar(value="All days")
        self.cut_start_idx: int | None = None
        self.cut_end_idx: int | None = None
        self.cut_range_var = tk.StringVar(value="Cut range: not set")
        self.cut_reason_var = tk.StringVar(value="bad_segmentation")
        self.cut_note_var = tk.StringVar(value="")

        self._build_ui()
        self._sync_day_selector()
        self._load_annotation_for_current_fruit()
        self._update_view()

    @property
    def all_current_frames(self) -> list[FruitFrame]:
        return self.frames_by_fruit[self.current_fruit]

    @property
    def current_frames(self) -> list[FruitFrame]:
        if self.selected_day == "All days":
            return self.all_current_frames
        selected_date = pd.Timestamp(self.selected_day).date()
        return [frame for frame in self.all_current_frames if frame.timestamp.date() == selected_date]

    def _day_options_for_current_fruit(self) -> list[str]:
        days = sorted({frame.timestamp.date().isoformat() for frame in self.all_current_frames})
        return ["All days", *days]

    def _sync_day_selector(self) -> None:
        options = self._day_options_for_current_fruit()
        self.day_combo.configure(values=options)
        if self.selected_day not in options:
            self.selected_day = "All days"
        self.day_var.set(self.selected_day)

    def _apply_day_filter(self, day_text: str) -> None:
        self.selected_day = day_text
        self.day_var.set(day_text)
        self._stop_playback()
        self.cut_start_idx = None
        self.cut_end_idx = None
        self._update_cut_range_text()
        self.frame_idx = 0
        self.slider.configure(to=max(0, len(self.current_frames) - 1))
        self._set_slider(0)
        self._update_view()

    @property
    def current_frame(self) -> FruitFrame:
        return self.current_frames[self.frame_idx]

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, padding=12)
        root_frame.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root_frame, width=310)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        right = ttk.Frame(root_frame)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="Fruit").pack(anchor=tk.W)
        self.fruit_combo = ttk.Combobox(left, values=self.fruit_ids, state="readonly")
        self.fruit_combo.set(self.current_fruit)
        self.fruit_combo.bind("<<ComboboxSelected>>", self._on_fruit_change)
        self.fruit_combo.pack(fill=tk.X, pady=(4, 12))

        ttk.Label(left, text="Day").pack(anchor=tk.W)
        self.day_combo = ttk.Combobox(left, textvariable=self.day_var, state="readonly")
        self.day_combo.bind("<<ComboboxSelected>>", self._on_day_change)
        self.day_combo.pack(fill=tk.X, pady=(4, 12))

        ttk.Label(left, text="Timeline").pack(anchor=tk.W)
        self.slider = ttk.Scale(
            left,
            from_=0,
            to=max(0, len(self.current_frames) - 1),
            orient=tk.HORIZONTAL,
            command=self._on_slider_change,
        )
        self.slider.pack(fill=tk.X, pady=(4, 8))

        nav = ttk.Frame(left)
        nav.pack(fill=tk.X, pady=(0, 12))
        ttk.Button(nav, text="Prev", command=lambda: self._step(-1)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(nav, text="Next", command=lambda: self._step(1)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(8, 0))

        play_row = ttk.Frame(left)
        play_row.pack(fill=tk.X, pady=(0, 12))
        self.play_button = ttk.Button(play_row, text="Play", command=self._toggle_play)
        self.play_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        control_box = ttk.Frame(play_row)
        control_box.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(control_box, text="Delay ms").pack(anchor=tk.W)
        self.delay_spin = ttk.Spinbox(
            control_box,
            from_=50,
            to=5000,
            increment=50,
            textvariable=self.play_delay_var,
            width=8,
        )
        self.delay_spin.pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(control_box, text="Skip frames").pack(anchor=tk.W, pady=(6, 0))
        self.skip_spin = ttk.Spinbox(
            control_box,
            from_=1,
            to=500,
            increment=1,
            textvariable=self.play_skip_var,
            width=8,
        )
        self.skip_spin.pack(anchor=tk.W, pady=(4, 0))
        ttk.Checkbutton(control_box, text="Repeat", variable=self.repeat_playback_var).pack(anchor=tk.W, pady=(8, 0))
        ttk.Checkbutton(control_box, text="Shuffle fruit", variable=self.shuffle_playback_var).pack(anchor=tk.W, pady=(4, 0))
        ttk.Checkbutton(control_box, text="Start from EOL", variable=self.start_from_eol_var).pack(anchor=tk.W, pady=(4, 0))

        ttk.Label(left, textvariable=self.timestamp_var, wraplength=300).pack(anchor=tk.W, pady=(0, 8))

        cut_frame = ttk.LabelFrame(left, text="Cut failed frame range")
        cut_frame.pack(fill=tk.X, pady=(4, 12))
        cut_buttons = ttk.Frame(cut_frame)
        cut_buttons.pack(fill=tk.X, padx=6, pady=(6, 4))
        ttk.Button(cut_buttons, text="Set Start", command=self._set_cut_start).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(cut_buttons, text="Set End", command=self._set_cut_end).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))
        ttk.Label(cut_frame, textvariable=self.cut_range_var, wraplength=285).pack(anchor=tk.W, padx=6, pady=(0, 6))

        ttk.Label(cut_frame, text="Reason").pack(anchor=tk.W, padx=6)
        self.cut_reason_combo = ttk.Combobox(
            cut_frame,
            values=[
                "bad_segmentation",
                "blur",
                "wrong_crop",
                "lighting_artifact",
                "occlusion",
                "duplicate_or_stalled_frame",
                "other",
            ],
            textvariable=self.cut_reason_var,
            state="readonly",
        )
        self.cut_reason_combo.pack(fill=tk.X, padx=6, pady=(4, 6))

        ttk.Label(cut_frame, text="Cut note").pack(anchor=tk.W, padx=6)
        self.cut_note_entry = ttk.Entry(cut_frame, textvariable=self.cut_note_var)
        self.cut_note_entry.pack(fill=tk.X, padx=6, pady=(4, 6))
        ttk.Button(cut_frame, text="Add Cut Range", command=self._add_cut_range).pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(cut_frame, text="Save Cut Ranges", command=self._save_exclusions).pack(fill=tk.X, padx=6, pady=(0, 8))

        ttk.Label(left, text="EOL basis").pack(anchor=tk.W)
        self.basis_combo = ttk.Combobox(
            left,
            values=[
                "visible_mold_onset",
                "visible_spoilage_onset",
                "firmness_zero_visual_confirmed",
                "manual_review",
            ],
            textvariable=self.basis_var,
            state="readonly",
        )
        self.basis_combo.pack(fill=tk.X, pady=(4, 12))

        ttk.Label(left, text="Confidence").pack(anchor=tk.W)
        self.conf_combo = ttk.Combobox(
            left,
            values=["high", "medium", "low"],
            textvariable=self.confidence_var,
            state="readonly",
        )
        self.conf_combo.pack(fill=tk.X, pady=(4, 12))

        ttk.Label(left, text="Visual note").pack(anchor=tk.W)
        self.note_entry = ttk.Entry(left, textvariable=self.note_var)
        self.note_entry.pack(fill=tk.X, pady=(4, 12))

        ttk.Button(left, text="Mark Current Frame As EOL", command=self._mark_current_as_eol).pack(fill=tk.X, pady=(8, 6))
        ttk.Button(left, text="Save All Annotations", command=self._save_annotations).pack(fill=tk.X, pady=(0, 12))

        ttk.Label(left, textvariable=self.eol_var, wraplength=300).pack(anchor=tk.W, pady=(0, 8))
        ttk.Label(left, textvariable=self.status_var, wraplength=300).pack(anchor=tk.W)

        display = ttk.Frame(right)
        display.pack(fill=tk.BOTH, expand=True)

        image_panel = ttk.Frame(display)
        image_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_label = ttk.Label(image_panel, anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        side_panel = ttk.Frame(display, width=420)
        side_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        side_panel.pack_propagate(False)

        ttk.Label(side_panel, text="Environment", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.env_canvas = tk.Canvas(side_panel, width=400, height=230, bg="white", highlightthickness=1, highlightbackground="#d1d5db")
        self.env_canvas.pack(fill=tk.X, pady=(4, 8))
        ttk.Label(side_panel, textvariable=self.environment_var, wraplength=390).pack(anchor=tk.W, pady=(0, 12))

        ttk.Label(side_panel, text="Firmness", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.firmness_canvas = tk.Canvas(side_panel, width=400, height=230, bg="white", highlightthickness=1, highlightbackground="#d1d5db")
        self.firmness_canvas.pack(fill=tk.X, pady=(4, 8))
        ttk.Label(side_panel, textvariable=self.firmness_var, wraplength=390).pack(anchor=tk.W)
        ttk.Label(side_panel, text="Notations", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(12, 0))
        ttk.Label(side_panel, textvariable=self.notation_var, wraplength=390).pack(anchor=tk.W, pady=(4, 0))

        trends = ttk.Frame(right)
        trends.pack(fill=tk.X, pady=(10, 0))

        rgb_box = ttk.Frame(trends)
        rgb_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        ttk.Label(rgb_box, text="RGB over time", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(rgb_box, textvariable=self.rgb_var, wraplength=740).pack(anchor=tk.W, pady=(0, 4))
        self.rgb_canvas = tk.Canvas(rgb_box, height=210, bg="white", highlightthickness=1, highlightbackground="#d1d5db")
        self.rgb_canvas.pack(fill=tk.X)

        lab_box = ttk.Frame(trends)
        lab_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(lab_box, text="LAB over time", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(lab_box, textvariable=self.lab_var, wraplength=740).pack(anchor=tk.W, pady=(0, 4))
        self.lab_canvas = tk.Canvas(lab_box, height=210, bg="white", highlightthickness=1, highlightbackground="#d1d5db")
        self.lab_canvas.pack(fill=tk.X)

    def _on_fruit_change(self, _event: object = None) -> None:
        self._stop_playback()
        self.current_fruit = self.fruit_combo.get()
        self.selected_day = "All days"
        self._sync_day_selector()
        self.frame_idx = 0
        self.slider.configure(to=len(self.current_frames) - 1)
        self._set_slider(0)
        self._load_annotation_for_current_fruit()
        self._update_view()

    def _on_day_change(self, _event: object = None) -> None:
        self._apply_day_filter(self.day_var.get())

    def _on_slider_change(self, value: str) -> None:
        if self.suppress_slider_callback:
            return
        self.frame_idx = int(float(value))
        if self.slider_job is not None:
            self.root.after_cancel(self.slider_job)
        self.slider_job = self.root.after(80, self._run_deferred_slider_update)

    def _run_deferred_slider_update(self) -> None:
        self.slider_job = None
        self._update_view()

    def _set_slider(self, value: int) -> None:
        self.suppress_slider_callback = True
        self.slider.set(value)
        self.suppress_slider_callback = False

    def _set_frame_idx(self, value: int) -> None:
        self.frame_idx = max(0, min(len(self.current_frames) - 1, value))
        self._set_slider(self.frame_idx)
        self._update_view()

    def _step(self, delta: int) -> None:
        self._set_frame_idx(self.frame_idx + delta)

    def _set_cut_start(self) -> None:
        self.cut_start_idx = self.frame_idx
        self._update_cut_range_text()
        self._update_view()

    def _set_cut_end(self) -> None:
        self.cut_end_idx = self.frame_idx
        self._update_cut_range_text()
        self._update_view()

    def _cut_endpoint_text(self, frame_idx: int | None) -> str:
        if frame_idx is None:
            return "not set"
        frame = self.current_frames[frame_idx]
        return f"{frame_idx + 1} ({frame.timestamp})"

    def _update_cut_range_text(self) -> None:
        self.cut_range_var.set(
            "Cut range: "
            f"start {self._cut_endpoint_text(self.cut_start_idx)} | "
            f"end {self._cut_endpoint_text(self.cut_end_idx)}"
        )

    def _add_cut_range(self) -> None:
        if self.cut_start_idx is None or self.cut_end_idx is None:
            messagebox.showwarning("Cut range incomplete", "Set both cut start and cut end before adding a range.")
            return

        start_idx, end_idx = sorted([self.cut_start_idx, self.cut_end_idx])
        start_frame = self.current_frames[start_idx]
        end_frame = self.current_frames[end_idx]
        self.exclusions.append(
            {
                "fruit_id": self.current_fruit,
                "start_frame_idx": str(start_idx),
                "end_frame_idx": str(end_idx),
                "start_timestamp": start_frame.timestamp.isoformat(sep=" "),
                "end_timestamp": end_frame.timestamp.isoformat(sep=" "),
                "start_image_path": str(start_frame.image_path),
                "end_image_path": str(end_frame.image_path),
                "frame_count": str(end_idx - start_idx + 1),
                "reason": self.cut_reason_var.get(),
                "note": self.cut_note_var.get(),
                "label_status": "proposed_exclusion",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self.status_var.set(
            f"Added cut range for {self.current_fruit}: frames {start_idx + 1}-{end_idx + 1}. Save cuts when ready."
        )
        self.cut_start_idx = None
        self.cut_end_idx = None
        self.cut_note_var.set("")
        self._update_cut_range_text()
        self._update_view()

    def _current_fruit_exclusions(self) -> list[dict[str, str]]:
        return [row for row in self.exclusions if row.get("fruit_id") == self.current_fruit]

    def _active_cut_timestamps(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        start = self.current_frames[self.cut_start_idx].timestamp if self.cut_start_idx is not None else None
        end = self.current_frames[self.cut_end_idx].timestamp if self.cut_end_idx is not None else None
        return start, end

    def _toggle_play(self) -> None:
        if self.playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        self._cancel_deferred_slider_update()
        if self.shuffle_playback_var.get():
            self._jump_to_random_fruit_for_playback()
        if self.start_from_eol_var.get():
            self._move_to_eol_or_start()
        else:
            self._set_frame_idx(0)
        self.playing = True
        self.play_button.configure(text="Stop")
        self.status_var.set("Playback running")
        self._schedule_next_frame()

    def _cancel_deferred_slider_update(self) -> None:
        if self.slider_job is not None:
            try:
                self.root.after_cancel(self.slider_job)
            except tk.TclError:
                pass
            self.slider_job = None

    def _stop_playback(self) -> None:
        self.playing = False
        self.play_button.configure(text="Play")
        if self.play_job is not None:
            try:
                self.root.after_cancel(self.play_job)
            except tk.TclError:
                pass
            self.play_job = None
        self._cancel_deferred_slider_update()

    def _schedule_next_frame(self) -> None:
        if not self.playing:
            return
        delay = max(50, int(self.play_delay_var.get() or 0))
        self.play_job = self.root.after(delay, self._play_step)

    def _jump_to_random_fruit_for_playback(self) -> None:
        choices = [fruit_id for fruit_id in self.fruit_ids if fruit_id != self.current_fruit]
        if not choices:
            self._set_frame_idx(0)
            return
        self.current_fruit = random.choice(choices)
        self.fruit_combo.set(self.current_fruit)
        self._sync_day_selector()
        self.cut_start_idx = None
        self.cut_end_idx = None
        self._update_cut_range_text()
        self.frame_idx = 0
        self.slider.configure(to=max(0, len(self.current_frames) - 1))
        self._set_slider(0)
        self._load_annotation_for_current_fruit()
        self._update_view()

    def _move_to_eol_or_start(self) -> None:
        eol_timestamp = self._annotated_eol_timestamp()
        if eol_timestamp is None:
            self._set_frame_idx(0)
            return
        if self.selected_day != "All days" and not any(frame.timestamp.date() == eol_timestamp.date() for frame in self.current_frames):
            self.selected_day = "All days"
            self._sync_day_selector()
            self.slider.configure(to=max(0, len(self.current_frames) - 1))
        nearest_idx = min(
            range(len(self.current_frames)),
            key=lambda idx: abs(self.current_frames[idx].timestamp - eol_timestamp),
        )
        self._set_frame_idx(nearest_idx)

    def _restart_playback_loop(self) -> None:
        if self.shuffle_playback_var.get():
            self._jump_to_random_fruit_for_playback()
            self.status_var.set(f"Playback shuffled to {self.current_fruit}")
            if self.start_from_eol_var.get():
                self._move_to_eol_or_start()
        else:
            if self.start_from_eol_var.get():
                self._move_to_eol_or_start()
                self.status_var.set("Playback repeating from EOL")
            else:
                self._set_frame_idx(0)
                self.status_var.set("Playback repeating from start")
        self._schedule_next_frame()

    def _play_step(self) -> None:
        self.play_job = None
        if not self.playing:
            return
        skip = max(1, int(self.play_skip_var.get() or 1))
        if self.frame_idx >= len(self.current_frames) - 1:
            if self.repeat_playback_var.get():
                self._restart_playback_loop()
            else:
                self._stop_playback()
                self.status_var.set("Playback reached the final frame")
            return
        next_idx = min(len(self.current_frames) - 1, self.frame_idx + skip)
        self._set_frame_idx(next_idx)
        if self.frame_idx >= len(self.current_frames) - 1:
            if self.repeat_playback_var.get():
                self._restart_playback_loop()
            else:
                self._stop_playback()
                self.status_var.set("Playback reached the final frame")
            return
        self._schedule_next_frame()

    def _load_annotation_for_current_fruit(self) -> None:
        row = self.annotations.get(self.current_fruit, {})
        self.basis_var.set(row.get("eol_basis", "visible_mold_onset"))
        self.confidence_var.set(row.get("eol_confidence", "medium"))
        self.note_var.set(row.get("visual_note", ""))

    def _annotated_eol_timestamp(self) -> pd.Timestamp | None:
        row = self.annotations.get(self.current_fruit)
        if not row:
            return None
        timestamp = row.get("eol_timestamp", "")
        if not timestamp:
            return None
        try:
            return pd.Timestamp(timestamp)
        except Exception:
            return None

    def _update_view(self) -> None:
        frame = self.current_frame
        context = firmness_context(self.firmness_df, self.current_fruit, frame.timestamp)
        env = environment_context(self.env_df, frame.timestamp)
        features = feature_context(self.feature_df, self.current_fruit, frame.timestamp)
        eol_timestamp = self._annotated_eol_timestamp()
        self.timestamp_var.set(
            f"{self.current_fruit} | {self.selected_day} | frame {self.frame_idx + 1}/{len(self.current_frames)} | {frame.timestamp}"
        )
        self._update_notations()
        self.environment_var.set(
            "Environment: {temperature_c} C, {humidity_rh}% RH\n"
            "Sensor timestamp: {env_timestamp}".format(**env)
        )
        self.firmness_var.set(
            "Latest firmness: {firmness_latest} on {firmness_date}\n"
            "First <=5 date: {first_le5_date} | first zero date: {first_zero_date}".format(**context)
        )
        if features:
            self.rgb_var.set(
                "R={mean_r:.1f}  G={mean_g:.1f}  B={mean_b:.1f} | "
                "green ratio={green_ratio:.3f}".format(**features)
            )
            self.lab_var.set(
                "L*={mean_lab_l:.1f}  a*={mean_lab_a:.1f}  b*={mean_lab_b:.1f} | "
                "dark coverage={dark_coverage:.3f}".format(**features)
            )
        else:
            self.rgb_var.set("No RGB feature data")
            self.lab_var.set("No LAB feature data")
        existing = self.annotations.get(self.current_fruit)
        if existing:
            self.eol_var.set(
                f"Saved EOL: {existing.get('eol_timestamp', '')}\n"
                f"Basis: {existing.get('eol_basis', '')} | confidence: {existing.get('eol_confidence', '')}"
            )
        else:
            self.eol_var.set("Saved EOL: not marked")

        self.current_image = self._load_thumbnail(frame.image_path)
        self.image_label.configure(image=self.current_image)
        self._update_plots(frame.timestamp, eol_timestamp)

    def _update_notations(self) -> None:
        fruit_number = int(self.current_fruit[1:])
        day_note = "all captured days" if self.selected_day == "All days" else self.selected_day
        notations = [
            f"{self.current_fruit}: Fruit {fruit_number:02d}",
            "EOL: End-of-Life label",
            "CUT: excluded frame range",
            "RUL: remaining useful life in hours",
            f"Day: {day_note}",
        ]
        self.notation_var.set("\n".join(notations[:5]))

    def _load_thumbnail(self, image_path: Path) -> ImageTk.PhotoImage:
        cached = self.image_cache.get(image_path)
        if cached is not None:
            self.image_cache.move_to_end(image_path)
            return cached

        image = Image.open(image_path).convert("RGBA")
        image.thumbnail((760, 760), Image.Resampling.LANCZOS)
        thumbnail = ImageTk.PhotoImage(image)
        self.image_cache[image_path] = thumbnail
        while len(self.image_cache) > self.max_cached_images:
            self.image_cache.popitem(last=False)
        return thumbnail

    def _update_plots(self, timestamp: pd.Timestamp, eol_timestamp: pd.Timestamp | None) -> None:
        self._draw_environment_plot(timestamp, eol_timestamp)
        self._draw_firmness_plot(timestamp, eol_timestamp)
        self._draw_rgb_plot(timestamp, eol_timestamp)
        self._draw_lab_plot(timestamp, eol_timestamp)

    def _plot_bounds(self, canvas: tk.Canvas) -> tuple[int, int, int, int, int, int]:
        width = max(canvas.winfo_width(), int(canvas.cget("width") or 520))
        height = max(canvas.winfo_height(), int(canvas.cget("height") or 220))
        left, top, right, bottom = 52, 28, width - 20, height - 34
        return width, height, left, top, right, bottom

    @staticmethod
    def _x_position(value: pd.Timestamp, values: pd.Series, left: int, right: int) -> float:
        min_x = values.min().value
        max_x = values.max().value
        if min_x == max_x:
            return (left + right) / 2
        return left + ((value.value - min_x) / (max_x - min_x)) * (right - left)

    @staticmethod
    def _y_position(value: float, values: pd.Series, top: int, bottom: int) -> float:
        min_y = float(values.min())
        max_y = float(values.max())
        if min_y == max_y:
            return (top + bottom) / 2
        return bottom - ((float(value) - min_y) / (max_y - min_y)) * (bottom - top)

    def _draw_axes(self, canvas: tk.Canvas, title: str, left: int, top: int, right: int, bottom: int) -> None:
        canvas.create_text(left, 12, text=title, anchor=tk.W, fill="#111827", font=("Arial", 10, "bold"))
        canvas.create_line(left, top, left, bottom, fill="#9ca3af")
        canvas.create_line(left, bottom, right, bottom, fill="#9ca3af")

    def _draw_series(
        self,
        canvas: tk.Canvas,
        x_values: pd.Series,
        y_values: pd.Series,
        color: str,
        left: int,
        top: int,
        right: int,
        bottom: int,
        y_ref: pd.Series | None = None,
    ) -> None:
        if len(x_values) < 2:
            return
        x_source = x_values
        y_source = y_ref if y_ref is not None else y_values
        if len(x_values) > 260:
            step = max(1, len(x_values) // 260)
            x_values = x_values.iloc[::step]
            y_values = y_values.iloc[::step]
        points: list[float] = []
        for x_val, y_val in zip(x_values, y_values):
            points.extend(
                [
                    self._x_position(pd.Timestamp(x_val), x_source, left, right),
                    self._y_position(float(y_val), y_source, top, bottom),
                ]
            )
        canvas.create_line(*points, fill=color, width=2)

    def _draw_current_marker(self, canvas: tk.Canvas, timestamp: pd.Timestamp, x_values: pd.Series, top: int, bottom: int, left: int, right: int) -> None:
        x = self._x_position(timestamp, x_values, left, right)
        canvas.create_line(x, top, x, bottom, fill="#111827", dash=(4, 3), width=1)

    def _draw_eol_marker(self, canvas: tk.Canvas, timestamp: pd.Timestamp, x_values: pd.Series, top: int, bottom: int, left: int, right: int) -> None:
        x = self._x_position(timestamp, x_values, left, right)
        canvas.create_line(x, top, x, bottom, fill="#dc2626", dash=(8, 4), width=2)
        canvas.create_text(x + 4, top + 4, text="EOL", anchor=tk.NW, fill="#dc2626", font=("Arial", 8, "bold"))

    def _draw_cut_ranges(self, canvas: tk.Canvas, x_values: pd.Series, top: int, bottom: int, left: int, right: int, normalize: bool = False) -> None:
        for row in self._current_fruit_exclusions():
            try:
                start = pd.Timestamp(row["start_timestamp"])
                end = pd.Timestamp(row["end_timestamp"])
            except Exception:
                continue
            if normalize:
                start = start.normalize()
                end = end.normalize()
            start_x = self._x_position(min(start, end), x_values, left, right)
            end_x = self._x_position(max(start, end), x_values, left, right)
            if abs(end_x - start_x) < 4:
                end_x = start_x + 4
            canvas.create_rectangle(
                start_x,
                top,
                end_x,
                bottom,
                fill="#fecaca",
                outline="#ef4444",
                stipple="gray25",
            )
            canvas.create_text(start_x + 3, bottom - 4, text="CUT", anchor=tk.SW, fill="#991b1b", font=("Arial", 8, "bold"))

    def _draw_active_cut_selection(
        self,
        canvas: tk.Canvas,
        x_values: pd.Series,
        top: int,
        bottom: int,
        left: int,
        right: int,
        normalize: bool = False,
    ) -> None:
        start, end = self._active_cut_timestamps()
        if start is None and end is None:
            return
        markers = [("START", start), ("END", end)]
        marker_positions: list[float] = []
        for label, value in markers:
            if value is None:
                continue
            marker_time = value.normalize() if normalize else value
            x = self._x_position(marker_time, x_values, left, right)
            marker_positions.append(x)
            canvas.create_line(x, top, x, bottom, fill="#7c2d12", dash=(3, 2), width=2)
            canvas.create_text(x + 4, top + 18, text=label, anchor=tk.NW, fill="#7c2d12", font=("Arial", 8, "bold"))

        if len(marker_positions) == 2:
            start_x, end_x = sorted(marker_positions)
            if abs(end_x - start_x) < 4:
                end_x = start_x + 4
            canvas.create_rectangle(
                start_x,
                top,
                end_x,
                bottom,
                fill="#fed7aa",
                outline="#f97316",
                stipple="gray12",
            )
            canvas.create_text(start_x + 3, top + 4, text="ACTIVE CUT", anchor=tk.NW, fill="#7c2d12", font=("Arial", 8, "bold"))

    def _nearest_index(self, values: pd.Series, target: pd.Timestamp) -> int:
        deltas = (pd.to_datetime(values.reset_index(drop=True)) - target).abs()
        return int(deltas.argmin())

    def _draw_environment_plot(self, timestamp: pd.Timestamp, eol_timestamp: pd.Timestamp | None) -> None:
        canvas = self.env_canvas
        canvas.delete("all")
        _width, _height, left, top, right, bottom = self._plot_bounds(canvas)
        self._draw_axes(canvas, "Environment: temp and humidity", left, top, right, bottom)
        if self.env_df.empty:
            canvas.create_text((left + right) / 2, (top + bottom) / 2, text="No environment data", fill="#6b7280")
            return

        x_values = self.env_df["datetime"]
        self._draw_cut_ranges(canvas, x_values, top, bottom, left, right)
        self._draw_active_cut_selection(canvas, x_values, top, bottom, left, right)
        self._draw_series(canvas, x_values, self.env_df["temperature_c"], "#c2410c", left, top, right, bottom)
        self._draw_series(canvas, x_values, self.env_df["humidity_rh"], "#2563eb", left, top, right, bottom)
        self._draw_current_marker(canvas, timestamp, x_values, top, bottom, left, right)
        if eol_timestamp is not None:
            self._draw_eol_marker(canvas, eol_timestamp, x_values, top, bottom, left, right)
        canvas.create_text(left, bottom + 18, text="red: temperature C", anchor=tk.W, fill="#c2410c", font=("Arial", 8))
        canvas.create_text(right, bottom + 18, text="blue: humidity %RH", anchor=tk.E, fill="#2563eb", font=("Arial", 8))
        canvas.create_text(left, top - 10, text=f"{self.env_df['temperature_c'].min():.1f}-{self.env_df['temperature_c'].max():.1f} C", anchor=tk.W, fill="#c2410c", font=("Arial", 8))
        canvas.create_text(right, top - 10, text=f"{self.env_df['humidity_rh'].min():.1f}-{self.env_df['humidity_rh'].max():.1f}% RH", anchor=tk.E, fill="#2563eb", font=("Arial", 8))

    def _draw_firmness_plot(self, timestamp: pd.Timestamp, eol_timestamp: pd.Timestamp | None) -> None:
        canvas = self.firmness_canvas
        canvas.delete("all")
        _width, _height, left, top, right, bottom = self._plot_bounds(canvas)
        self._draw_axes(canvas, f"Firmness: {self.current_fruit}", left, top, right, bottom)

        col = firmness_column(self.current_fruit)
        if self.firmness_df.empty or col not in self.firmness_df.columns:
            canvas.create_text((left + right) / 2, (top + bottom) / 2, text="No firmness data", fill="#6b7280")
            return

        x_values = self.firmness_df["date"]
        y_values = self.firmness_df[col]
        y_ref = pd.Series([0, 5, float(y_values.max())])
        self._draw_cut_ranges(canvas, x_values, top, bottom, left, right, normalize=True)
        self._draw_active_cut_selection(canvas, x_values, top, bottom, left, right, normalize=True)
        self._draw_series(canvas, x_values, y_values, "#15803d", left, top, right, bottom, y_ref=y_ref)
        for x_val, y_val in zip(x_values, y_values):
            x = self._x_position(pd.Timestamp(x_val), x_values, left, right)
            y = self._y_position(float(y_val), y_ref, top, bottom)
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#15803d", outline="")

        for ref_value, color, label in [(0, "#991b1b", "0"), (5, "#f59e0b", "5")]:
            y = self._y_position(ref_value, y_ref, top, bottom)
            canvas.create_line(left, y, right, y, fill=color, dash=(2, 3))
            canvas.create_text(right, y - 6, text=label, anchor=tk.E, fill=color, font=("Arial", 8))

        self._draw_current_marker(canvas, timestamp.normalize(), x_values, top, bottom, left, right)
        if eol_timestamp is not None:
            self._draw_eol_marker(canvas, eol_timestamp.normalize(), x_values, top, bottom, left, right)
            eol_idx = self._nearest_index(x_values, eol_timestamp.normalize())
            eol_y = self._y_position(float(y_values.iloc[eol_idx]), y_ref, top, bottom)
            eol_x = self._x_position(pd.Timestamp(x_values.iloc[eol_idx]), x_values, left, right)
            canvas.create_oval(eol_x - 5, eol_y - 5, eol_x + 5, eol_y + 5, fill="#dc2626", outline="")
        canvas.create_text(left, bottom + 18, text="green: measured daily firmness", anchor=tk.W, fill="#15803d", font=("Arial", 8))

    def _draw_numeric_trend_plot(
        self,
        canvas: tk.Canvas,
        title: str,
        df: pd.DataFrame,
        x_column: str,
        series_specs: list[tuple[str, str]],
        timestamp: pd.Timestamp,
        footer: str,
        eol_timestamp: pd.Timestamp | None,
    ) -> None:
        canvas.delete("all")
        _width, _height, left, top, right, bottom = self._plot_bounds(canvas)
        self._draw_axes(canvas, title, left, top, right, bottom)
        if df.empty or x_column not in df.columns:
            canvas.create_text((left + right) / 2, (top + bottom) / 2, text="No feature data", fill="#6b7280")
            return

        x_values = df[x_column]
        if x_values.empty:
            canvas.create_text((left + right) / 2, (top + bottom) / 2, text="No feature data", fill="#6b7280")
            return

        y_min = min(float(df[column].min()) for column, _color in series_specs)
        y_max = max(float(df[column].max()) for column, _color in series_specs)
        if y_min == y_max:
            y_min -= 1
            y_max += 1
        y_ref = pd.Series([y_min, y_max])

        legend_x = left
        self._draw_cut_ranges(canvas, x_values, top, bottom, left, right)
        self._draw_active_cut_selection(canvas, x_values, top, bottom, left, right)
        for column, color in series_specs:
            self._draw_series(canvas, x_values, df[column], color, left, top, right, bottom, y_ref=y_ref)
            canvas.create_text(legend_x, top - 10, text=column.replace("_", " "), anchor=tk.W, fill=color, font=("Arial", 8))
            legend_x += 110

        self._draw_current_marker(canvas, timestamp, x_values, top, bottom, left, right)
        if eol_timestamp is not None:
            self._draw_eol_marker(canvas, eol_timestamp, x_values, top, bottom, left, right)
            eol_idx = self._nearest_index(x_values, eol_timestamp)
            for column, color in series_specs:
                eol_y = self._y_position(float(df[column].iloc[eol_idx]), y_ref, top, bottom)
                eol_x = self._x_position(pd.Timestamp(x_values.iloc[eol_idx]), x_values, left, right)
                canvas.create_oval(eol_x - 4, eol_y - 4, eol_x + 4, eol_y + 4, fill=color, outline="#111827")
        canvas.create_text(left, bottom + 18, text=footer, anchor=tk.W, fill="#4b5563", font=("Arial", 8))

    def _draw_rgb_plot(self, timestamp: pd.Timestamp, eol_timestamp: pd.Timestamp | None) -> None:
        fruit_df = self.feature_df[self.feature_df["fruit_id"] == self.current_fruit] if not self.feature_df.empty else pd.DataFrame()
        self._draw_numeric_trend_plot(
            self.rgb_canvas,
            f"RGB: {self.current_fruit}",
            fruit_df,
            "timestamp",
            [("mean_r", "#dc2626"), ("mean_g", "#16a34a"), ("mean_b", "#2563eb")],
            timestamp,
            "red: mean_r | green: mean_g | blue: mean_b",
            eol_timestamp,
        )

    def _draw_lab_plot(self, timestamp: pd.Timestamp, eol_timestamp: pd.Timestamp | None) -> None:
        fruit_df = self.feature_df[self.feature_df["fruit_id"] == self.current_fruit] if not self.feature_df.empty else pd.DataFrame()
        self._draw_numeric_trend_plot(
            self.lab_canvas,
            f"LAB: {self.current_fruit}",
            fruit_df,
            "timestamp",
            [("mean_lab_l", "#0f766e"), ("mean_lab_a", "#7c3aed"), ("mean_lab_b", "#d97706")],
            timestamp,
            "teal: L* | purple: a* | amber: b*",
            eol_timestamp,
        )

    def _mark_current_as_eol(self) -> None:
        frame = self.current_frame
        context = firmness_context(self.firmness_df, self.current_fruit, frame.timestamp)
        self.annotations[self.current_fruit] = {
            "fruit_id": self.current_fruit,
            "eol_timestamp": frame.timestamp.isoformat(sep=" "),
            "eol_image_path": str(frame.image_path),
            "eol_basis": self.basis_var.get(),
            "eol_confidence": self.confidence_var.get(),
            "visual_note": self.note_var.get(),
            "firmness_latest": str(context["firmness_latest"]),
            "firmness_date": str(context["firmness_date"]),
            "first_firmness_le5_date": str(context["first_le5_date"]),
            "first_firmness_zero_date": str(context["first_zero_date"]),
            "label_status": "proposed",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.status_var.set(f"Marked {self.current_fruit} EOL at {frame.timestamp}. Save when ready.")
        self._update_view()

    def _save_annotations(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "fruit_id",
            "eol_timestamp",
            "eol_image_path",
            "eol_basis",
            "eol_confidence",
            "visual_note",
            "firmness_latest",
            "firmness_date",
            "first_firmness_le5_date",
            "first_firmness_zero_date",
            "label_status",
            "updated_at",
        ]
        rows = [self.annotations[key] for key in sorted(self.annotations)]
        with self.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self._write_exclusions()
        self.status_var.set(f"Saved {len(rows)} EOL annotations and {len(self.exclusions)} cut ranges.")
        messagebox.showinfo("Saved", f"Saved {len(rows)} EOL annotations and {len(self.exclusions)} cut ranges.")

    def _write_exclusions(self) -> None:
        self.exclusions_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "fruit_id",
            "start_frame_idx",
            "end_frame_idx",
            "start_timestamp",
            "end_timestamp",
            "start_image_path",
            "end_image_path",
            "frame_count",
            "reason",
            "note",
            "label_status",
            "updated_at",
        ]
        rows = sorted(
            self.exclusions,
            key=lambda row: (
                row.get("fruit_id", ""),
                int(row.get("start_frame_idx", "0") or 0),
                int(row.get("end_frame_idx", "0") or 0),
            ),
        )
        with self.exclusions_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _save_exclusions(self) -> None:
        self._write_exclusions()
        self.status_var.set(f"Saved {len(self.exclusions)} cut ranges to {self.exclusions_csv}")
        messagebox.showinfo("Saved", f"Saved {len(self.exclusions)} cut ranges.")


def main() -> None:
    root = tk.Tk()
    try:
        app = EolLabelingApp(root)
    except Exception as exc:
        messagebox.showerror("EOL labeling startup failed", str(exc))
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
