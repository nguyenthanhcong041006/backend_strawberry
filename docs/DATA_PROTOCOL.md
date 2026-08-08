# Data Protocol

This protocol defines how raw and processed data must be recorded so avocado and strawberry experiments remain traceable, comparable, and leakage-safe.

## Required Concepts

| Field | Meaning | Required |
| --- | --- | --- |
| `experiment_id` | Unique ID for one recording experiment or batch | Yes |
| `fruit_type` | `avocado` or `strawberry` | Yes |
| `fruit_id` | Stable physical fruit ID based on fixed ROI position | Yes for fruit-level data |
| `roi_id` | Static position in the 3x2 box | Yes |
| `timestamp` | Capture timestamp for the frame/sample | Yes |
| `frame_id` | Sequential frame or sample number | Yes for images |
| `source_path` | Path to the raw source file | Yes |
| `image_path` | Path to processed/model-ready image | Yes after preprocessing |
| `temperature_c` | Room/box temperature aligned from a real sensor timestamp | Required for strawberry MVP; missing only when no sensor reading is matched |
| `humidity_pct` | Room/box humidity aligned from a real sensor timestamp | Required for strawberry MVP; missing only when no sensor reading is matched |
| `firmness_avg` | Daily average firmness for that fruit | Avocado only |
| `firmness_n` | Number of firmness points measured | Avocado only |
| `valid_frame` | Whether frame is allowed in model-ready manifests | Yes after QC |
| `exclude_reason` | Reason if `valid_frame = false` | Required when excluded |

Use empty values or `NA` for unavailable optional fields. Do not fabricate numeric values.

## Fruit ID Rule

The 3x2 experimental box defines six stable ROI positions. Each physical fruit must remain in the same ROI throughout the experiment.

Small drift or rotation after firmness measurement is acceptable. Swapping fruits between ROI positions is not acceptable unless the correction is documented manually in the metadata.

Recommended ID format:

```text
F01 F02 F03
F04 F05 F06
```

The exact top/bottom and left/right mapping must be recorded in the experiment metadata before recording starts.

## Raw Data Rules

- Raw videos, raw frames, and raw sensor logs are immutable.
- Never overwrite raw files during preprocessing.
- Store derived files under `data/02_processed/`.
- Store split/model-ready outputs under `data/03_split/`.
- Record capture failures instead of deleting evidence.

## Recommended Naming

Use timestamp-first names for easy verification and joining.

Raw frame:

```text
<experiment_id>_<fruit_type>_<YYYYMMDD>T<HHMMSS>_frame<frame_id>.jpg
```

Processed fruit image:

```text
<experiment_id>_<fruit_type>_<fruit_id>_<YYYYMMDD>T<HHMMSS>_frame<frame_id>.png
```

Example:

```text
EXP001_strawberry_F03_20260617T153000_frame000123.png
```

The current prototype scripts use names such as `frame-1_12-26-28.jpg`. That is acceptable for existing sample data, but new recorded data should move toward the standard above.

## Timestamp Mapping

Every processed image must trace back to one capture timestamp.

Environmental data is measured once for the box/room, not per fruit. Map temperature and humidity to frames by timestamp. Prefer exact timestamp joins when sensor and frame timestamps match. If the sensor frequency differs, use the nearest timestamp within a documented tolerance and record the mapping method in the report.

For the current strawberry workflow, temperature and humidity must come from a real sensor CSV or remain missing. Do not invent environment values for training, evaluation, notebooks, or backend predictions.

Hung must confirm the exact mapping behavior used by the current acquisition system before the mapping report is treated as final.

Required mapping report fields:

```text
image_path,timestamp,sensor_timestamp,temperature_c,humidity_pct,mapping_method,mapping_delta_seconds
```

Recommended strawberry sensor CSV columns:

```text
timestamp,temperature_c,humidity_pct
```

Accepted aliases are `temperature` for `temperature_c` and `humidity_rh` for `humidity_pct`.

## Firmness Mapping

Firmness is optional in the shared schema.

For avocado:

- Measure once per day per fruit.
- Take five firmness readings on different points.
- Store the average as `firmness_avg`.
- Store the number of readings as `firmness_n = 5`.
- Map the daily fruit-level average to all frames for that same `fruit_id` and date.

For strawberry:

- Leave firmness fields empty or `NA`.
- Set `firmness_available = false` where that field exists.

## Exclusion Rules

Do not delete raw data. Exclude invalid samples only from model-ready manifests.

A frame should be excluded when it is:

- black or blank;
- unreadable or corrupted;
- blocked by a hand, measurement device, or other non-fruit object;
- missing timestamp metadata;
- missing required environmental mapping;
- assigned to an uncertain fruit ID;
- associated with a failed or suspicious mask;
- duplicated without explanation.

Recommended exclusion log:

```text
experiment_id,fruit_id,timestamp,raw_path,processed_path,exclude_reason,detected_by,reviewed_by,notes
```

## Integrity Checklist

Before preprocessing is accepted:

- Each experiment has metadata describing fruit type, ROI layout, camera setup, and recording period.
- Each fruit has a stable `fruit_id`.
- Each raw file has a timestamp.
- Sensor logs cover the recording interval.
- Firmness records exist for avocado or are explicitly unavailable for strawberry.
- Raw and processed counts are reported.
- Exclusions are logged with reason codes.
- No processed file is untraceable to raw data.
