# Avocado MVP Status

## Current Progress

- Raw avocado time-lapse frames are available under `data/01_raw/avocado/output`.
- Environmental readings are available in `data/01_raw/avocado/th10s_readings.csv`.
- Daily firmness readings are available in `data/01_raw/avocado/hardness/hardness.csv`.
- Segmentation has been completed with good fruit-level outputs under `data/02_processed/avocado/segmented/F01` through `F06`.
- EDA utilities were moved to `src/avocado/stage2_eda/`, with outputs under `data/02_processed/avocado/eda/`.
- Model architecture prototypes exist under `src/avocado/stage4_training/models/`, but model training is not part of this environment and the current trainer still uses dummy data.

## MVP Definition

The immediate MVP is to create model-ready avocado data, not to train final models here.

The first two goals are:

1. Build EOL annotations and a labeled hourly table.
2. Build leakage-safe sequence datasets for model training on a stronger device.

## EOL Label Tool

Use:

```bash
python src/avocado/stage3_preprocessing/eol_labeling_ui.py
```

Inputs:

- `data/02_processed/avocado/segmented/F01-F06`
- `data/01_raw/avocado/hardness/hardness.csv`

Output:

- `data/02_processed/avocado/labels/eol_annotations.csv`

The default endpoint is `visible_mold_onset`, meaning first externally visible mold/spoilage onset in segmented fruit-level images. Firmness is used as supporting validation.

## Next Steps

1. Use the EOL labeling UI to propose one EOL timestamp for each fruit.
2. Review and approve `eol_annotations.csv`.
3. Run `python src/avocado/stage2_eda/run_avocado_eda.py` and `python src/avocado/stage3_preprocessing/inspect_rul_dataset.py`, or open the companion notebook, to build the EDA and frame-level RUL inspection tables.
4. Build `crop_index.csv` from segmented images.
5. Build `hourly_table.csv` by aligning image crops, temperature, humidity, latest-known firmness, EOL, and RUL.
6. Decide whether model-ready data excludes post-EOL frames or keeps a separate post-EOL buffer.
7. Build sequence windows after fruit-level fold assignment.
8. Export model-ready files and a model-development handoff plan.
