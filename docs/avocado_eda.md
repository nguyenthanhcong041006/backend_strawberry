# Avocado EDA Stage

This stage contains exploratory analysis tools that help inspect the segmented avocado dataset after preprocessing. These outputs are useful for QA, threshold validation, paper figures, and later downstream feature optimization, but they are not the MVP baseline model-ready dataset.

## Code Location

```text
src/avocado/stage2_eda/
  extract_features.py
  generate_eda_graphs.py
  visual_inspection_ui.py
  modules/
    feature_extractor.py
    temporal_features.py
```

## Outputs

EDA artifacts are written under:

```text
data/02_processed/avocado/eda/
  features/avocado_features.csv
  graphs/
  flags/bad_segmentation_flags.csv
```

## Execution

Run these after segmentation has produced `data/02_processed/avocado/segmented/F01` through `F06`.

```bash
python src/avocado/stage2_eda/run_avocado_eda.py
python src/avocado/stage2_eda/extract_features.py
python src/avocado/stage2_eda/generate_eda_graphs.py
python src/avocado/stage2_eda/visual_inspection_ui.py
```

The combined notebook entrypoint is `notebooks/avocado_rul_dataset_inspection.ipynb`, which runs the EDA step first and then the RUL inspection step.

## MVP Boundary

The immediate MVP data-readiness path should not depend on these derived EDA features. Baseline model data should first be built from:

- segmented crop image paths;
- fruit IDs and timestamps;
- environmental temperature/humidity;
- daily firmness forward-filled without future leakage;
- fruit-specific EOL and RUL labels;
- hourly sequence windows.

EDA features such as mask area, dark coverage, size change, and texture can be used later for feature-augmented experiments once the baseline image/numeric/firmness models are established.
