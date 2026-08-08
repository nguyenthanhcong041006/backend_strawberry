# Avocado Pipeline & Architecture Documentation

This document summarizes the recent architectural changes made to accommodate the Avocado RUL prediction pipeline alongside the existing Strawberry pipeline, as well as the implementation of the new multimodal Avocado models.

## 1. Top-Level Fruit Separation in `src/`

To prevent cross-contamination of logic and maintain a clean codebase, the `src/` directory has been reorganized at the top level by fruit type.

### New Directory Structure
```text
src/
├── api/             # Shared API endpoints
├── config_app/      # Shared configuration files
├── services/        # Shared prediction services (e.g. predictor.py)
├── shared/          # Shared utilities/classes
├── strawberry/      # 🍓 ALL STRAWBERRY-SPECIFIC CODE
│   ├── stage1_...
│   ├── stage2_...
│   ├── stage3_preprocessing/
│   │   └── main_preprocessing.py  # Original strawberry preprocessing router
│   ├── stage4_training/
│   └── stage5_evaluation/
└── avocado/         # 🥑 ALL AVOCADO-SPECIFIC CODE
    ├── stage3_preprocessing/
    │   ├── data_ingestion.py
    │   ├── segmentation.py
    │   └── ...
    └── stage4_training/
        ├── train_avocado.py       # Avocado-specific LOOCV training loop
        └── models/                # Avocado model architectures
```

### Running Strawberry Preprocessing
Since the structures are isolated, you can run the Strawberry pipeline exactly as before by navigating to its specific directory:
```bash
cd src/strawberry/stage3_preprocessing
python main_preprocessing.py
```

---

## Avocado EDA Tools

Post-segmentation exploratory analysis now lives under `src/avocado/stage2_eda/`.

```bash
python src/avocado/stage2_eda/extract_features.py
python src/avocado/stage2_eda/generate_eda_graphs.py
python src/avocado/stage2_eda/visual_inspection_ui.py
```

These tools write to `data/02_processed/avocado/eda/` and are intended for QA, dataset understanding, and later feature-augmented experiments. They are separate from the MVP model-ready data construction.

---

## 2. Avocado Model Implementations

Based on Section 9 of the Avocado RUL specification, several deep learning architectures have been developed to evaluate performance across different modalities (image and numeric).

These models are located in `src/avocado/stage4_training/models/`.

### A. Numeric Baselines (`numeric_baselines.py`)
Uses only environmental (temperature, humidity) and firmness data.
- **`NumericBaselineMLP`**: Flattens the 24-hour sequence and uses a Multi-Layer Perceptron.
- **`NumericBaselineGRU`**: Uses a Gated Recurrent Unit (GRU) to temporally encode the sequence.

### B. Image Baselines (`image_baselines.py`)
Uses only visual data.
- **`ImageBaselineViT`**: Uses a pretrained `ViT_B_16` (frozen weights) to extract spatial features from image crops, then performs temporal averaging across the sequence before regression.

### C. ViT + Mamba (`vit_mamba.py`)
The primary multimodal sequence model.
- **`ViTMambaRULModel`**: Uses ViT for spatial extraction per frame, and passes the resulting sequence into a **Mamba** state-space model for temporal sequence learning. 
- *Note:* If the `mamba_ssm` library is not installed, the code safely falls back to a PyTorch `GRU` to prevent crashes.

### D. Fusion Models (`fusion_models.py`)
Explores different strategies for combining numeric and visual data.
- **`EarlyFusionModel`**: Concatenates ViT image features with numeric features at *each timestamp* before passing them to a temporal encoder.
- **`LateFusionModel`**: Processes the image sequence and numeric sequence through completely separate temporal encoders, then concatenates the final latent vectors before the regression head.
- **`MBTFusionModel`**: A conceptual Multimodal Bottleneck Transformer implementation using cross-attention bottlenecks to exchange information between the modalities.

---

## 3. How to Run & Test the Models

Currently, the data preprocessing step is incomplete (data only exists in `data/01_raw/avocado`). However, all model files contain built-in tests with dummy tensors to verify their forward passes and tensor dimension math.

You can run these tests directly from your terminal:

**Test Numeric Baselines:**
```bash
python src/avocado/stage4_training/models/numeric_baselines.py
```

**Test Image Baselines:**
```bash
python src/avocado/stage4_training/models/image_baselines.py
```

**Test ViT + Mamba:**
```bash
python src/avocado/stage4_training/models/vit_mamba.py
```

**Test Fusion Models:**
```bash
python src/avocado/stage4_training/models/fusion_models.py
```

### Installing Mamba
To unleash the full potential of the temporal sequence models, ensure you have the `mamba_ssm` library installed in your environment:
```bash
pip install causal-conv1d>=1.2.0
pip install mamba-ssm
```
