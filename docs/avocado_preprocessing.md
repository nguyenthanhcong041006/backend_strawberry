# Avocado Preprocessing Pipeline

This document details the architecture, workflow, and execution details for the Avocado Preprocessing Pipeline, now orchestrated by `scripts/phase3_1_segmentation.py`.

## Overview

The purpose of this pipeline is to process raw time-lapse images of avocado trays (containing 6 avocados each, recorded over 11 days), isolate each individual avocado, perform deep-learning-based background removal (segmentation), and orient the fruit consistently. Finally, it outputs perfectly aligned transparent images into dedicated subfolders (`F01` to `F06`) and generates a mapping CSV.

## Modular Architecture

To ensure code maintainability and execution flexibility, the pipeline is separated into distinct, decoupled modules under `src/avocado/stage3_preprocessing/modules/`:

1. **`roi_extractor.py`**: Handles reading frames, applying `GRAYWORLD` & `CLAHE` color enhancements, and cropping out the 6 regions of interest.
2. **`segmenter.py`**: Handles AI inference using `U2Net-Silueta` via the `rembg` library, backed by ONNX Runtime with retry logic.
3. **`reorienter.py`**: Extracts the angle of the segmented avocado mask and rotates the crop so it sits perfectly vertical (with the narrower peak at the top).
4. **`mask_applier.py`**: Merges the rotated mask and crop to produce a final RGBA image with a transparent background.
5. **`reporter.py`**: Generates `mask_fruits.csv` mapping every processed frame to its final outputs.

## Producer-Consumer Optimization

When run end-to-end, the orchestrator utilizes a **Multi-threaded Producer-Consumer Architecture** to achieve high throughput without freezing the system:
- **Producer (CPU)**: A background thread handles the I/O-heavy task of extracting and enhancing crops, buffering them into a thread-safe queue (size 2).
- **Consumer (GPU)**: The main thread pulls crops, runs the ONNX neural network, reorients the fruit, saves the outputs, and yields 10ms to the Windows UI manager to prevent desktop lag.
- **Thread Limits**: OpenCV is restricted to 4 threads (`cv2.setNumThreads(4)`), and ONNX fallback CPU operations are restricted to 2 (`OMP_NUM_THREADS=2`).

## Output Directory Structure

The pipeline generates several intermediate and final directories under `data/02_processed/avocado/`:
- `crops/`: Contains the un-rotated, colorful raw crops directly from the original frames.
- `raw_masks/`: Contains the un-rotated, raw black-and-white masks from U2Net.
- `masks/`: Contains the final, perfectly rotated and centered masks.
- `segmented/`: Contains the final output! Inside are subdirectories `F01` through `F06`, which contain the transparent, vertically aligned avocado images.

Exploratory feature extraction, graphing, and visual inspection tools are documented separately in `docs/avocado_eda.md` and live under `src/avocado/stage2_eda/`. Their outputs are stored under `data/02_processed/avocado/eda/` so they do not get confused with the MVP model-ready dataset.

## Execution

You can run the script via the command line. By default, it runs the **entire pipeline** and saves all intermediate and final files:

```bash
python scripts/phase3_1_segmentation.py
```

### Individual Execution Modes (Optional)
If you already ran the full pipeline once and want to tweak or rerun only a specific phase, you can use these flags (which read the saved intermediate files off your disk):

- `--extract-only`: Only reads raw frames and generates raw crops.
- `--segment-only`: Only runs the AI model on saved raw crops to generate `raw_masks`.
- `--reorient-only`: Only calculates angles and rotates saved raw crops and raw masks.
- `--apply-only`: Only reads reoriented crops and masks, merging them into transparent PNGs in `F01-F06`.
- `--report-only`: Only generates the `mask_fruits.csv`.

### Other Arguments
- `--input-dir`: Directory containing the raw frames (Default: `data/01_raw/avocado/output`).
- `--output-dir`: Base directory for outputs (Default: `data/02_processed/avocado`).
- `--min-area`: Minimum bounding box area required to pass the U2Net retry gate (Default: `100000`).
- `--disable-grayworld` / `--disable-clahe`: Disables image enhancements.

## Hardware & Environment Compatibility

Because `rembg` relies on ONNX Runtime (`onnxruntime-gpu`) for neural network inference, strict alignment between your NVIDIA drivers, CUDA Toolkit, and cuDNN libraries is required.

> [!WARNING]
> If you experience `Error 126` or immediate fallback to the CPU, it is almost certainly a mismatch between your cuDNN version and your `onnxruntime-gpu` version.

### Known Working Configurations

**Configuration 1: CUDA 12.x + cuDNN 8.x (Current Environment)**
- **CUDA Toolkit:** 12.1
- **cuDNN:** 8.9.x (e.g., `nvidia-cudnn-cu12==8.9.7.29`)
- **Required ONNX Version:** `onnxruntime-gpu==1.17.1` or `1.18.1`
- *Note:* The manual DLL loader in `scripts/phase3_1_segmentation.py` is currently configured for this environment, explicitly seeking `cudnn64_8.dll`.

**Configuration 2: CUDA 13.x (or 12.x) + cuDNN 9.x**
- **CUDA Toolkit:** 12.x or 13.x
- **cuDNN:** 9.x
- **Required ONNX Version:** `onnxruntime-gpu>=1.19.0`
- *Note:* If you move this project to a machine with cuDNN 9, you **must** update the manual DLL loader at the top of `scripts/phase3_1_segmentation.py` to target `cudnn64_9.dll` instead of `cudnn64_8.dll`, and you should upgrade ONNX Runtime to `>=1.19.0`.
