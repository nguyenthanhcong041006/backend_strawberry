# =====================================================================
# STEP 1: COMPREHENSIVE CUDA & CUDNN DIRECT LOADER (REMOVE THIS FOR RTX 4050, ADD FOR RTX 3060TI)
# =====================================================================
import os
import sys
import site
import ctypes

# 1. Path to your CUDA Toolkit Binaries
cuda_bin_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"

# 2. Add DLL search paths for Windows (Python 3.8+)
if os.path.exists(cuda_bin_dir):
    os.add_dll_directory(cuda_bin_dir)

for path in site.getsitepackages():
    nvidia_dir = os.path.join(path, "nvidia")
    if os.path.exists(nvidia_dir):
        for root, dirs, _ in os.walk(nvidia_dir):
            if "bin" in dirs:
                os.add_dll_directory(os.path.join(root, "bin"))

# 3. FORCE Python's memory manager to pre-load critical dependencies.
# This prevents Error 126 when onnxruntime attempts to resolve its symbols.
dlls_to_load = [
    # CUDA Core
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    # cuDNN Core
    "cudnn64_8.dll",
    "cudnn_ops_infer64_8.dll",
    "cudnn_cnn_infer64_8.dll",
    "cudnn_adv_infer64_8.dll"
]

for dll in dlls_to_load:
    try:
        # First try loading from system / local site-packages paths
        ctypes.CDLL(dll)
    except Exception:
        # If standard load fails, try direct path fallback
        direct_cuda_path = os.path.join(cuda_bin_dir, dll)
        if os.path.exists(direct_cuda_path):
            try:
                ctypes.CDLL(direct_cuda_path)
            except Exception as e:
                print(f"[Warning] Failed to direct-load {dll}: {e}")
# =====================================================================

import argparse
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Thread Constraints
os.environ["OMP_NUM_THREADS"] = "2"
import cv2
cv2.setNumThreads(4)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.avocado.stage3_preprocessing.modules.roi_extractor import ImageEnhancer, TiltedAvocadoRoiMap
from src.avocado.stage3_preprocessing.modules.segmenter import SmoothU2NetMasker
from src.avocado.stage3_preprocessing.modules.reorienter import reorient_fruit
from src.avocado.stage3_preprocessing.modules.mask_applier import apply_mask_to_crop
from src.avocado.stage3_preprocessing.modules.reporter import generate_report


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "01_raw" / "avocado" / "output"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "02_processed" / "avocado"


@dataclass
class SegmentationConfig:
    input_dir: Path
    output_dir: Path
    use_grayworld: bool
    use_clahe: bool
    clahe_clip_limit: float
    clahe_grid_size: int
    min_area: int
    mode: str

    @property
    def crops_dir(self): return self.output_dir / "crops"
    @property
    def raw_masks_dir(self): return self.output_dir / "raw_masks"
    @property
    def reoriented_masks_dir(self): return self.output_dir / "masks"
    @property
    def segmented_dir(self): return self.output_dir / "segmented"
    @property
    def report_path(self): return self.output_dir / "mask_fruits.csv"

    def make_dirs(self):
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.raw_masks_dir.mkdir(parents=True, exist_ok=True)
        self.reoriented_masks_dir.mkdir(parents=True, exist_ok=True)
        self.segmented_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, 7):
            (self.segmented_dir / f"F{i:02d}").mkdir(exist_ok=True)


class ModularPipeline:
    def __init__(self, config: SegmentationConfig):
        self.config = config
        self.config.make_dirs()
        self.roi_map = TiltedAvocadoRoiMap()
        self.u2net = SmoothU2NetMasker() if config.mode in ["all", "segment"] else None
        self.enhancer = ImageEnhancer(
            use_grayworld=config.use_grayworld,
            use_clahe=config.use_clahe,
            clahe_clip_limit=config.clahe_clip_limit,
            clahe_grid_size=config.clahe_grid_size,
        )

    def run(self):
        if self.config.mode == "all":
            self._run_all()
        elif self.config.mode == "extract":
            self._run_extract()
        elif self.config.mode == "segment":
            self._run_segment()
        elif self.config.mode == "reorient":
            self._run_reorient()
        elif self.config.mode == "apply":
            self._run_apply()
        elif self.config.mode == "report":
            self._run_report()

    # =========================================================================
    # END-TO-END PIPELINE (Producer-Consumer)
    # =========================================================================
    def _producer_worker(self, input_paths: list[Path], q: queue.Queue):
        for frame_path in input_paths:
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                print(f"\n[WARNING] Failed to read {frame_path}")
                continue

            height, width = image_bgr.shape[:2]
            enhanced_bgr = self.enhancer.enhance(image_bgr)
            rois = self.roi_map.build_bounding_boxes(width, height)

            crops = []
            for fruit_id, (x, y, w, h) in rois:
                pad = 40
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(width, x + w + pad), min(height, y + h + pad)
                crop_bgr = enhanced_bgr[y1:y2, x1:x2]
                
                # Save crop intermediate
                crop_out = self.config.crops_dir / f"{frame_path.stem}_fruit_{fruit_id}.png"
                cv2.imwrite(str(crop_out), crop_bgr)
                
                crops.append((fruit_id, crop_bgr, str(crop_out)))
                
            q.put((frame_path, crops))
        q.put(None)

    def _run_all(self):
        input_paths = sorted(list(self.config.input_dir.glob("*.jpg")) + list(self.config.input_dir.glob("*.png")))
        if not input_paths:
            print(f"No images found in {self.config.input_dir}")
            return

        q = queue.Queue(maxsize=2)
        producer_thread = threading.Thread(target=self._producer_worker, args=(input_paths, q))
        producer_thread.start()

        results = []
        total_frames = len(input_paths)
        processed_count = 0

        while True:
            item = q.get()
            if item is None:
                break
                
            frame_path, crops = item
            processed_count += 1
            sys.stdout.write(f"\rProgress: {(processed_count/total_frames)*100:.1f}% ({processed_count}/{total_frames}) | Processing {frame_path.name}{' '*10}")
            sys.stdout.flush()
            
            for fruit_id, crop_bgr, crop_out_path in crops:
                success = False
                final_mask = None
                
                for attempt, model_name in enumerate(["silueta", "silueta", "u2net"]):
                    try:
                        alpha = self.u2net.predict_alpha(crop_bgr, model_name=model_name)
                        binary_mask = self.u2net.border_normalizer(alpha)
                        
                        mx, my, mw, mh = cv2.boundingRect(binary_mask)
                        if (mw * mh) >= self.config.min_area:
                            final_mask = binary_mask
                            success = True
                            if attempt > 0:
                                print(f"\n[WARNING] Fruit {fruit_id}: Success on retry {attempt+1} ({model_name})")
                            break
                    except Exception as e:
                        print(f"\n[WARNING] Fruit {fruit_id}: Attempt {attempt+1} ({model_name}) error: {e}")
                
                mask_path = ""
                segmented_path = ""
                
                if success and final_mask is not None:
                    # Save raw mask intermediate
                    raw_mask_out = self.config.raw_masks_dir / f"{frame_path.stem}_fruit_{fruit_id}.png"
                    cv2.imwrite(str(raw_mask_out), final_mask)
                    
                    reoriented_mask, reoriented_crop = reorient_fruit(final_mask, crop_bgr)
                    
                    mask_out = self.config.reoriented_masks_dir / f"{frame_path.stem}_fruit_{fruit_id}_mask.png"
                    cv2.imwrite(str(mask_out), reoriented_mask)
                    mask_path = str(mask_out)
                    
                    segmented_rgba = apply_mask_to_crop(reoriented_mask, reoriented_crop)
                    seg_out = self.config.segmented_dir / f"F{fruit_id:02d}" / f"{frame_path.stem}_fruit_{fruit_id}.png"
                    cv2.imwrite(str(seg_out), segmented_rgba)
                    segmented_path = str(seg_out)
                else:
                    print(f"\n[WARNING] Fruit {fruit_id}: Failed all retries on {frame_path.name}")
                
                results.append({
                    "frame_path": str(frame_path),
                    "fruit_id": fruit_id,
                    "mask_path": mask_path,
                    "segmented_path": segmented_path
                })
            
            time.sleep(0.01)
        
        producer_thread.join()
        generate_report(results, self.config.report_path)
        print(f"\nPipeline complete. Report saved to {self.config.report_path}")

    # =========================================================================
    # INDIVIDUAL PHASES
    # =========================================================================
    def _run_extract(self):
        print("Running Extraction Only...")
        input_paths = sorted(list(self.config.input_dir.glob("*.jpg")) + list(self.config.input_dir.glob("*.png")))
        for i, frame_path in enumerate(input_paths):
            sys.stdout.write(f"\rExtracting {i+1}/{len(input_paths)}: {frame_path.name}{' '*10}")
            sys.stdout.flush()
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                continue
            enhanced_bgr = self.enhancer.enhance(image_bgr)
            rois = self.roi_map.build_bounding_boxes(image_bgr.shape[1], image_bgr.shape[0])
            for fruit_id, (x, y, w, h) in rois:
                pad = 40
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(image_bgr.shape[1], x + w + pad), min(image_bgr.shape[0], y + h + pad)
                crop_bgr = enhanced_bgr[y1:y2, x1:x2]
                cv2.imwrite(str(self.config.crops_dir / f"{frame_path.stem}_fruit_{fruit_id}.png"), crop_bgr)
        print("\nExtraction complete.")

    def _run_segment(self):
        print("Running Segmentation Only...")
        crop_paths = sorted(list(self.config.crops_dir.glob("*.png")))
        for i, crop_path in enumerate(crop_paths):
            sys.stdout.write(f"\rSegmenting {i+1}/{len(crop_paths)}: {crop_path.name}{' '*10}")
            sys.stdout.flush()
            crop_bgr = cv2.imread(str(crop_path))
            success = False
            for model_name in ["silueta", "silueta", "u2net"]:
                try:
                    alpha = self.u2net.predict_alpha(crop_bgr, model_name=model_name)
                    binary_mask = self.u2net.border_normalizer(alpha)
                    mx, my, mw, mh = cv2.boundingRect(binary_mask)
                    if (mw * mh) >= self.config.min_area:
                        cv2.imwrite(str(self.config.raw_masks_dir / crop_path.name), binary_mask)
                        success = True
                        break
                except:
                    pass
            if not success:
                print(f"\n[WARNING] Segmentation failed for {crop_path.name}")
        print("\nSegmentation complete.")

    def _run_reorient(self):
        print("Running Reorientation Only...")
        mask_paths = sorted(list(self.config.raw_masks_dir.glob("*.png")))
        for i, mask_path in enumerate(mask_paths):
            sys.stdout.write(f"\rReorienting {i+1}/{len(mask_paths)}: {mask_path.name}{' '*10}")
            sys.stdout.flush()
            raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            crop_bgr = cv2.imread(str(self.config.crops_dir / mask_path.name))
            if raw_mask is None or crop_bgr is None:
                continue
            
            reoriented_mask, _ = reorient_fruit(raw_mask, crop_bgr)
            out_name = mask_path.name.replace(".png", "_mask.png")
            cv2.imwrite(str(self.config.reoriented_masks_dir / out_name), reoriented_mask)
        print("\nReorientation complete.")

    def _run_apply(self):
        print("Running Apply Masks Only...")
        # To apply correctly we need both the raw mask AND the raw crop to pass through reorient_fruit to get the reoriented crop
        # OR we could save reoriented_crops in _run_reorient, but passing through reorient_fruit again is fine
        mask_paths = sorted(list(self.config.raw_masks_dir.glob("*.png")))
        for i, mask_path in enumerate(mask_paths):
            sys.stdout.write(f"\rApplying {i+1}/{len(mask_paths)}: {mask_path.name}{' '*10}")
            sys.stdout.flush()
            
            raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            crop_bgr = cv2.imread(str(self.config.crops_dir / mask_path.name))
            if raw_mask is None or crop_bgr is None:
                continue
                
            reoriented_mask, reoriented_crop = reorient_fruit(raw_mask, crop_bgr)
            segmented_rgba = apply_mask_to_crop(reoriented_mask, reoriented_crop)
            
            # Extract fruit_id from filename e.g. "webcam_..._fruit_1.png"
            fruit_id_str = mask_path.stem.split("_")[-1]
            try:
                fruit_id = int(fruit_id_str)
            except:
                fruit_id = 1
                
            out_path = self.config.segmented_dir / f"F{fruit_id:02d}" / mask_path.name
            cv2.imwrite(str(out_path), segmented_rgba)
        print("\nApply complete.")

    def _run_report(self):
        print("Running Report Generation Only...")
        # Scans the segmented_dir and masks_dir to build the CSV
        results = []
        frames_dict = {}
        
        # Infer original frame paths from the crop names
        for crop_path in self.config.crops_dir.glob("*.png"):
            parts = crop_path.stem.split("_fruit_")
            if len(parts) == 2:
                frame_stem = parts[0]
                fruit_id = parts[1]
                # Assuming original was .jpg or .png
                frame_path = self.config.input_dir / f"{frame_stem}.jpg"
                if not frame_path.exists():
                    frame_path = self.config.input_dir / f"{frame_stem}.png"
                
                mask_out = self.config.reoriented_masks_dir / f"{frame_stem}_fruit_{fruit_id}_mask.png"
                seg_out = self.config.segmented_dir / f"F{int(fruit_id):02d}" / f"{frame_stem}_fruit_{fruit_id}.png"
                
                results.append({
                    "frame_path": str(frame_path),
                    "fruit_id": fruit_id,
                    "mask_path": str(mask_out) if mask_out.exists() else "",
                    "segmented_path": str(seg_out) if seg_out.exists() else ""
                })
        
        generate_report(results, self.config.report_path)
        print(f"Report saved to {self.config.report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--disable-grayworld", action="store_true")
    parser.add_argument("--disable-clahe", action="store_true")
    parser.add_argument("--clahe-clip-limit", type=float, default=1.2)
    parser.add_argument("--clahe-grid-size", type=int, default=7)
    parser.add_argument("--min-area", type=int, default=100000)
    
    # Modes
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--segment-only", action="store_true")
    parser.add_argument("--reorient-only", action="store_true")
    parser.add_argument("--apply-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    mode = "all"
    if args.extract_only: mode = "extract"
    elif args.segment_only: mode = "segment"
    elif args.reorient_only: mode = "reorient"
    elif args.apply_only: mode = "apply"
    elif args.report_only: mode = "report"

    config = SegmentationConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_grayworld=not args.disable_grayworld,
        use_clahe=not args.disable_clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_grid_size=args.clahe_grid_size,
        min_area=args.min_area,
        mode=mode
    )

    pipeline = ModularPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
