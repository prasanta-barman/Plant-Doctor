"""
Plant Doctor POC: YOLOv8 PyTorch Fine-Tuning Script
===================================================
Location: C:\Work\Plant Doctor\03_finetune_yolov8.py

Purpose:
1. Loads pre-trained Ultralytics YOLOv8 nano classification model ('yolov8n-cls.pt').
2. Uses transfer learning on Guava (মধুৰিআম) & Tomato (বিলাহী) leaf images.
3. Configures PyTorch DataLoader, mixed precision (AMP), and Cosine LR scheduler.
4. Validates on the 20% validation split and tests on the 10% test split.
5. Saves the top performing weights as: C:\Work\Plant Doctor\models\best.pt
"""

import sys
import torch
from pathlib import Path
from ultralytics import YOLO

# Force UTF-8 terminal encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_DIR = Path("C:/Work/Plant Doctor").resolve()
DATASETS_DIR = WORKSPACE_DIR / "datasets"
MODELS_DIR = WORKSPACE_DIR / "models"


def check_compute_device():
    """Detects NVIDIA CUDA GPU acceleration or falls back to CPU."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        print(f"[COMPUTE] NVIDIA GPU Acceleration Enabled: {device_name} (Total GPUs: {device_count})")
        return 0  # GPU device 0
    else:
        print("[COMPUTE] No CUDA GPU detected. Running on multi-core CPU.")
        return "cpu"


def main():
    print("==============================================================================")
    print("[PLANT DOCTOR POC] Fine-Tuning YOLOv8 with PyTorch for Guava & Tomato")
    print("==============================================================================")

    # 1. Verify dataset structure exists
    train_dir = DATASETS_DIR / "train"
    val_dir = DATASETS_DIR / "val"
    test_dir = DATASETS_DIR / "test"

    if not (train_dir.exists() and val_dir.exists()):
        print("[ERROR] Datasets directory is missing or empty!")
        print("Please run: python 02_download_and_structure_dataset.py first.")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = check_compute_device()

    # 2. Initialize pre-trained YOLOv8 nano classifier
    print("\n[1/3] Loading pre-trained YOLOv8n-cls backbone...")
    model = YOLO("yolov8n-cls.pt")  # Lightweight, ideal for mobile export

    # 3. Fine-tuning parameters
    print("[2/3] Commencing model training...")
    print(f"      Dataset root: {DATASETS_DIR}")
    print(f"      Epochs: 25 | Image size: 224x224 | Batch: 32 | Optimizer: AdamW\n")

    results = model.train(
        data=str(DATASETS_DIR),       # Points to datasets/ with train/ and val/
        epochs=25,                    # 25 epochs is optimal for POC convergence
        imgsz=224,                    # 224x224 standard for mobile inference latency
        batch=32,                     # Batch size (reduce to 16 if low RAM)
        device=device,                # 0 for GPU, 'cpu' for CPU
        workers=4,                    # Multiprocessing DataLoader workers
        optimizer="AdamW",            # AdamW optimizer with weight decay
        lr0=0.001,                    # Initial learning rate
        lrf=0.01,                     # Final learning rate fraction (cosine)
        dropout=0.2,                  # Dropout for regularizing foliage features
        augment=True,                 # Flips, rotations, illumination jitter
        project=str(WORKSPACE_DIR / "runs"),
        name="plant_doctor_yolov8",
        exist_ok=True
    )

    # 4. Copy best checkpoint to models/best.pt
    best_checkpoint = Path(results.save_dir) / "weights" / "best.pt"
    target_best_pt = MODELS_DIR / "best.pt"

    if best_checkpoint.exists():
        import shutil
        shutil.copy2(best_checkpoint, target_best_pt)
        print(f"\n[SUCCESS] Model successfully trained and saved!")
        print(f"          Target model: {target_best_pt}")
    else:
        # Fallback save
        model.save(str(target_best_pt))
        print(f"\n[SUCCESS] Model saved to {target_best_pt}")

    # 5. Evaluate on held-out test split
    if test_dir.exists():
        print("\n[3/3] Evaluating accuracy on unseen 10% test split...")
        metrics = model.val(data=str(DATASETS_DIR), split="test")
        print(f"      Top-1 Accuracy on Test Set: {metrics.top1:.4f} ({metrics.top1 * 100:.2f}%)")
        print(f"      Top-5 Accuracy on Test Set: {metrics.top5:.4f} ({metrics.top5 * 100:.2f}%)")

    print("\n==============================================================================")
    print("TRAINING PIPELINE COMPLETE")
    print("Next step: Put mobile leaf photos into 'test_photos/' and run:")
    print("           python 04_infer_offline_phone_photos.py")
    print("==============================================================================")


if __name__ == "__main__":
    main()
