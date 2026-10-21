"""
Plant Doctor POC: Mobile Model Exporter (ONNX & TFLite)
======================================================
Location: C:\Work\Plant Doctor\05_export_mobile.py

Purpose:
Converts C:\Work\Plant Doctor\models\best.pt into optimized mobile formats:
1. models/best.onnx   -> Compatible with ONNX Runtime Mobile (Android / iOS / Flutter)
2. models/best.tflite -> Compatible with TensorFlow Lite Android & iOS camera runners
"""

import sys
from pathlib import Path
from ultralytics import YOLO

# Force UTF-8 terminal encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_DIR = Path("C:/Work/Plant Doctor").resolve()
MODELS_DIR = WORKSPACE_DIR / "models"


def main():
    print("==============================================================================")
    print("[PLANT DOCTOR POC] Exporting YOLOv8 PyTorch Weights for Mobile Deployment")
    print("==============================================================================")

    model_path = MODELS_DIR / "best.pt"
    if not model_path.exists():
        print(f"[ERROR] Trained model weights not found at: {model_path}")
        print("Please train the model first by running: python 03_finetune_yolov8.py")
        sys.exit(1)

    print(f"[1/3] Loading PyTorch model from: {model_path}")
    model = YOLO(str(model_path))

    # Export to ONNX
    print("\n[2/3] Exporting to ONNX format (models/best.onnx)...")
    try:
        onnx_file = model.export(format="onnx", imgsz=224, dynamic=False)
        print(f"      [SUCCESS] Exported ONNX model: {onnx_file}")
    except Exception as e:
        print(f"      [WARNING] ONNX export encountered: {e}")

    # Export to TFLite (TensorFlow Lite)
    print("\n[3/3] Exporting to TFLite format (models/best_saved_model/best_float32.tflite)...")
    try:
        tflite_file = model.export(format="tflite", imgsz=224)
        print(f"      [SUCCESS] Exported TFLite mobile model: {tflite_file}")
    except Exception as e:
        print(f"      [WARNING] TFLite export encountered: {e}")
        print("                Ensure tensorflow is installed: pip install tensorflow")

    print("\n==============================================================================")
    print("MOBILE EXPORT COMPLETE")
    print("You can now embed these exported models directly into an offline Android or")
    print("iOS application for realtime camera leaf diagnosis without any server!")
    print("==============================================================================")


if __name__ == "__main__":
    main()
