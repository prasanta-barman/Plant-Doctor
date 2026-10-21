# Plant-Doctor
A multi‑modal edge AI‑powered application for plant disease identification and remedy recommendation. 


**Bilingual Diagnostic Pipeline (English & Assamese / অসমীয়া)**

---

## 1. Quick Start in 4 Steps

Open your Windows Command Prompt or PowerShell in `Plant Doctor`:

```cmd
cd "Plant Doctor"
```

### Step 1: Initialize Python Environment
```cmd
01_setup_env.bat
```
*This checks Python, creates the `.venv` virtual environment, and installs PyTorch, Ultralytics YOLOv8, and Roboflow SDK.*

### Step 2: Download & Structure Datasets (70/20/10 Split)
```cmd
python 02_download_and_structure_dataset.py
```
*Fetches Guava and Tomato disease images from Roboflow Universe and organizes them into strict 70% train, 20% validation, and 10% test splits across the 8 classes.*

### Step 3: Fine-Tune YOLOv8 with PyTorch
```cmd
python 03_finetune_yolov8.py
```
*Trains `yolov8n-cls` using transfer learning on your GPU or CPU, evaluates on test data, and saves `models/best.pt`.*

### Step 4: Verify Mobile Photos 
1. Copy or drop any photos taken with your mobile phone into:
   `Plant Doctor\test_photos\`
2. Run the offline bilingual verification script:
   ```cmd
   python 04_infer_offline_phone_photos.py
   ```
3. Check the console output for dual-language reports (English + Assamese), and view annotated diagnostic cards in `output/`!

### Step 5: (Optional) Export for Mobile App
```cmd
python 05_export_mobile.py
```
*Exports `models/best.pt` to ONNX and TensorFlow Lite (`.tflite`) for embedding into mobile applications.*

---

## 2. Supported Disease Classes & Translations

| Class Name | English Name | Assamese Name (অসমীয়া) | Pathogen |
| :--- | :--- | :--- | :--- |
| `guava_healthy` | Healthy Guava | সুস্থ মধুৰিআম | Psidium guajava |
| `guava_wilt` | Guava Wilt | মধুৰিআমৰ লেৰেলি যোৱা বেমাৰ | Fusarium oxysporum |
| `guava_anthracnose` | Guava Anthracnose | মধুৰিআমৰ এনথ্ৰাকনোজ / ক'লা দাগ | Colletotrichum gloeosporioides |
| `guava_canker` | Guava Canker | মধুৰিআমৰ কেনকাৰ / খহু বেমাৰ | Pestalotiopsis psidii |
| `tomato_healthy` | Healthy Tomato | সুস্থ বিলাহী | Solanum lycopersicum |
| `tomato_early_blight` | Tomato Early Blight | বিলাহীৰ আগতীয়া পাত পচা দাগ | Alternaria solani |
| `tomato_late_blight` | Tomato Late Blight | বিলাহীৰ পলমকৈ হোৱা পচন বেমাৰ | Phytophthora infestans |
| `tomato_leaf_curl` | Tomato Leaf Curl | বিলাহীৰ পাত কোঁচ খোৱা ভাইৰাছ | TYLCV (Begomovirus) |

---

## 3. Directory Layout in `Plant Doctor`

```
Plant Doctor\
├── .venv\                                 # Isolated virtual environment
├── datasets\                              # 70/20/10 split datasets
│   ├── train\                             # 70% of photos
│   ├── val\                               # 20% of photos
│   └── test\                              # 10% of photos
├── models\                                # Trained weights (.pt, .onnx, .tflite)
├── test_photos\                           # Mobile camera photos to test
├── output\                                # Verified visual result cards
├── 01_setup_env.bat
├── 02_download_and_structure_dataset.py
├── 03_finetune_yolov8.py
├── 04_infer_offline_phone_photos.py
├── 05_export_mobile.py
├── bilingual_dictionary.json
├── roboflow_config.json
├── requirements.txt
└── README.md
```
