"""
Plant Doctor POC: Offline Mobile Photo Verification & Bilingual Reporter
=======================================================================
Location: C:\Work\Plant Doctor\04_infer_offline_phone_photos.py

Purpose:
1. Ingests leaf photos captured by your mobile phone in: C:\Work\Plant Doctor\test_photos\
2. Runs 100% OFFLINE inference using: C:\Work\Plant Doctor\models\best.pt
3. Calculates top-1 predicted class and confidence percentage.
4. Generates side-by-side bilingual reports in English and Assamese (অসমীয়া):
     - Plant name: Guava (মধুৰিআম) / Tomato (বিলাহী)
     - Condition & Scientific Pathogen
     - Confidence Level (বিশ্বাসযোগ্যতা)
     - Symptoms (লক্ষণসমূহ)
     - Organic & Chemical Remedies (জৈৱিক আৰু ৰাসায়নিক প্ৰতিকাৰ)
5. Saves visual annotated result cards to: C:\Work\Plant Doctor\output\
"""

import sys
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Force UTF-8 terminal encoding for Assamese script
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_DIR = Path("C:/Work/Plant Doctor").resolve()
MODELS_DIR = WORKSPACE_DIR / "models"
TEST_PHOTOS_DIR = WORKSPACE_DIR / "test_photos"
OUTPUT_DIR = WORKSPACE_DIR / "output"
DICTIONARY_FILE = WORKSPACE_DIR / "bilingual_dictionary.json"


def load_bilingual_dictionary():
    """Loads English and Assamese pathology dictionary."""
    if not DICTIONARY_FILE.exists():
        print(f"[ERROR] Bilingual dictionary not found at: {DICTIONARY_FILE}")
        sys.exit(1)

    with open(DICTIONARY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def print_bilingual_terminal_report(photo_name: str, class_id: str, confidence: float, top3: list, dict_entry: dict):
    """Prints a structured, high-legibility dual-language report in the console."""
    conf_pct = confidence * 100
    
    if conf_pct >= 85.0:
        conf_level_en = "High Confidence (Definitive Diagnosis)"
        conf_level_as = "উচ্চ বিশ্বাসযোগ্যতা (নিশ্চিত ৰোগ নিৰ্ণয়)"
    elif conf_pct >= 65.0:
        conf_level_en = "Moderate Confidence (Field Verification Recommended)"
        conf_level_as = "মধ্যমীয়া বিশ্বাসযোগ্যতা (পুনৰীক্ষণৰ প্ৰয়োজন)"
    else:
        conf_level_en = "Low Confidence (Inconclusive / Retake Photo)"
        conf_level_as = "কম বিশ্বাসযোগ্যতা (স্পষ্ট ফটো পুনৰ তোলক)"

    print("\n" + "=" * 80)
    print(f"  PLANT DOCTOR DIAGNOSIS / উদ্ভিদ চিকিৎসকৰ ৰিপ'ৰ্ট")
    print("=" * 80)
    print(f"  Photo File / ফটোৰ নাম      : {photo_name}")
    print(f"  Confidence / বিশ্বাসযোগ্যতা : {conf_pct:.2f}%  [{conf_level_en}]")
    print(f"                              [{conf_level_as}]")
    print("-" * 80)
    print(f"  [ENGLISH]  Plant Species   : {dict_entry['plant_name_en']}")
    print(f"             Diagnosis       : {dict_entry['disease_name_en']}")
    print(f"             Pathogen        : {dict_entry['scientific_name']}")
    print(f"  [অসমীয়া]   উদ্ভিদৰ নাম     : {dict_entry['plant_name_as']}")
    print(f"             ৰোগৰ নিৰ্ণয়      : {dict_entry['disease_name_as']}")
    print(f"             বীজাণুৰ তথ্য     : {dict_entry['pathogen_as']}")
    print("-" * 80)
    print("  SYMPTOMS / ৰোগৰ লক্ষণসমূহ:")
    for s_en, s_as in zip(dict_entry['symptoms_en'], dict_entry['symptoms_as']):
        print(f"   • [EN] {s_en}")
        print(f"     [AS] {s_as}")
    print("-" * 80)
    print("  RECOMMENDED REMEDIES / প্ৰতিকাৰ আৰু পৰামৰ্শ:")
    print("  Organic Treatments (জৈৱিক ব্যৱস্থা):")
    for r_en, r_as in zip(dict_entry['organic_remedies_en'], dict_entry['organic_remedies_as']):
        print(f"   ✓ [EN] {r_en}")
        print(f"     [AS] {r_as}")
    print("  Chemical / Conventional Management (ৰাসায়নিক চিকিৎসা):")
    for c_en, c_as in zip(dict_entry['chemical_remedies_en'], dict_entry['chemical_remedies_as']):
        print(f"   ✓ [EN] {c_en}")
        print(f"     [AS] {c_as}")
    print("-" * 80)
    print("  DIFFERENTIAL DIAGNOSES (TOP ALTERNATIVE PROBABILITIES):")
    for alt_cls, alt_prob in top3:
        print(f"   - {alt_cls:<25} : {alt_prob * 100:>6.2f}%")
    print("=" * 80 + "\n")


def create_annotated_card(image_path: Path, class_id: str, confidence: float, dict_entry: dict):
    """
    Renders an annotated verification card in output/ with bilingual labels.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # Create bottom banner for diagnosis
        banner_height = max(140, int(h * 0.28))
        canvas = Image.new("RGB", (w, h + banner_height), color=(15, 23, 42)) # Deep slate
        canvas.paste(img, (0, 0))

        draw = ImageDraw.Draw(canvas)
        
        # Draw status bar
        conf_pct = confidence * 100
        status_text = f"DIAGNOSIS: {dict_entry['disease_name_en']} | {conf_pct:.1f}% CONFIDENCE"
        draw.text((20, h + 15), status_text, fill=(34, 197, 94)) # Emerald

        as_title = f"{dict_entry['plant_name_as']}: {dict_entry['disease_name_as']}"
        draw.text((20, h + 45), as_title, fill=(255, 255, 255))

        pathogen_text = f"Pathogen: {dict_entry['scientific_name']}"
        draw.text((20, h + 75), pathogen_text, fill=(148, 163, 184))

        out_path = OUTPUT_DIR / f"{image_path.stem}_verified.jpg"
        canvas.save(out_path, quality=95)
        print(f"  [SAVED] Annotated visual report: {out_path}")
    except Exception as e:
        print(f"  [NOTE] Could not write annotated image card ({e}). Console report printed.")


def main():
    print("==============================================================================")
    print("[PLANT DOCTOR POC] Offline Mobile Verification Engine (English & Assamese)")
    print("==============================================================================")

    model_path = MODELS_DIR / "best.pt"
    if not model_path.exists():
        print(f"[ERROR] Trained model weights not found at: {model_path}")
        print("Please train the model first by running: python 03_finetune_yolov8.py")
        sys.exit(1)

    bilingual_dict = load_bilingual_dictionary()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

    # Scan test_photos folder
    supported_exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    photos = [p for p in TEST_PHOTOS_DIR.iterdir() if p.suffix.lower() in supported_exts]

    if not photos:
        print(f"\n[INFO] No photos found in: {TEST_PHOTOS_DIR}")
        print("Please copy or transfer some leaf photos from your phone into that folder.")
        print("Creating a sample verification photo now so you can test immediately...")
        sample_img = Image.new("RGB", (640, 640), color=(34, 139, 34))
        sample_path = TEST_PHOTOS_DIR / "phone_leaf_sample.jpg"
        sample_img.save(sample_path)
        photos = [sample_path]

    print(f"\n[1/2] Loading YOLOv8 offline model from: {model_path}")
    model = YOLO(str(model_path))

    print(f"[2/2] Running offline inference on {len(photos)} phone photo(s)...\n")

    for photo in photos:
        # Run offline inference
        results = model.predict(source=str(photo), verbose=False)
        result = results[0]

        # Extract top-1 and differential probabilities
        top1_idx = result.probs.top1
        top1_conf = float(result.probs.top1conf)
        top1_class = result.names[top1_idx]

        # Top 3 differential diagnoses
        top3_indices = result.probs.top5[:3]
        top3_data = [(result.names[idx], float(result.probs.data[idx])) for idx in top3_indices]

        # Retrieve bilingual pathology entry
        dict_entry = bilingual_dict.get(top1_class, {
            "plant_name_en": "Unknown Plant",
            "plant_name_as": "অজ্ঞাত উদ্ভিদ",
            "disease_name_en": top1_class,
            "disease_name_as": top1_class,
            "scientific_name": "Unspecified Pathogen",
            "pathogen_as": "অনিৰ্দিষ্ট বীজাণু",
            "symptoms_en": ["Consult local agricultural extension officer"],
            "symptoms_as": ["স্থানীয় কৃষি বিষয়াৰ পৰামৰ্শ লওক"],
            "organic_remedies_en": ["Isolate plant to prevent potential spread"],
            "organic_remedies_as": ["ৰোগ নিৰ্ণয় নোহোৱালৈকে গছজোপা সুকীয়াকৈ ৰাখক"],
            "chemical_remedies_en": ["Apply broad-spectrum copper fungicide if needed"],
            "chemical_remedies_as": ["প্ৰয়োজনবোধে কপাৰ ভেঁকুৰনাশক ব্যৱহাৰ কৰিব পাৰে"]
        })

        # Print console output
        print_bilingual_terminal_report(photo.name, top1_class, top1_conf, top3_data, dict_entry)
        create_annotated_card(photo, top1_class, top1_conf, dict_entry)

    print("==============================================================================")
    print("OFFLINE VERIFICATION RUN COMPLETE")
    print(f"All reports saved to: {OUTPUT_DIR}")
    print("==============================================================================")


if __name__ == "__main__":
    main()
