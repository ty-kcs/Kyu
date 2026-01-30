import os
import sys

from face_analyzer import FaceAnalyzer

MODEL_PATH = "fatigue_model.pkl"
OPENFACE_BIN = "PATH TO FeatureExtraction.exe"  # ←パス入れ替える
TEMP_DIR = "temp_inference"


def predict_fatigue(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    analyzer = FaceAnalyzer(
        model_path=MODEL_PATH,
        openface_bin=OPENFACE_BIN,
        temp_dir=TEMP_DIR,
    )

    print(f"Processing {image_path}...")
    result = analyzer.analyze_image(image_path)
    if result is None:
        print("Could not extract features.")
        return

    result_label, confidence = result

    print("-" * 30)
    print(f"Prediction: {result_label}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 30)

    return result_label, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_fatigue.py <path_to_image>")
    else:
        img_path = sys.argv[1]
        predict_fatigue(img_path)