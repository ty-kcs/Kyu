import os
import sys
import subprocess
import pandas as pd
import joblib
import shutil

MODEL_PATH = 'fatigue_model.pkl'

OPENFACE_BIN = './OpenFace/build/bin/FeatureExtraction' 

# Temporary  for OpenFace outputs
TEMP_DIR = 'temp_inference'

# Class Labels (Must match the training mapping: 0=Low, 1=Mod, 2=High)
LABELS = {
    0: "疲れ度:低 (1-2)",
    1: "疲れ度:中 (3-5)",
    2: "疲れ度:高 (6-7)"
}

def extract_features(image_path):
    """
    Runs OpenFace on the image and extracts Action Units.
    """
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    cmd = [OPENFACE_BIN, '-f', image_path, '-au_static', '-out_dir', TEMP_DIR]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("Error: OpenFace failed to process the image.")
        return None
    except FileNotFoundError:
        print(f"Error: Could not find OpenFace binary at {OPENFACE_BIN}")
        return None

    filename = os.path.basename(image_path)
    csv_name = os.path.splitext(filename)[0] + '.csv'
    csv_path = os.path.join(TEMP_DIR, csv_name)
    
    if not os.path.exists(csv_path):
        print("Error: Output CSV not found.")
        return None
        
    try:
        df = pd.read_csv(csv_path)
        shutil.rmtree(TEMP_DIR)
        return df
    except Exception as e:
        print(f"Error reading features: {e}")
        return None

def predict_fatigue(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    clf = joblib.load(MODEL_PATH)

    print(f"Processing {image_path}...")
    feat_df = extract_features(image_path)
    
    if feat_df is None or feat_df.empty:
        print("Could not extract features.")
        return

    au_cols = [c for c in feat_df.columns if 'AU' in c and '_r' in c]
    
    features = feat_df[au_cols]

    if features.shape[1] != clf.n_features_in_:
        print(f"Feature mismatch: Model expects {clf.n_features_in_} features, but got {features.shape[1]}.")
        return

    prediction_idx = clf.predict(features)[0]
    probabilities = clf.predict_proba(features)[0]
    
    result_label = LABELS.get(prediction_idx, "Unknown")
    confidence = probabilities[prediction_idx] * 100

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