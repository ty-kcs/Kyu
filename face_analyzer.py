"""
FaceAnalyzer Module
OpenFaceのAction Units (AU) から客観的疲労度を算出するモジュール。
分類モデルで「低/中/高」ラベルを返す。
"""

from typing import Dict, Optional, Tuple
import os
import shutil
import subprocess

import joblib
import pandas as pd


class FaceAnalyzer:
    """
    顔写真から客観的な疲れ度を算出するクラス。
    OpenFaceのAU (Action Units) 出力をもとに、分類モデルでラベルを推論する。
    """

    def __init__(
        self,
        model_path: str = "fatigue_model.pkl",
        openface_bin: Optional[str] = None,
        temp_dir: str = "temp_inference",
        labels: Optional[Dict[int, str]] = None
    ):
        """
        Args:
            model_path: 事前学習済みモデルのパス
            openface_bin: OpenFace FeatureExtraction のパス
            temp_dir: OpenFaceの出力一時ディレクトリ
            labels: 分類ラベルのマッピング
        """
        self.model_path = model_path
        self.openface_bin = openface_bin
        self.temp_dir = temp_dir
        self.labels = labels or {
            0: "疲れ度:低 (1-2)",
            1: "疲れ度:中 (3-5)",
            2: "疲れ度:高 (6-7)"
        }
        self.model = None

    def load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = joblib.load(self.model_path)

    def extract_action_units(self, image_path: str) -> Optional[pd.DataFrame]:
        """
        画像からAction Unitsを抽出する。

        Args:
            image_path: 顔写真へのパス

        Returns:
            OpenFaceのCSVを読み込んだDataFrame。失敗時はNone。
        """
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not self.openface_bin:
            raise ValueError("OpenFace binary path is not set.")

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        cmd = [self.openface_bin, "-f", image_path, "-au_static", "-out_dir", self.temp_dir]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print("Error: OpenFace failed to process the image.")
            return None
        except FileNotFoundError:
            print(f"Error: Could not find OpenFace binary at {self.openface_bin}")
            return None

        filename = os.path.basename(image_path)
        csv_name = os.path.splitext(filename)[0] + ".csv"
        csv_path = os.path.join(self.temp_dir, csv_name)

        if not os.path.exists(csv_path):
            print("Error: Output CSV not found.")
            return None

        try:
            df = pd.read_csv(csv_path)
            shutil.rmtree(self.temp_dir)
            return df
        except Exception as e:
            print(f"Error reading features: {e}")
            return None

    def calculate_objective_fatigue(self, au_df: pd.DataFrame) -> Tuple[str, float, list]:
        """
        AU特徴量から分類モデルで疲労度ラベルを推論する。

        Args:
            au_df: extract_action_units() の出力DataFrame

        Returns:
            (ラベル, 信頼度[%], クラス確率配列)
        """
        if self.model is None:
            self.load_model()

        au_cols = [c for c in au_df.columns if "AU" in c and "_r" in c]
        features = au_df[au_cols]

        if features.empty:
            raise ValueError("No AU features found in OpenFace output.")

        if features.shape[1] != self.model.n_features_in_:
            raise ValueError(
                f"Feature mismatch: Model expects {self.model.n_features_in_} features, "
                f"but got {features.shape[1]}."
            )

        prediction_idx = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]  # 確率配列
        result_label = self.labels.get(prediction_idx, "Unknown")
        confidence = float(probabilities[prediction_idx] * 100)

        return result_label, confidence, probabilities.tolist()

    def analyze_image(self, image_path: Optional[str]) -> Optional[Tuple[str, float, list]]:
        """
        画像パスから客観的疲労度ラベルを算出する統合メソッド。
        
        Args:
            image_path: 顔写真へのパス。Noneまたは空文字列の場合はNoneを返す。
        
        Returns:
            (疲労度ラベル, 信頼度, クラス確率配列) 。画像がない場合はNone。
        """
        if not image_path or image_path.strip() == "":
            return None
        
        try:
            au_df = self.extract_action_units(image_path)
            if au_df is None or au_df.empty:
                return None
            result = self.calculate_objective_fatigue(au_df)
            return result
        except Exception as e:
            print(f"Warning: Failed to analyze {image_path}: {e}")
            return None


if __name__ == "__main__":
    # 簡易テスト
    analyzer = FaceAnalyzer(
        model_path="fatigue_model.pkl",
        openface_bin="PATH TO FeatureExtraction.exe"
    )

    test_paths = [
        "https://drive.google.com/open?id=1vt1tm4N4f7r834ZNPq1PYEX9vmIjhq9Q",
        "https://drive.google.com/open?id=1ZI3vWqWUJcs3R9fFmAQZet-rMh-fhYp6",
    ]

    for path in test_paths:
        result = analyzer.analyze_image(path)
        print(f"Path: {path} -> {result}")
