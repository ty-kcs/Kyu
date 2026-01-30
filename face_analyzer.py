"""
FaceAnalyzer Module
OpenFaceのAction Units (AU) から客観的疲労度を算出するモジュール。
分類モデルで「低/中/高」ラベルを返す。
"""

from typing import Dict, Optional, Tuple
import os
import shutil
import subprocess
import json
import io
from datetime import datetime

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
        labels: Optional[Dict[int, str]] = None,
        cache_dir: str = "face_cache"
    ):
        """
        Args:
            model_path: 事前学習済みモデルのパス
            openface_bin: OpenFace FeatureExtraction のパス
            temp_dir: OpenFaceの出力一時ディレクトリ
            labels: 分類ラベルのマッピング
            cache_dir: 画像と推論結果のキャッシュディレクトリ
        """
        self.model_path = model_path
        self.openface_bin = openface_bin
        self.temp_dir = temp_dir
        self.cache_dir = cache_dir
        self.labels = labels or {
            0: "疲れ度:低 (1-2)",
            1: "疲れ度:中 (3-5)",
            2: "疲れ度:高 (6-7)"
        }
        self.model = None
        self.drive_service = None
        
        # キャッシュディレクトリを作成
        os.makedirs(os.path.join(cache_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "results"), exist_ok=True)

    def load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = joblib.load(self.model_path)

    def _extract_file_id(self, url: str) -> Optional[str]:
        """
        Google Drive URLからfile_idを抽出
        """
        if 'id=' in url:
            return url.split('id=')[1].split('&')[0]
        elif '/d/' in url:
            return url.split('/d/')[1].split('/')[0]
        return None

    def _init_drive_service(self):
        """
        Google Drive APIクライアントを初期化
        """
        if self.drive_service is None:
            try:
                from googleapiclient.discovery import build
                from google.auth import default
                creds, _ = default()
                self.drive_service = build('drive', 'v3', credentials=creds)
            except Exception as e:
                print(f"Warning: Failed to initialize Drive service: {e}")
                return False
        return True

    def _download_from_drive(self, url: str) -> Optional[str]:
        """
        Google Drive URLから画像をダウンロード（キャッシュあれば再利用）
        
        Args:
            url: Google Drive共有URL
        
        Returns:
            ローカルパス or None
        """
        file_id = self._extract_file_id(url)
        if not file_id:
            print(f"Warning: Could not extract file_id from URL: {url}")
            return None
        
        # キャッシュパスを生成
        cache_path = os.path.join(self.cache_dir, "images", f"{file_id}.jpg")
        
        # 既にキャッシュがあれば再利用
        if os.path.exists(cache_path):
            return cache_path
        
        # Drive APIクライアント初期化
        if not self._init_drive_service():
            return None
        
        # ダウンロード実行
        try:
            from googleapiclient.http import MediaIoBaseDownload
            
            request = self.drive_service.files().get_media(fileId=file_id)
            fh = io.FileIO(cache_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            return cache_path
        except Exception as e:
            print(f"Error downloading {file_id}: {e}")
            return None

    def _get_result_cache_path(self, file_id: str) -> str:
        """
        推論結果のキャッシュパスを取得
        """
        return os.path.join(self.cache_dir, "results", f"{file_id}.json")

    def _load_result_cache(self, file_id: str) -> Optional[Tuple[str, float, list]]:
        """
        推論結果のキャッシュを読み込み
        """
        cache_path = self._get_result_cache_path(file_id)
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return (data['label'], data['confidence'], data['probabilities'])
        except Exception as e:
            print(f"Warning: Failed to load cache {cache_path}: {e}")
            return None

    def _save_result_cache(self, file_id: str, result: Tuple[str, float, list]):
        """
        推論結果をキャッシュに保存
        """
        cache_path = self._get_result_cache_path(file_id)
        label, confidence, probabilities = result
        
        data = {
            "file_id": file_id,
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_path}: {e}")

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
            proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.stdout:
                print(proc.stdout)
            if proc.stderr:
                print(proc.stderr)
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
        画像パスまたはGoogle Drive URLから客観的疲労度ラベルを算出する統合メソッド。
        
        Args:
            image_path: 顔写真へのパス or Google Drive URL。Noneまたは空文字列の場合はNoneを返す。
        
        Returns:
            (疲労度ラベル, 信頼度, クラス確率配列) 。画像がない場合はNone。
        """
        if not image_path or image_path.strip() == "":
            return None
        
        # Google Drive URLの場合は自動ダウンロード
        file_id = None
        if image_path.startswith("https://drive.google.com"):
            file_id = self._extract_file_id(image_path)
            
            # キャッシュされた結果があれば返す
            if file_id:
                cached_result = self._load_result_cache(file_id)
                if cached_result:
                    return cached_result
            
            # 画像をダウンロード
            image_path = self._download_from_drive(image_path)
            if not image_path:
                return None
        
        try:
            au_df = self.extract_action_units(image_path)
            if au_df is None or au_df.empty:
                return None
            result = self.calculate_objective_fatigue(au_df)
            
            # 結果をキャッシュに保存（Drive URLの場合のみ）
            if file_id:
                self._save_result_cache(file_id, result)
            
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
