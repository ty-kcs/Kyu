"""
FaceAnalyzer Module
OpenFaceのAction Units (AU) から客観的疲労度を算出するモジュール。
今回は仮実装。
"""

from typing import Dict, Optional
import numpy as np
import os


class FaceAnalyzer:
    """
    顔写真から客観的な疲れ度を算出するクラス。
    OpenFaceのAU (Action Units) 出力をもとに、疲労スコア(1-7)を計算する。
    """

    def __init__(
        self,
        weight_au04: float = 1.0,
        weight_au43: float = 1.5,
        scaling_factor: float = 1.2
    ):
        """
        Args:
            weight_au04: AU04 (Brow Lowerer - ストレス・緊張) の重み
            weight_au43: AU43 (Eyes Closed - 眠気) の重み
            scaling_factor: 最終スコアへの変換スケール係数
        """
        self.weight_au04 = weight_au04
        self.weight_au43 = weight_au43
        self.scaling_factor = scaling_factor

    def extract_action_units(self, image_path: str) -> Dict[str, float]:
        """
        画像からAction Unitsを抽出する (モック実装)。
        本番環境ではOpenFaceバイナリを実行してCSV出力を解析する。
        今回は画像ファイルの存在チェックのみ行い、ランダムなAU値を返す。
        Args:
            image_path: 顔写真へのパス (Google Driveリンクの場合は事前ダウンロード想定)
        Returns:
            AU値の辞書 (キー: AU名, 値: 強度 0.0-5.0)
        
        Raises:
            FileNotFoundError: 画像ファイルが存在しない場合
        """
        # 実装時のダミーチェック
        if image_path and not image_path.startswith("http"):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

        # --- モック: 実際はOpenFaceを実行 ---
        # 例: subprocess.run(['FeatureExtraction', '-f', image_path, '-out_dir', ...])
        # その後、生成されたCSVを読み込み
        
        # 今はランダム値を返すが、実データでは0.0-5.0の範囲でAUが出力される
        np.random.seed(hash(image_path) % 2**32)  # 再現性のため
        au_values = {
            "AU04_r": np.random.uniform(0.5, 3.5),  # Brow Lowerer
            "AU43_r": np.random.uniform(0.0, 2.5),  # Eyes Closed
            "AU06_r": np.random.uniform(0.0, 2.0),  # Cheek Raiser (参考用)
            "AU12_r": np.random.uniform(0.0, 3.0),  # Lip Corner Puller (笑顔)
        }
        return au_values

    def calculate_objective_fatigue(self, au_values: Dict[str, float]) -> float:
        """
        Action Units から客観的疲労度スコア(1-7)を計算。
        
        Formula:
            Score_obj = 1 + min(6, (w1 * AU04 + w2 * AU43) * scaling_factor)
        
        Args:
            au_values: extract_action_units() の出力
        
        Returns:
            客観的疲労度 (1.0 - 7.0)
        """
        au04 = au_values.get("AU04_r", 0.0)
        au43 = au_values.get("AU43_r", 0.0)
        
        weighted_sum = (
            self.weight_au04 * au04 +
            self.weight_au43 * au43
        )
        
        score = 1.0 + min(6.0, weighted_sum * self.scaling_factor)
        return round(score, 2)

    def analyze_image(self, image_path: Optional[str]) -> Optional[float]:
        """
        画像パスから客観的疲労度を算出する統合メソッド。
        
        Args:
            image_path: 顔写真へのパス。Noneまたは空文字列の場合はNoneを返す。
        
        Returns:
            客観的疲労度スコア (1-7)。画像がない場合はNone。
        """
        if not image_path or image_path.strip() == "":
            return None
        
        try:
            au_values = self.extract_action_units(image_path)
            objective_score = self.calculate_objective_fatigue(au_values)
            return objective_score
        except Exception as e:
            print(f"Warning: Failed to analyze {image_path}: {e}")
            return None


if __name__ == "__main__":
    # 簡易テスト
    analyzer = FaceAnalyzer()
    
    # テスト用パス
    test_paths = [
        "https://drive.google.com/open?id=1vt1tm4N4f7r834ZNPq1PYEX9vmIjhq9Q",
        "https://drive.google.com/open?id=1ZI3vWqWUJcs3R9fFmAQZet-rMh-fhYp6",
    ]
    
    for path in test_paths:
        if path:
            au = analyzer.extract_action_units(path)
            score = analyzer.calculate_objective_fatigue(au)
            print(f"Path: {path}")
            print(f"  AU04: {au['AU04_r']:.2f}, AU43: {au['AU43_r']:.2f}")
            print(f"  Objective Fatigue: {score}")
        else:
            result = analyzer.analyze_image(path)
            print(f"Path: (empty) -> {result}")
