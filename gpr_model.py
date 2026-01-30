"""
FatiguePredictor Module
ガウス過程回帰 (GPR) を用いて、24時間の中で疲れやすい時間帯を予測する。
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import pandas as pd


class FatiguePredictor:
    """
    疲労度予測モデル (Gaussian Process Regression)
    
    特徴量: 時刻 (0.0 ~ 24.0)
    ターゲット: 主観的疲労度 (1-7)、または客観補正後の疲労度
    """

    def __init__(
        self,
        use_objective_correction: bool = True,
        correction_threshold: float = 2.0,
        correction_weight: float = 0.7
    ):
        """
        Args:
            use_objective_correction: 客観補正を使用するかどうか
            correction_threshold: 主観-客観の差分がこの値を超えたら補正を適用
            correction_weight: 補正時の客観値の重み (0.0-1.0)
                              例: 0.7 = 客観70% + 主観30%
        """
        self.use_objective_correction = use_objective_correction
        self.correction_threshold = correction_threshold
        self.correction_weight = correction_weight
        
        # Kernel: RBF (滑らかな変化) + WhiteKernel (ノイズ)
        #length_scale: 値が大きいほど、よりゆったりした変化になる
        kernel = RBF(length_scale=3.0, length_scale_bounds=(0.5, 10.0)) + \
                 WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3, 2.0))
        
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True
        )
        
        self.is_fitted = False

    def preprocess_timestamps(self, timestamps: pd.Series) -> np.ndarray:
        """
        タイムスタンプを「一日のうちの時刻」(0.0 ~ 24.0)に変換
        
        Args:
            timestamps: datetime型のPandas Series
        
        Returns:
            時刻の配列 (N,) shape
        """
        timestamps = pd.to_datetime(timestamps)
        hours = timestamps.dt.hour + timestamps.dt.minute / 60.0
        return hours.values

    def apply_objective_correction(
        self,
        subjective: np.ndarray,
        objective: np.ndarray
    ) -> np.ndarray:
        """
        主観と客観の乖離が大きい場合、客観値を重視して補正。
        
        Args:
            subjective: 主観的疲労度 (N,)
            objective: 客観的疲労度 (N,)
        
        Returns:
            補正後の疲労度 (N,)
        """
        corrected = subjective.copy()
        objective_series = pd.to_numeric(pd.Series(objective), errors="coerce")
        objective_values = objective_series.to_numpy()
        
        for i in range(len(subjective)):
            if np.isnan(objective_values[i]):
                # 客観データ(顔写真）がない場合は主観をそのまま使用
                continue
            
            diff = objective_values[i] - subjective[i]
            
            # 客観が主観よりthreshold以上高い場合のみ補正
            if diff > self.correction_threshold:
                # 客観値を重視した加重平均
                corrected[i] = (
                    self.correction_weight * objective_values[i] +
                    (1 - self.correction_weight) * subjective[i]
                )
                
        return corrected

    def fit(
        self,
        timestamps: pd.Series,
        subjective_fatigue: np.ndarray,
        objective_fatigue: Optional[np.ndarray] = None
    ) -> None:
        """
        モデルを学習。
        
        Args:
            timestamps: タイムスタンプ列
            subjective_fatigue: 主観的疲労度 (1-7)
            objective_fatigue: 客観的疲労度 (1-7)。use_objective_correction=True時に必要。
        """
        X = self.preprocess_timestamps(timestamps).reshape(-1, 1)
        y = subjective_fatigue.copy()
        
        # ラベルに客観補正を適用
        if self.use_objective_correction and objective_fatigue is not None:
            y = self.apply_objective_correction(y, objective_fatigue)
        
        # NaN除去
        valid_mask = ~np.isnan(y) #yがNaNでないインデックス
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 3:
            raise ValueError(f"学習データが不足しています。最低3件必要ですが{len(X)}件しかありません。")
        
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(
        self,
        timestamps: Optional[pd.Series] = None,
        hours: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        疲労度を予測。
        
        Args:
            timestamps: 予測対象のタイムスタンプ。Noneの場合は24時間分を自動生成。
            hours: 直接時刻配列を指定 (0-24)。timestampsより優先。
        
        Returns:
            (予測平均値, 予測標準偏差) のタプル
            - 平均値: 疲労度の期待値
            - 標準偏差: 予測の不確かさ (高いほど「データ不足」)
        """
        if not self.is_fitted:
            raise RuntimeError("モデルが学習されていません。先にfit()を実行してください。")
        
        if hours is not None:
            X_test = np.array(hours).reshape(-1, 1)#2次元配列に変換
        elif timestamps is not None:
            X_test = self.preprocess_timestamps(timestamps).reshape(-1, 1)
        else:
            # デフォルト: 24時間を0.5時間刻みで予測
            X_test = np.arange(0, 24, 0.5).reshape(-1, 1)
        
        #GPRのpredict methodで予測
        y_mean, y_std = self.model.predict(X_test, return_std=True)
        return y_mean, y_std

    def get_high_uncertainty_periods(
        self,
        threshold_std: float = 1.0
    ) -> np.ndarray:
        """
        予測の不確かさ(標準偏差)が高い時間帯を抽出。
        これらの時間帯はデータ不足なので、chatgpt apiを活用して、
        通知を送ってデータ収集すべき時間帯として活用すべし。
        
        Args:
            threshold_std(float): この値以上の標準偏差を持つ時間帯を返す
    
        Returns:
            不確かさが高い時刻の配列 (例: [3.5, 4.0, 14.5, ...])
        """
        if not self.is_fitted:
            raise RuntimeError("モデルが学習されていません。")
        
        X_test = np.arange(0, 24, 0.5).reshape(-1, 1)
        _, y_std = self.model.predict(X_test, return_std=True)
        
        high_uncertainty_mask = y_std >= threshold_std
        high_uncertainty_hours = X_test[high_uncertainty_mask].flatten()
        
        return high_uncertainty_hours
   
    def get_peak_time(self):
        if not self.is_fitted:
            return None
        hours = np.arange(0, 24, 0.25)
        mean_normal, _ = self.predict(hours=hours)
        max_index = np.argmax(mean_normal)
        peak_time = hours[max_index]
        return peak_time


if __name__ == "__main__":
    '''
    ダミーデータに対して、主観のみと客観補正ありで簡易テストを行う。
    15時の予測値と、不確かさの高い時間帯を表示。
    '''
    from datetime import datetime, timedelta
    
    # ダミーデータ生成
    base_time = datetime(2025, 1, 1, 9, 0)
    timestamps = pd.Series([
        base_time + timedelta(hours=i) for i in [0, 3, 6, 9, 12, 15, 18, 21]
    ])
    
    # ex)朝は元気(1)、夕方に疲れる(5)パターン
    subjective = np.array([1, 2, 2, 3, 4, 5, 4, 3])
    objective = np.array([2, 2, 3, 4, 6, 6, 5, 4])  # 客観は主観より高め

    
    # 主観のみ予測
    predictor_normal = FatiguePredictor(use_objective_correction=False)
    predictor_normal.fit(timestamps, subjective)
    
    # 客観補正あり予測
    predictor_corrected = FatiguePredictor(use_objective_correction=True)
    predictor_corrected.fit(timestamps, subjective, objective)
    
    # 15時の予測を比較
    test_hour = np.array([15.0])
    mean_normal, std_normal = predictor_normal.predict(hours=test_hour)
    mean_corrected, std_corrected = predictor_corrected.predict(hours=test_hour)
    
    print("=== 15:00の疲労度予測 ===")
    print(f"主観のみ:     {mean_normal[0]:.2f} ± {std_normal[0]:.2f}")
    print(f"客観補正あり: {mean_corrected[0]:.2f} ± {std_corrected[0]:.2f}")
    
    # 不確かさの高い時間帯
    uncertain_hours = predictor_corrected.get_high_uncertainty_periods(threshold_std=0.8)
    print(f"\nデータ不足時間帯: {uncertain_hours}")
