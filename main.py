"""
Main Script
データロード、FaceAnalyzer + FatiguePredictor の統合実行、可視化。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime, timedelta
from pathlib import Path

from face_analyzer import FaceAnalyzer
from gpr_model import FatiguePredictor


# 日本語フォント設定 (環境によって調整が必要)
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meiryo']
rcParams['axes.unicode_minus'] = False


def load_data(csv_path: str) -> pd.DataFrame:
    """
    CSVデータを読み込み、前処理を行う。
    
    Args:
        csv_path: response.csv のパス
    
    Returns:
        前処理済みDataFrame
    """
    df = pd.read_csv(csv_path)
    
    # タイムスタンプをdatetime型に変換
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %H:%M:%S')
    
    # photo_linkがない行は空文字に統一
    df['photo_link'] = df['photo_link'].fillna('')
    
    # 主観的疲労度の欠損値を除外
    df = df.dropna(subset=['subjective_fatigue'])
    
    # ソート
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    return df


def analyze_faces(df: pd.DataFrame, analyzer: FaceAnalyzer) -> pd.DataFrame:
    """
    DataFrameの各行に対して顔解析を実行し、objective_fatigueカラム(客観疲労度スコアを入れる）を追加。
    
    Args:
        df: 元のDataFrame
        analyzer: FaceAnalyzerインスタンス
    
    Returns:
        objective_fatigueカラムが追加されたDataFrame
    """
    df = df.copy()
    objective_scores = []
    
    for idx, row in df.iterrows():
        photo_link = row['photo_link']
        score = analyzer.analyze_image(photo_link)
        objective_scores.append(score)
    
    df['objective_fatigue'] = objective_scores
    return df


def plot_fatigue_timeline(
    df: pd.DataFrame,
    predictor_normal: FatiguePredictor,
    predictor_corrected: FatiguePredictor,
    user_id: str = None
):
    """
    疲労度タイムラインをプロット。
    
    Args:
        df: データフレーム
        predictor_normal: 客観補正なしの予測器
        predictor_corrected: 客観補正ありの予測器
        user_id: 特定ユーザーのみ表示する場合
    """
    if user_id:
        df = df[df['user_id'] == user_id].copy()
    
    # 24時間分の予測
    hours = np.arange(0, 24, 0.25)
    mean_normal, std_normal = predictor_normal.predict(hours=hours)
    mean_corrected, std_corrected = predictor_corrected.predict(hours=hours)
    
    # 実測データの時刻抽出
    df['hour'] = df['Timestamp'].dt.hour + df['Timestamp'].dt.minute / 60.0
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # --- Plot 1: 主観のみ ---
    ax1.scatter(df['hour'], df['subjective_fatigue'], 
                alpha=0.5, label='主観的疲労度', color='blue', s=50)
    
    ax1.plot(hours, mean_normal, 'b-', label='予測 (主観のみ)', linewidth=2)
    ax1.fill_between(hours, 
                     mean_normal - 1.96 * std_normal,
                     mean_normal + 1.96 * std_normal,
                     alpha=0.2, color='blue', label='95%信頼区間')
    
    ax1.set_xlabel('時刻', fontsize=12)
    ax1.set_ylabel('疲労度 (1-7)', fontsize=12)
    ax1.set_title(f'疲労度タイムライン - 主観のみ (User: {user_id or "All"})', fontsize=14)
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, 8)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # --- Plot 2: 客観補正あり ---
    ax2.scatter(df['hour'], df['subjective_fatigue'], 
                alpha=0.5, label='主観的疲労度', color='blue', s=50)
    
    # 客観データがある箇所もプロット
    df_with_obj = df[df['objective_fatigue'].notna()]
    ax2.scatter(df_with_obj['hour'], df_with_obj['objective_fatigue'], 
                alpha=0.5, label='客観的疲労度 (OpenFace)', color='red', s=50, marker='x')
    
    ax2.plot(hours, mean_corrected, 'r-', label='予測 (客観補正あり)', linewidth=2)
    ax2.fill_between(hours, 
                     mean_corrected - 1.96 * std_corrected,
                     mean_corrected + 1.96 * std_corrected,
                     alpha=0.2, color='red', label='95%信頼区間')
    
    ax2.set_xlabel('時刻', fontsize=12)
    ax2.set_ylabel('疲労度 (1-7)', fontsize=12)
    ax2.set_title(f'疲労度タイムライン - 客観補正あり (User: {user_id or "All"})', fontsize=14)
    ax2.set_xlim(0, 24)
    ax2.set_ylim(0, 8)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    #TODO! これをChatGPT APIに渡してアドバイス生成とかに活用する？
    # --- ChatGPT APIに渡すと効果的なグラフ？ ---
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    out_png = output_dir / f'fatigue_timeline_{user_id or "all"}.png'
    plt.savefig(out_png, dpi=150)
    print(f"グラフを保存しました: {out_png}")
    plt.show()


def analyze_discrepancy(df: pd.DataFrame):
    """
    主観と客観のズレ(乖離)を分析してレポート。→うまくこれをchatgpt apiに投げる？
    """
    #objective_fatigueが存在する行のみ抽出
    df_valid = df[df['objective_fatigue'].notna()].copy()
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    if len(df_valid) == 0:
        print("客観データがないため、乖離分析をスキップします。")
        return
    
    df_valid['discrepancy'] = df_valid['objective_fatigue'] - df_valid['subjective_fatigue']
    
    # 統計情報
    mean_discrepancy = df_valid['discrepancy'].mean()
    std_discrepancy = df_valid['discrepancy'].std()
    
    # 過小評価ケース (客観が主観より2以上高い、つまり疲れているのに疲れていないと感じている)
    underestimate_cases = df_valid[df_valid['discrepancy'] > 2.0]
    
    print("\n" + "="*60)
    print("【主観と客観の乖離分析】")
    print("="*60)
    print(f"分析対象データ数: {len(df_valid)}件")
    print(f"平均乖離: {mean_discrepancy:.2f} (正 = 客観 > 主観)")
    print(f"標準偏差: {std_discrepancy:.2f}")
    print(f"\n「頑張りすぎ」検出:")
    print(f"  - 客観が主観より2以上高いケース: {len(underestimate_cases)}件 ({len(underestimate_cases)/len(df_valid)*100:.1f}%)")
    
    if len(underestimate_cases) > 0:
        print(f"\n【危険な「自覚なき疲労」の例】")
        print(underestimate_cases[['Timestamp', 'user_id', 'subjective_fatigue', 
                                   'objective_fatigue', 'discrepancy']].head(5))
    #TODO! これをChatGPT APIに渡してアドバイス生成とかに活用する？
    # --- ChatGPT APIに渡すと効果的な情報をtxtで保存 ---
    # 1. 乖離統計
    with open(output_dir / "discrepancy_stats.txt", "w", encoding="utf-8") as f:
        f.write(f"分析対象データ数: {len(df_valid)}件\n")
        f.write(f"平均乖離: {mean_discrepancy:.2f} (正 = 客観 > 主観)\n")
        f.write(f"標準偏差: {std_discrepancy:.2f}\n")
        f.write(f"頑張りすぎ検出: 客観が主観より2以上高いケース: {len(underestimate_cases)}件 ({len(underestimate_cases)/len(df_valid)*100:.1f}%)\n")
    # 2. 乖離の大きい例
    if len(underestimate_cases) > 0:
        underestimate_cases.head(10).to_csv(output_dir / "danger_underestimate_cases.txt", sep="\t", index=False)
    # 3. 全データ（主観・客観・乖離）
    df_valid.to_csv(output_dir / "all_fatigue_discrepancy.txt", sep="\t", index=False)


# Placeholder for ChatGPT API integration
#TODO: komiyaくん実装お願いします。
def generate_advice(subjective: float, objective: float) -> str:
    """
    Placeholder for ChatGPT API integration to generate advice.

    Args:
        subjective: 主観的疲れ度
        objective: 客観的疲れ度

    Returns:
        Placeholder message.
    """
    return "[ChatGPT API integration required here]"


# Placeholder for Slack Bot notification
def send_slack_message(message: str):
    """
    Placeholder for Slack Bot notification.
    #TODO: komiyaくん実装お願いします。

    Args:
        message: 送信するメッセージ
    """
    print("[Slack Bot integration required here]")


def main():
    """
    テスト
    """
    print("="*60)
    print("Kyu (休) MVP - Fatigue Prediction System")
    print("="*60)
    
    # 1. データ読み込み
    csv_path = '../response.csv'  # 一つ上のディレクトリをデフォルトに設定してます
    if not Path(csv_path).exists():
        csv_path = 'response.csv'  # 同じディレクトリも確認
    
    print(f"\n[1] データ読み込み: {csv_path}")
    df = load_data(csv_path)
    print(f"  - 読み込み件数: {len(df)}件")
    print(f"  - 期間: {df['Timestamp'].min()} ~ {df['Timestamp'].max()}")
    print(f"  - ユーザー数: {df['user_id'].nunique()}名")
    
    # 2. 顔解析 (OpenFaceモック)
    print("\n[2] 顔写真解析 (OpenFace)")
    analyzer = FaceAnalyzer()
    df = analyze_faces(df, analyzer)
    photo_count = df['objective_fatigue'].notna().sum()
    print(f"  - 解析成功: {photo_count}件 / {len(df)}件")
    
    # 3. 乖離分析
    analyze_discrepancy(df)
    
    # 4. GPRモデル学習
    print("\n[3] ガウス過程回帰モデル学習")
    
    # 特定ユーザーで分析 (データが多いユーザーを選択)
    target_user = df['user_id'].value_counts().idxmax()
    df_user = df[df['user_id'] == target_user].copy()
    print(f"  - 対象ユーザー: {target_user} ({len(df_user)}件)")
    
    # モデル1: 主観のみ
    predictor_normal = FatiguePredictor(use_objective_correction=False)
    predictor_normal.fit(
        df_user['Timestamp'],
        df_user['subjective_fatigue'].values
    )
    print("  ✓ 主観のみモデル学習完了")
    
    # モデル2: 客観補正あり
    predictor_corrected = FatiguePredictor(
        use_objective_correction=True,
        correction_threshold=2.0,
        correction_weight=0.7
    )
    predictor_corrected.fit(
        df_user['Timestamp'],
        df_user['subjective_fatigue'].values,
        df_user['objective_fatigue'].values
    )
    print("  ✓ 客観補正モデル学習完了")
    
    # 5. 不確かさ分析
    print("\n[4] データ不足時間帯の検出")
    uncertain_hours = predictor_normal.get_high_uncertainty_periods(threshold_std=0.8)
    if len(uncertain_hours) > 0:
        print(f"  - 予測が不安定な時間帯: {len(uncertain_hours)}箇所")
        print(f"  - 具体例: {uncertain_hours[:5]}")
        print("  → これらの時間帯に通知を送ってデータ収集を促すべき")
    else:
        print("  - 全時間帯でデータが十分です")
    
    # 6. 可視化
    print("\n[5] タイムライン可視化")
    plot_fatigue_timeline(df, predictor_normal, predictor_corrected, user_id=target_user)
    
    # 7. サンプル予測
    print("\n[6] 予測例")
    test_hours = [9.0, 15.0, 21.0]  # 9時、15時、21時
    for hour in test_hours:
        mean_n, std_n = predictor_normal.predict(hours=np.array([hour]))
        mean_c, std_c = predictor_corrected.predict(hours=np.array([hour]))
        
        print(f"\n  {int(hour)}:00の予測疲労度")
        print(f"    - 主観のみ:     {mean_n[0]:.2f} ± {std_n[0]:.2f}")
        print(f"    - 客観補正あり: {mean_c[0]:.2f} ± {std_c[0]:.2f}")
    
    # 8. ChatGPT APIとSlack通知
    print("\n[7] ChatGPT APIとSlack通知")
    for idx, row in df.iterrows():
        subjective = row['subjective_fatigue']
        objective = row['objective_fatigue']
        advice = generate_advice(subjective, objective)
        message = (
            f"ユーザー: {row['user_id']}\n"
            f"主観的疲れ度: {subjective}\n"
            f"客観的疲れ度: {objective}\n"
            f"アドバイス: {advice}"
        )
        send_slack_message(message)
    
    print("\n" + "="*60)#しきりデザイン
    print("分析完了!")
    print("="*60)


if __name__ == "__main__":
    main()
