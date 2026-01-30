"""
Main Script
データロード、FaceAnalyzer + FatiguePredictor の統合実行、可視化。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import requests
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path

from face_analyzer import FaceAnalyzer
from gpr_model import FatiguePredictor
from openai import OpenAI
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from datetime import datetime,timezone,timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv(override=True) 

#環境変数
spreadsheet_id = os.getenv("SPREADSHEET_ID")
gid = os.getenv("GID")
openai_api_key= os.getenv("OPENAI_API_KEY")
slack_api_token= os.getenv("SLACK_API_TOKEN")
channel_id= os.getenv("SLACK_CHANNEL_ID")
CHANNEL = '#all-果実とナッツ'

MODEL_PATH = os.getenv("FATIGUE_MODEL_PATH", "fatigue_model.pkl")
OPENFACE_BIN = os.getenv("OPENFACE_BIN", "PATH TO FeatureExtraction.exe")
TEMP_DIR = os.getenv("OPENFACE_TEMP_DIR", "temp_inference")
FACE_CACHE_DIR = os.getenv("FACE_CACHE_DIR", "face_cache")

from matplotlib import rcParams
from matplotlib import font_manager

# 日本語フォント設定
rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
rcParams['font.family'] = 'sans-serif'
rcParams['axes.unicode_minus'] = False

# フォントが正しく設定されているか確認
print("Available fonts:", font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))
print("Current font settings:", rcParams['font.sans-serif'])

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
    print(df)

    # 主観的疲労度の欠損値を除外
    df = df.dropna(subset=['subjective_fatigue'])
    
    # ソート
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    return df


def compute_score_from_probs(probabilities: Optional[list], label: Optional[str]) -> Optional[float]:
    """
    FaceAnalyzerの出力（確率配列があれば）を使って数値スコアを返す。
    - 確率配列がある場合：各クラスの代表値(低=1.5, 中=4.0, 高=6.5)で期待値を計算
    - なければ従来のラベル固定値にフォールバック
    """
    class_means = [1.5, 4.0, 6.5]  # クラス順は FaceAnalyzerのモデルクラス順に合わせる前提
    label_map = {"疲れ度:低 (1-2)": 1.5, "疲れ度:中 (3-5)": 4.0, "疲れ度:高 (6-7)": 6.5}
    if probabilities is not None:
        probs = list(probabilities)
        # 長さが合わない場合
        if len(probs) == len(class_means):
            return float(sum(p * m for p, m in zip(probs, class_means)))
    return label_map.get(label)


def analyze_faces(df: pd.DataFrame, analyzer: FaceAnalyzer) -> pd.DataFrame:
    """
    DataFrameの各行に対して顔解析を実行し、客観疲労度ラベル/信頼度/スコアを追加。
    
    Args:
        df: 元のDataFrame
        analyzer: FaceAnalyzerインスタンス
    
    Returns:
        objective_fatigue_label/confidence/scoreが追加されたDataFrame
    """
    df = df.copy()
    objective_labels = []
    objective_confidences = []
    objective_scores = []
    
    for idx, row in df.iterrows():
        photo_link = row['photo_link']
        result = analyzer.analyze_image(photo_link)
        if result is None:
            objective_labels.append(None)
            objective_confidences.append(None)
            objective_scores.append(None)
            continue

        # FaceAnalyzer -> (label, confidence, probabilities)
        if len(result) == 3:
            label, confidence, probabilities = result
        else:
            label, confidence = result
            probabilities = None

        objective_labels.append(label)
        objective_confidences.append(confidence)
        objective_scores.append(compute_score_from_probs(probabilities, label))
    
    df['objective_fatigue_label'] = objective_labels
    df['objective_fatigue_confidence'] = objective_confidences
    df['objective_fatigue_score'] = objective_scores
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
    df_with_obj = df[df['objective_fatigue_score'].notna()]
    ax2.scatter(df_with_obj['hour'], df_with_obj['objective_fatigue_score'], 
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
    return out_png


def analyze_discrepancy(df: pd.DataFrame, user_id: Optional[str] = None):
    """
    主観と客観のズレ(乖離)を分析してレポート。
    
    Args:
        df: 元のDataFrame
        user_id: 特定ユーザーのみ分析する場合
    """
    # user_idでフィルタリング
    if user_id:
        df = df[df['user_id'] == user_id].copy()

    # objective_fatigue_scoreが存在する行のみ抽出
    df_valid = df[df['objective_fatigue_score'].notna()].copy()
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    if len(df_valid) == 0:
        print("客観データがないため、乖離分析をスキップします。")
        return

    df_valid['discrepancy'] = df_valid['objective_fatigue_score'] - df_valid['subjective_fatigue']

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
                       'objective_fatigue_score', 'discrepancy']].head(5))
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


# ChatGPT API integration
def generate_advice(hours: list) -> str:
    """
    アドバイスを生成
    
    hours list(int が2つ): 主観と客観の差が大きいとされた時間帯上位2つのリスト
    """
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.8,
        messages=[
            {
                "role": "user",
                "content": f"""
                あなたはユーザーの生活に関するアドバイザーです。以下のデータを元に感情のあるアドバイスをして下さい。
                #データ
                主観的疲労度と客観的疲労度の差が多い時間帯は、{hours[0]}時や{hours[1]}時

                #メッセージの構成
                メッセージは、挨拶の後、
                時間を大まかに特定できる表現  に主観的な疲労度と客観的な疲労度の差がある可能性があります。
                のような文で開始して下さい。具体的な時刻は書かないて下さい。午後や夜だけではなく細かく述べて下さい。
                次に、具体的で小さな行動の提案を含めて下さい。

                #注意事項
                200文字以内で回答して下さい。
                断定を避けて下さい。
                具体的な時刻は書かないて下さい。
                """
            }
        ]
        )
    advise= completion.choices[0].message.content
    return advise


# Placeholder for Slack Bot notification
def send_slack_message(message: str, slack_api_token=slack_api_token):
    """
    メッセージをslackに送信
    Args:
        message: 送信するメッセージ
    """
    url = "https://slack.com/api/chat.postMessage"
    headers = {"Authorization": "Bearer "+slack_api_token}
    data  = {
    'channel': CHANNEL,
    'text': message
    }
    r = requests.post(url, headers=headers, data=data)
    print("return ", r.json())

def send_png_to_slack(file_path:str):
    """pngファイルをslackに送信

    Args:
        file_path (str): pngファイルのパス
    """
    client = WebClient(token=slack_api_token)
    try:
        response = client.files_upload_v2(
            channel=channel_id, 
            file=file_path,
            )
        assert response["file"]  # the uploaded file
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")

def slack_scheduled_message(unix_time:int, message:str):
    """指定のUNIX時間にslackにメッセージを送信する

    Args:
        unix_time (int): UNIX時間
        message(str):送信するメッセージ
    """
    url = "https://slack.com/api/chat.scheduleMessage"
    headers = {"Authorization": "Bearer "+slack_api_token}
    data  = {
    'channel': CHANNEL,
    'post_at':unix_time,
    'text': message,
    }
    r = requests.post(url, headers=headers, data=data)
    print("scheduled message return ", r.json())

def next_time_jst_and_unix(time: np):
    """
    gpr関数で算出された時刻 12.0などから次のその時刻のUNIX時間を返す

    time: 12.0など
    戻り: (next_dt_jst, unix_seconds)
    """
    JST = ZoneInfo("Asia/Tokyo")
    now = datetime.now(JST)

    # "a:b" をパース
    parts = str(time).strip().split(".")
    print(parts)

    hour = int(parts[0])
    minute= int((float(time) - hour)*60) # 小数点0.25 が15分

    # 今日のその時刻（秒は0固定）
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # すでに過ぎていれば明日へ
    if candidate <= now:
        candidate += timedelta(days=1)

    unix_seconds = int(candidate.timestamp())  # UNIX時間（秒）
    return candidate, unix_seconds



def google_sheet_to_csv(spreadsheet_id:str, gid:str):
    csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"
    df = pd.read_csv(csv_url)
    new_cols={}
    for c in df.columns:
        if c.startswith('現在の「疲れ具合」を教えてください'):
            new_cols[c] = "subjective_fatigue"
        if c.startswith('今の顔画像をアップロードして'):
            new_cols[c] = "photo_link"
        if c == "ユーザー(識別子)":
            new_cols[c] = "user_id"

    df = df.rename(columns=new_cols)
    df.to_csv("../response.csv", index=False)

def main(user_id):
    """
    テスト
    """
    print("="*60)
    print("Kyu (休)")
    print("="*60)

    # google formの回答読み込み
    google_sheet_to_csv(spreadsheet_id,gid)

    # 1. データ読み込み
    csv_path = '../response.csv'  # 一つ上のディレクトリをデフォルトに設定してます
    if not Path(csv_path).exists():
        csv_path = 'response.csv'  # 同じディレクトリも確認
    
    print(f"\n[1] データ読み込み: {csv_path}")
    df = load_data(csv_path)
    print(f"  - 読み込み件数: {len(df)}件")
    print(f"  - 期間: {df['Timestamp'].min()} ~ {df['Timestamp'].max()}")
    print(f"  - ユーザー数: {df['user_id'].nunique()}名")
    
    # 2. 顔解析 
    print("\n[2] 顔写真解析 (OpenFace)")
    analyzer = FaceAnalyzer(
        model_path=MODEL_PATH,
        openface_bin=OPENFACE_BIN,
        temp_dir=TEMP_DIR,
        cache_dir=FACE_CACHE_DIR,
    )
    df = analyze_faces(df, analyzer)
    photo_count = df['objective_fatigue_score'].notna().sum()
    print(f"  - 解析成功: {photo_count}件 / {len(df)}件")
    
    # 3. 乖離分析
    analyze_discrepancy(df, user_id=user_id)
    
    # 4. GPRモデル学習
    print("\n[3] ガウス過程回帰モデル学習")
    
    # 特定ユーザーで分析 
    target_user = user_id
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
        df_user['objective_fatigue_score'].values
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
    png_file_path= plot_fatigue_timeline(df, predictor_normal, predictor_corrected, user_id=target_user)
    
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

    #df_user は1ユーザーの全データ
    df_user["hour"] = df_user["Timestamp"].dt.hour
    df_user["diff"]= df_user["objective_fatigue_score"]- df_user["subjective_fatigue"]
    median_by_hour= df_user.groupby("hour")["diff"].median()
    print("各時刻の客観-主観の中央値",median_by_hour)
    top2_hours = median_by_hour.dropna().sort_values(ascending=False).head(2).index.tolist()
    print(top2_hours)

    advise= generate_advice(top2_hours)
    print(advise)
    send_slack_message(advise, slack_api_token)
    # png_file_path= "/Users/komiya/Downloads/Slack_icon_2019.svg.png"
    send_png_to_slack(file_path= png_file_path)

    # 予約投稿
    peak_time = predictor_corrected.get_peak_time()#ユーザーが単一の場合
    print(f"予測疲労度が最も高い時間: {peak_time}")
    jst_timestamp, unix_time= next_time_jst_and_unix(peak_time)
    print(jst_timestamp, unix_time)
    
    #デモ用の時刻指定
    # unix_time= 1769066220		
    
    # 予約投稿
    message = """
    顔画像から疲労度を推定するため、こちらから顔画像をアップロードお願いします。
    https://docs.google.com/forms/d/e/1FAIpQLSeVNL2F26IbOEw55kCpjlcZLTGDdXC-goEN3kioUgZT-ywbEA/viewform?usp=header
    """
    slack_scheduled_message(unix_time, message)
    
    print("\n" + "="*60)#しきりデザイン
    print("分析完了!")
    print("="*60)


if __name__ == "__main__":
    user_id = input("Google Formに記入したユーザー名を入力してください: ")
    main(user_id)
