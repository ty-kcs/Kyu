# Kyu (休) - ウェルビーイング支援アプリ MVP

## 概要
現状では、「主観的疲労度」と「客観的表情分析」のズレを可視化し、日本人に多い「まだ頑張れる」を検知するシステム。

## 疲労度の周期性について
「1日（24時間）単位で疲労度の周期が繰り返される」ことを前提としています。
つまり、（少なくとも現状の実装では、）疲労度の予測や分析は「時刻（0.0〜24.0）」のみを特徴量とし、日付や曜日などは考慮していません。
（例えば「15時」の疲労度はどの日でも同じように扱われます。）

## システム構成

### 1. face_analyzer.py
- OpenFaceのAction Units から客観的疲労度を算出

### 2. gpr_model.py
- ガウス過程回帰 (GPR) による24時間疲労度予測
- **特徴量**: 時刻 (0.0 ~ 24.0)
- **ターゲット**: 疲労度 (1-7)
- **客観補正機能**: 主観と客観の乖離が大きい場合、客観値を重視

### 3. main.py
- データ統合・分析・可視化のメインスクリプト
- 主観/客観の乖離分析
- 疲労度タイムライン生成

## セットアップ

### 環境構築 (仮想環境)
1. **Pythonのインストール確認**:
   - Python 3.8以上がインストールされていることを確認してください。
   ```bash
   python --version
   ```

2. **仮想環境の作成**:
   ```bash
   python -m venv venv
   ```

3. **仮想環境のアクティブ化**:
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - Windows:
     ```bash
     venv\Scripts\activate
     ```

4. **依存関係のインストール**:
   ```bash
   pip install -r requirements.txt
   ```

### メイン実行
```bash
# 実行
python main.py
```

## データ形式 (response.csv)
一個上のディレクトリにある想定。google formのスプレッドシートと少しカラム名変えたので注意です。（スラックボットであのまま収集する場合、前処理としてcsv変換・カラム名変えるスクリプト必要。）
```csv
Timestamp,user_id,subjective_fatigue,photo_link
2025-01-01 09:00:00,user1,2,https://drive.google.com/...
```

## 出力
- `fatigue_timeline_<user_id>.png`: 疲労度予測グラフ
- コンソール: 乖離分析、不確かさ検出結果

### 客観補正ロジック
日本人は「まだ大丈夫」と主観を過小評価しがち。客観データ (顔のAU) が主観より高い場合、実際はもっと疲れていると判断。

```python
# 客観補正なし
predictor_normal = FatiguePredictor(use_objective_correction=False)

# 客観補正あり (客観70% + 主観30%)
predictor_corrected = FatiguePredictor(
    use_objective_correction=True,
    correction_threshold=2.0,
    correction_weight=0.7
)
```

### OpenFace導入

#### OpenFace実行方法
- このレポジトリをクローン
- OpenFaceのモデルをダウンロード(OpenFace/download_models.ps1を実行)
- ライブラリをダウンロード(OpenFace/download_libraries.ps1)
- OpenFaceをビルド(Visual StudioでOpenFace/OpenFace.slnをBuild Solution)
- predict_fatigue.pyの10行目(OPENFACE_BIN)をビルドしたexecutableに入れ替える
- 以下のコマンドで実行(入力: 顔画像 出力:疲れ度(三段階: 低(1-2), 中(3-5), 高(6-7)) 信頼度: %)
  ```bash
  python predict_fatigue.py path/to/your/image.jpg
  ``` 
---
