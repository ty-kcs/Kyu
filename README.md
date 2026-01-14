# Kyu (休) - ウェルビーイング支援アプリ MVP

## 概要
「主観的疲労度」と「客観的表情分析」のズレを可視化し、日本人に多い「まだ頑張れる」を検知するシステム。

## 疲労度の周期性について
「1日（24時間）単位で疲労度の周期が繰り返される」ことを前提としています。
つまり、（少なくとも現状の実装では、）疲労度の予測や分析は「時刻（0.0〜24.0）」のみを特徴量とし、日付や曜日などは考慮していません。
（例えば「15時」の疲労度はどの日でも同じように扱われます。）

## システム構成

### 1. face_analyzer.py
- OpenFaceのAction Units (AU04, AU43) から客観的疲労度を算出
- **AU04** (Brow Lowerer): ストレス・緊張指標
- **AU43** (Eyes Closed): 眠気指標
- 計算式: `Score_obj = 1 + min(6, (w1*AU04 + w2*AU43) * scaling_factor)`(この式はまじチャットGPT作成です。追々ちゃんとしたシステムにします。)

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

### 不確かさの活用
GPRの標準偏差が高い時間帯 = データ不足 → 通知を送るべき時間帯として活用。

## MVP検証ポイント
1. ✅ 主観と客観のズレが定量化できるか
2. ✅ GPRで疲労パターンを学習できるか
3. ✅ データ不足時間帯を検出できるか
4. → 次: ChatGPT APIとの連携で「罪悪感ゼロの休息提案」生成
