## 1 `01_mnist_visual_convergence.ipynb`

**テーマ: "Purification of Chaos" (視覚的な収束)**

このNotebookのゴールは、ノイズに埋もれた「3」が、思考ステップを経るごとに鮮明な「3」の概念（不動点）に変わっていく様をアニメーション的に見せることです。

### 構成手順

1. **Setup & Config**

      * `CnnAdapterConfig` を使用（MNISTは画像データのため）。
      * `GaussianNoiseAugmenterConfig` を設定し、推論時にあえて強いノイズ（max_noise_std=0.5以上）を加える設定にする。
      * `max_steps=10` 程度に設定し、思考の刻みを細かくする。

2. **Quick Training (Self-contained)**

      * Notebook内で簡易学習を実行（Stage 0 → Stage 1 → Stage 2 → Stage 3、各1-2 epoch）。
      * **Note:** 軌跡（SantanaLog）は `SamathaEngine._run_vicara_loop` 内で自動的に記録されるため、特別な設定は不要。

3. **Inference & Trajectory Extraction**

      * ノイズを加えた画像を `system(x)` に通す。
      * 戻り値の `result.santana` (SantanaLog) にアクセスする。
      * `santana.to_tensor()` で `(Steps, Batch, Dim)` のテンソルを取得。

4. **Visualization (The "Wow" Factor)**

      * **Cell 1: Image Evolution**
          * `samatha_recon_head` を使い、各ステップの潜在状態 $S_t$ を画像にデコードして並べる。
          * 左端（ノイズ画像）→ 中央（徐々に形になる）→ 右端（きれいな数字）という遷移を表示。
      * **Cell 2: PCA/t-SNE Animation**
          * 潜在空間を2次元に圧縮し、点が「アトラクタ（正解の数字のクラスタ中心）」に吸い寄せられていく軌跡をプロットする。

-----

## 2 `02_trust_score_explained.ipynb`

**テーマ: "The Trust Score" (境界と未知の可視化)**

このNotebookのゴールは、「自信満々に間違える（Softmax）」と「自信がないことを自覚する（Vipassana）」の違いをヒートマップで対比させることです。

### 構成手順

1. **Data Generation**

      * `sklearn.datasets.make_moons` で「三日月形」の2クラス分類データを作成。
      * ノイズを少し加え、クラス間の境界を曖昧にする。

2. **Config Strategy**

      * `MlpAdapterConfig` (input\_dim=2) を使用。
      * **比較対象:** 通常のMLP（Softmax出力）と、Satipatthana（ConditionalDecoder）を用意。

3. **Training (Stage 2 Focus, Self-contained)**

      * Notebook内で4-Stage Curriculum学習を実行。
      * Stage 2（Vipassana Training）を重点的に行う。
      * **重要:** `Drunk Path` (内部摂動) と `Mismatch Path` (論理矛盾) のデータ生成が機能していることを確認する。

4. **Visualization (Heatmap Comparison)**

      * $-2.0 \sim 2.0$ の範囲でグリッド状にテスト点を作成（データの存在しない空白地帯を含む）。
      * **Plot A (Softmax):** 空白地帯でも「99% クラスA」といった高い確率が出ることを示す（過信）。
      * **Plot B (Trust Score):** データの存在する三日月の上だけが高く、**空白地帯は「0.0（低信頼）」で真っ黒になる**ことを示す。
      * **解説:** これこそが「知っていること」と「知らないこと」の区別であることを強調。

-----

## 3 `03_fraud_detection_tutorial.ipynb`

**テーマ: "4-Stage Curriculum Implementation" (実装ガイド)**

このNotebookのゴールは、開発者が「自分のデータでどう学習させればいいか」を理解するための、コピー＆ペースト可能なレシピブックです。

### 構成手順

1. **Dataset Preparation**

      * KaggleのCredit Card Fraud Detectionを使用（実践的な不均衡データ）。
      * `!kaggle datasets download -d mlg-ulb/creditcardfraud` でダウンロード。
      * **Note:** 初回は `~/.kaggle/kaggle.json` の設定が必要（[Kaggle API Token](https://www.kaggle.com/docs/api)）。
      * `torch.utils.data.Dataset` を継承し、`{"x": ..., "y": ...}` を返すクラスを定義。

2. **System Config (Best Practice)**

      * `MlpAdapterConfig` を使用。
      * `use_label_guidance=True` を設定し、Stage 1でラベル情報を利用して収束を助ける設定にする。

3. **The Trainer Loop (Step-by-Step)**

      * `SatipatthanaTrainer` をインスタンス化。
      * **Stage 0 (Pre-training):** `trainer.train_stage(TrainingStage.ADAPTER_PRETRAINING, num_epochs=5)` を実行し、Reconstruction Lossが下がる様子をプロット。
      * **Stage 1 (Samatha):** `trainer.train_stage(TrainingStage.SAMATHA_TRAINING, num_epochs=10)` を実行。Stability Lossが低下し、不動点が形成される過程をログで確認。
      * **Stage 2 (Vipassana):** `trainer.train_stage(TrainingStage.VIPASSANA_TRAINING, num_epochs=5)` を実行。BCE Lossの推移を確認。
      * **Stage 3 (Decoder):** `trainer.train_stage(TrainingStage.DECODER_FINETUNING, num_epochs=5)` で最終タスクへのファインチューニング。
      * **または:** `trainer.run_curriculum(stage0_epochs=5, stage1_epochs=10, stage2_epochs=5, stage3_epochs=5)` で一括実行。

4. **Evaluation**

      * 推論を実行し、Trust Scoreが低いサンプル（Vipassanaが警告を出したサンプル）に実際にFraudが多い、あるいは「判定困難なケース」が含まれているかを確認する分析を行う。

-----

## 4 `04_timeseries_anomaly_xai.ipynb`

**テーマ: "Explainable Anomaly Detection" (応用と説明)**

このNotebookのゴールは、時系列データにおいて「どこで異常が起きたか」だけでなく「なぜ異常と判断したか（思考の乱れ）」を示すことです。

### 構成手順

1. **Config for Time Series**

      * `LstmAdapterConfig` (seq\_len=50など) を使用。
      * `use_label_guidance=False` (教師なし学習モード) に設定。異常検知は「正常からの逸脱」を見るため、教師なしが基本となる。

2. **Training (Reconstruction & Stability, Self-contained)**

      * 合成時系列データ（正弦波 + ノイズ）を使用。外部データダウンロード不要。
      * 正常データのみで学習を行う。
      * Stage 1までで「正常な波形ならスムーズに収束するSamatha」を育てる。
      * Stage 2で「スムーズな収束＝高信頼、乱れた収束＝低信頼」というメタ認知をVipassanaに教える。

3. **Inference on Anomaly Data**

      * 異常（スパイクやドリフト）を含むテストデータを入力。

4. **XAI Visualization**

      * **上段プロット:** 元の時系列波形（異常区間をハイライト）。
      * **下段プロット:** タイムステップごとの **Trust Score ($\alpha$)** の推移。
      * **洞察:** 異常が発生した瞬間に、再構成誤差が出るより先に **「思考が収束しなくなる（Trust Scoreが急落する）」** という予兆検知的な挙動を示すことができればベストです。

-----

### リードアーキテクトからのアドバイス

作成にあたっては、各Notebookの冒頭に、以下の**「お約束」**を記載してください。

```python
# Satipatthana Framework Setup
# 1. ログ設定は必ず最初に行う
from satipatthana.utils.logger import setup_logging
setup_logging()

# 2. Configは個別にimport（型安全なdataclass）
from satipatthana.configs.system import SystemConfig, SamathaConfig, VipassanaEngineConfig
from satipatthana.configs.adapters import CnnAdapterConfig, MlpAdapterConfig, LstmAdapterConfig
from satipatthana.configs.decoders import ConditionalDecoderConfig, CnnDecoderConfig
# ... 必要なConfigを追加

# 3. Factory関数を使う場合
from satipatthana.configs.factory import create_adapter_config, create_decoder_config
```

これで、ユーザーは「作法」を自然と学ぶことができます。

### 設計方針

* **Self-contained**: 各Notebookは外部データやモデルファイルに依存しない。学習→可視化が1つのNotebook内で完結する。
* **Quick Training**: デモ用に軽量な学習設定（少ないepoch数、小さなデータセット）を使用。
* **Progressive Complexity**: 01 → 04 に向けて徐々に複雑なユースケースを扱う。

-----

### 既知の問題と対処法

実装時に発生した問題と対処法を [docs/issues/001_training_issues_v4.md](../issues/001_training_issues_v4.md) にまとめています。

#### 重要: 学習時の設定

1. **Vitakka gate_threshold**

   学習時は `gate_threshold=-1.0` に設定してゲートを常に開放する必要があります。
   デフォルト値（0.6）のままだと、学習初期にゲートが閉じて入力情報が消失します。

   ```python
   vitakka_config = StandardVitakkaConfig(
       dim=64,
       n_probes=10,
       gate_threshold=-1.0,  # 学習中は常にゲートを開く
   )
   ```

2. **SatipatthanaTrainerの使用**

   `SatipatthanaTrainer.train_stage()` を使用すると、以下が自動的に処理されます：
   * Stage間でのオプティマイザリセット
   * Stage 2開始前のVipassanaネットワーク初期化

   手動で学習ループを書く場合は、これらを明示的に処理する必要があります。
