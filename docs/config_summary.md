# Samadhiモデル設定パラメータ概要

`SamadhiModel`とその学習ループにおける`config`辞書の中心的な役割について、以下の通り、`src/model`、`src/components`、`src/train`で使用されているすべてのパラメータを包括的に分類します。

### 1. モデルアーキテクチャ (`src/model` & `src/components`)

これらのパラメータは、コアモデルコンポーネント（VitakkaおよびVicara）の構造と動作を定義します。

| キー | デフォルト値 | 必須 | ファイル | 説明 |
| :--- | :--- | :--- | :--- | :--- |
| **`dim`** | - | **はい** | `samadhi.py`, `vitakka.py`, `vicara.py` | 潜在状態ベクトル（$S$）の次元。 |
| **`input_dim`** | - | **はい** | `mlp_samadhi.py`, `lstm_samadhi.py`, `transformer_samadhi.py` | 入力データの次元 (MLPの場合: 特徴量数、LSTM/Transformerの場合: 各タイムステップの特徴量数)。 |
| **`seq_len`** | - | **はい** | `lstm_samadhi.py`, `transformer_samadhi.py` | *(LSTM/Transformerのみ)* 時系列シーケンスの長さ (スライディングウィンドウサイズ)。 |
| **`n_probes`** | - | **はい** | `vitakka.py`, `vicara.py` | 検索対象となる「概念」またはプローブの数。 |
| **`vicara_type`** | `"standard"` | いいえ | `samadhi.py` | リファインメントモジュールのタイプ: `"standard"`、`"weighted"`、または`"probe_specific"`。 |
| **`channels`** | `3` | いいえ | `conv_samadhi.py` | *(ConvSamadhiのみ)* 入力画像のチャンネル数。 |
| **`img_size`** | `32` | いいえ | `conv_samadhi.py` | *(ConvSamadhiのみ)* 入力画像のサイズ（正方形を想定）。 |
| **`labels`** | `[]` | いいえ | `samadhi.py` | プローブインデックスにマッピングされる文字列ラベルのリスト（人間が読めるロギング用）。 |
| **`adapter_hidden_dim`** | `256` (MLP), `128` (LSTM/Transformer) | いいえ | `mlp_samadhi.py`, `lstm_samadhi.py`, `transformer_samadhi.py` | アダプター（エンコーダー）内の隠れ層の次元。 |
| **`lstm_layers`** | `1` | いいえ | `lstm_samadhi.py` | *(LstmSamadhiのみ)* LSTM層の数。 |
| **`transformer_layers`** | `2` | いいえ | `transformer_samadhi.py` | *(TransformerSamadhiのみ)* Transformer Encoder層の数。 |
| **`transformer_heads`** | `4` | いいえ | `transformer_samadhi.py` | *(TransformerSamadhiのみ)* Multi-head Attentionのヘッド数。 |


### Vitakka (Search)
| Key | Symbol | Recommended Value | Description |
| :--- | :--- | :--- | :--- |
| **`gate_threshold`** | $\theta$ | 0.3 - 0.5 | 妄想（ノイズ）を弾く強度。高いほど厳格。 |
| **`softmax_temp`** | $\tau$ | 0.1 - 0.2 | 低いほど「一境性（単一テーマ）」を選び取る。 |
| **`mix_alpha`** | $\alpha_{mix}$ | 0.5 | 初期状態生成時の入力とプローブの混合比率。 |
| **`training_attention_mode`** | - | `"soft"` | 学習時（`model.train()`）にVitakkaが使用するアテンションモード。 |
| **`prediction_attention_mode`** | - | `"hard"` | 推論時（`model.eval()`）にVitakkaが使用するアテンションモード。 |

### 3. リファインメントダイナミクス (`src/components/vicara.py`)

これらのパラメータは、「Vicara」（リファインメント）ループを制御し、時間が経つにつれて状態が精製されます。

| キー | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`refine_steps`** | - | **はい** | フォワードパスごとに実行される再帰的精製ステップ（$\Phi$）の数。 |
| **`inertia`** | `0.7` | いいえ | 状態更新のための慣性（EMA）: $S_{new} = \alpha \cdot S_{old} + (1-\alpha) \cdot Residual$。値が高いほど、変化は滑らかで遅くなります。 |

### 4. トレーニングハイパーパラメータ (`src/train`)

これらのパラメータは、トレーニング中（教師ありまたは教師なし）の異なる損失コンポーネントをスケーリングします。

| キー | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`stability_coeff`**| `0.01` | いいえ | **安定性損失**の重み（状態の動き$||S_t - S_{t-1}||$にペナルティを与えます）。 |
| **`entropy_coeff`** | `0.1` | いいえ | **エントロピー損失**の重み（プローブ選択における躊躇にペナルティを与えます）。 |
| **`balance_coeff`** | `0.001` | いいえ | **ロードバランス損失**の重み（均一な使用を強制することで「プローブ崩壊」を防ぎます）。 |
| **`anomaly_margin`** | `5.0` | いいえ | *(AnomalyTrainerのみ)* 異常データの復元誤差がこれ以下の場合にペナルティを与えるマージン。高いほど厳しくなります。 |
| **`anomaly_weight`** | `1.0` | いいえ | *(AnomalyTrainerのみ)* 異常データに対するマージン損失の重み。高いほど異常を遠ざける力が強まります。 |

### 必須キーの要約

モデルとトレーナーを正常に初期化するには、`config`辞書には少なくとも以下が含まれている必要があります。
```python
config = {
    "dim": ...,             # int (例: 64)
    "input_dim": ...,       # int (例: 29 for tabular, features per timestep for sequence)
    "seq_len": ...,         # int (例: 30 for sequence models)
    "n_probes": ...,        # int (例: 10)
    "gate_threshold": ...,  # float (例: 0.5)
    "refine_steps": ...,    # int (例: 5)
}
