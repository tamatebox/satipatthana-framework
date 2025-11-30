# Samadhiモデル設定パラメータ概要 (v3.1 Config System)

このドキュメントでは、`SamadhiFramework`のConfigパラメータについて説明します。v3.1以降、設定は従来の辞書ベース（`Dict[str, Any]`）から、より型安全で構造化されたDataclassベースのシステムに移行しました。これにより、各パラメータは特定のConfigオブジェクトの属性としてアクセスされ、コードの可読性と保守性が向上しています。

Configのルートは`src/configs/main.py`で定義されている`SamadhiConfig`オブジェクトです。

## 1. ルート設定: `SamadhiConfig` (`src/configs/main.py`)

`SamadhiConfig`は、モデル全体のグローバルパラメータと、各主要コンポーネントのサブ設定を保持します。

| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`dim`** | `64` | いいえ | 潜在状態ベクトル（$S$）の次元。 |
| **`seed`** | `42` | いいえ | モデル初期化のための乱数シード。 |
| **`labels`** | `[]` | いいえ | プローブインデックスにマッピングされる文字列ラベルのリスト（人間が読めるロギング用）。 |
| **`adapter`** | `MlpAdapterConfig` (default) | いいえ | アダプターコンポーネントの設定オブジェクト（`BaseAdapterConfig`のサブクラス）。 |
| **`vitakka`** | `StandardVitakkaConfig` (default) | いいえ | ヴィタッカーコンポーネントの設定オブジェクト（`BaseVitakkaConfig`のサブクラス）。 |
| **`vicara`** | `StandardVicaraConfig` (default) | いいえ | ヴィチャーラコンポーネントの設定オブジェクト（`BaseVicaraConfig`のサブクラス）。 |
| **`decoder`** | `ReconstructionDecoderConfig` (default) | いいえ | デコーダーコンポーネントの設定オブジェクト（`BaseDecoderConfig`のサブクラス）。 |
| **`objective`** | `ObjectiveConfig` (default) | いいえ | 学習目標の設定オブジェクト（`ObjectiveConfig`のインスタンス）。 |

## 2. アダプター設定 (`src/configs/adapters.py`)

入力データから潜在状態を生成するアダプターの動作を定義します。具体的なConfigクラスは`AdapterType`によって異なります。

### `BaseAdapterConfig` (共通属性)
| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`type`** | `AdapterType.MLP` | いいえ | アダプターのタイプ（`AdapterType` Enum）。 |
| **`dropout`** | `0.1` | いいえ | アダプター内のドロップアウト率。 |

### `MlpAdapterConfig` (タイプ: `AdapterType.MLP`)
| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`input_dim`** | `10` | いいえ | 入力データの次元（特徴量数）。 |
| **`adapter_hidden_dim`** | `256` | いいえ | 隠れ層の次元。 |

### `CnnAdapterConfig` (タイプ: `AdapterType.CNN`)
| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`channels`** | `3` | いいえ | 入力画像のチャンネル数。 |
| **`img_size`** | `32` | いいえ | 入力画像のサイズ（正方形を想定）。 |

### `LstmAdapterConfig` (タイプ: `AdapterType.LSTM`)
| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`input_dim`** | `10` | いいえ | 各タイムステップの特徴量数。 |
| **`seq_len`** | `50` | いいえ | 時系列シーケンスの長さ。 |
| **`adapter_hidden_dim`** | `128` | いいえ | 隠れ層の次元。 |
| **`lstm_layers`** | `1` | いいえ | LSTM層の数。 |

### `TransformerAdapterConfig` (タイプ: `AdapterType.TRANSFORMER`)
| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`input_dim`** | `10` | いいえ | 各タイムステップの特徴量数。 |
| **`seq_len`** | `50` | いいえ | 時系列シーケンスの長さ。 |
| **`adapter_hidden_dim`** | `128` | いいえ | 隠れ層の次元。 |
| **`transformer_layers`** | `2` | いいえ | Transformer Encoder層の数。 |
| **`transformer_heads`** | `4` | いいえ | Multi-head Attentionのヘッド数。 |

## 3. ヴィタッカー設定 (`src/configs/vitakka.py`)

初期状態（$S0$）の生成とプローブの管理を制御します。現在、`StandardVitakkaConfig`のみが提供されています。

### `StandardVitakkaConfig` (タイプ: デフォルト)
| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`n_probes`** | `10` | いいえ | 検索対象となる「概念」またはプローブの数。 |
| **`probe_trainable`** | `True` | いいえ | プローブが学習可能かどうか。 |
| **`gate_threshold`** | `0.6` | いいえ | 妄想（ノイズ）を弾く強度。高いほど厳格。 |
| **`softmax_temp`** | `0.2` | いいえ | ソフトマックスの温度パラメータ。低いほど「一境性（単一テーマ）」を選び取る。 |
| **`mix_alpha`** | `0.5` | いいえ | 初期状態生成時の入力とプローブの混合比率。 |
| **`training_attention_mode`** | "soft" | いいえ | 学習時（`model.train()`）にVitakkaが使用するアテンションモード。 |
| **`prediction_attention_mode`** | "hard" | いいえ | 推論時（`model.eval()`）にVitakkaが使用するアテンションモード。 |

## 4. ヴィチャーラ設定 (`src/configs/vicara.py`)

潜在状態の再帰的精製プロセスを制御します。具体的なConfigクラスは`VicaraType`によって異なります。

### `BaseVicaraConfig` (共通属性)
| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`type`** | `VicaraType.STANDARD` | いいえ | ヴィチャーラのタイプ（`VicaraType` Enum）。 |
| **`refine_steps`** | `5` | いいえ | フォワードパスごとに実行される再帰的精製ステップ（$\Phi$）の数。 |
| **`inertia`** | `0.7` | いいえ | 状態更新のための慣性（EMA）: $S_{new} = \alpha \cdot S_{old} + (1-\alpha) \cdot Residual$。値が高いほど、変化は滑らかで遅くなります。 |

### `StandardVicaraConfig` (タイプ: `VicaraType.STANDARD`)
*   追加の属性はありません。

### `WeightedVicaraConfig` (タイプ: `VicaraType.WEIGHTED`)
*   追加の属性はありません。

### `ProbeVicaraConfig` (タイプ: `VicaraType.PROBE_SPECIFIC`)
| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`n_probes`** | `10` | いいえ | プローブの数（このタイプのVicaraはプローブごとにリファイナーを持つため）。 |
| **`training_attention_mode`** | "soft" | いいえ | 学習時（`model.train()`）にVicaraが使用するアテンションモード。 |
| **`prediction_attention_mode`** | "hard" | いいえ | 推論時（`model.eval()`）にVicaraが使用するアテンションモード。 |

## 5. デコーダー設定 (`src/configs/decoders.py`)

精製された潜在状態から出力を生成するデコーダーの動作を定義します。具体的なConfigクラスは`DecoderType`によって異なります。

### `BaseDecoderConfig` (共通属性)
| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`type`** | `DecoderType.RECONSTRUCTION` | いいえ | デコーダーのタイプ（`DecoderType` Enum）。 |

### `ReconstructionDecoderConfig` (タイプ: `DecoderType.RECONSTRUCTION`)
| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`input_dim`** | `10` | いいえ | 再構成ターゲットの次元（通常は入力データの次元と同じ）。 |
| **`decoder_hidden_dim`** | `64` | いいえ | デコーダー内の隠れ層の次元。 |

### `CnnDecoderConfig` (タイプ: `DecoderType.CNN`)
| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`channels`** | `3` | いいえ | 出力画像のチャンネル数。 |
| **`img_size`** | `32` | いいえ | 出力画像のサイズ。 |
| **`decoder_hidden_dim`** | `64` | いいえ | デコーダー内の隠れ層の次元。 |

### `LstmDecoderConfig` (タイプ: `DecoderType.LSTM`)
| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`output_dim`** | `10` | いいえ | 出力シーケンスの各タイムステップの特徴量数。 |
| **`seq_len`** | `50` | いいえ | 出力シーケンスの長さ。 |
| **`decoder_hidden_dim`** | `128` | いいえ | 隠れ層の次元。 |
| **`lstm_layers`** | `1` | いいえ | LSTM層の数。 |

### `SimpleSequenceDecoderConfig` (タイプ: `DecoderType.SIMPLE_SEQUENCE`)
| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`output_dim`** | `10` | いいえ | 出力シーケンスの各タイムステップの特徴量数。 |
| **`seq_len`** | `50` | いいえ | 出力シーケンスの長さ。 |
| **`decoder_hidden_dim`** | `128` | いいえ | 隠れ層の次元。 |

## 6. Objective設定 (`src/configs/objectives.py`)

トレーニング中の損失関数（Objective）に関するパラメータを定義します。これは`SamadhiConfig.objective`としてアクセスされます。

### `ObjectiveConfig`
| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`stability_coeff`**| `0.01` | いいえ | **安定性損失**の重み（状態の動き$||S_t - S_{t-1}||$にペナルティを与えます）。 |
| **`entropy_coeff`** | `0.1` | いいえ | **エントロピー損失**の重み（プローブ選択における躊躇にペナルティを与えます）。 |
| **`balance_coeff`** | `0.001` | いいえ | **ロードバランス損失**の重み（均一な使用を強制することで「プローブ崩壊」を防ぎます）。 |
| **`anomaly_margin`** | `5.0` | いいえ | 異常データの復元誤差がこれ以下の場合にペナルティを与えるマージン。高いほど厳しくなります。 |
| **`anomaly_weight`** | `1.0` | いいえ | 異常データに対するマージン損失の重み。高いほど異常を遠ざける力が強まります。 |

## 7. 旧Configパラメータとの対応表

廃止された旧`config`辞書キーと新Configオブジェクト属性の対応を以下に示します。

| 旧`config`キー | 新`SamadhiConfig`属性へのパス |
| :--- | :--- |
| `dim` | `config.dim` |
| `input_dim` | `config.adapter.input_dim` (MLP/LSTM/Transformer), `config.decoder.input_dim` (Reconstruction) |
| `seq_len` | `config.adapter.seq_len`, `config.decoder.seq_len` |
| `n_probes` | `config.vitakka.n_probes` |
| `vicara_type` | `config.vicara.type` |
| `channels` | `config.adapter.channels`, `config.decoder.channels` |
| `img_size` | `config.adapter.img_size`, `config.decoder.img_size` |
| `labels` | `config.labels` |
| `adapter_hidden_dim` | `config.adapter.adapter_hidden_dim` |
| `lstm_layers` | `config.adapter.lstm_layers`, `config.decoder.lstm_layers` |
| `transformer_layers` | `config.adapter.transformer_layers` |
| `transformer_heads` | `config.adapter.transformer_heads` |
| `gate_threshold` | `config.vitakka.gate_threshold` |
| `softmax_temp` | `config.vitakka.softmax_temp` |
| `mix_alpha` | `config.vitakka.mix_alpha` |
| `training_attention_mode` | `config.vitakka.training_attention_mode`, `config.vicara.training_attention_mode` (ProbeVicara) |
| `prediction_attention_mode` | `config.vitakka.prediction_attention_mode`, `config.vicara.prediction_attention_mode` (ProbeVicara) |
| `refine_steps` | `config.vicara.refine_steps` |
| `inertia` | `config.vicara.inertia` |
| `stability_coeff`| `config.objective.stability_coeff` |
| `entropy_coeff` | `config.objective.entropy_coeff` |
| `balance_coeff` | `config.objective.balance_coeff` |
| `anomaly_margin` | `config.objective.anomaly_margin` |
| `anomaly_weight` | `config.objective.anomaly_weight` |

**注意:** 旧Configで存在した一部のObjective固有パラメータは、現在は`SamadhiConfig`の`objective`属性内の`ObjectiveConfig`によって管理されています。
