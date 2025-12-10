# Satipatthana モデル設定パラメータ概要 (v4.0 Config System)

このドキュメントでは、`Satipatthana Framework`のConfigパラメータについて説明します。v4.0では、3エンジン構造（SamathaEngine, VipassanaEngine, ConditionalDecoder）に対応した新しいConfig体系に移行しました。

Configのルートは`samadhi/configs/system.py`で定義されている`SystemConfig`オブジェクトです。

## 1. ルート設定: `SystemConfig` (`samadhi/configs/system.py`)

`SystemConfig`は、モデル全体のグローバルパラメータと、各主要エンジンのサブ設定を保持します。

| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`samatha`** | `SamathaConfig()` | いいえ | SamathaEngineの設定オブジェクト |
| **`vipassana`** | `VipassanaEngineConfig()` | いいえ | VipassanaEngineの設定オブジェクト |
| **`use_label_guidance`** | `False` | いいえ | Stage 1でラベルガイダンスを使用するか |
| **`seed`** | `42` | いいえ | モデル初期化のための乱数シード |

## 2. SamathaEngine設定: `SamathaConfig`

SamathaEngine（収束エンジン）の構成を定義します。

| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`latent_dim`** | `64` | いいえ | 潜在状態ベクトル（$S$）の次元 |
| **`adapter`** | `MlpAdapterConfig()` | いいえ | Adapterコンポーネントの設定 |
| **`augmenter`** | `AugmenterConfig()` | いいえ | Augmenterコンポーネントの設定 |
| **`vitakka`** | `VitakkaConfig()` | いいえ | Vitakkaコンポーネントの設定 |
| **`vicara`** | `StandardVicaraConfig()` | いいえ | Vicaraコンポーネントの設定 |
| **`sati`** | `SatiConfig()` | いいえ | Satiコンポーネントの設定 |
| **`max_steps`** | `10` | いいえ | Vicaraループの最大ステップ数 |

## 3. VipassanaEngine設定: `VipassanaEngineConfig`

VipassanaEngine（内省エンジン）の構成を定義します。

| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`vipassana`** | `StandardVipassanaConfig()` | いいえ | Vipassanaコンポーネントの設定 |

## 4. アダプター設定 (`samadhi/configs/adapters.py`)

入力データから潜在状態を生成するアダプターの動作を定義します。

### `MlpAdapterConfig` (タイプ: `AdapterType.MLP`)

| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`input_dim`** | - | **はい** | 入力データの次元（特徴量数） |
| **`latent_dim`** | `64` | いいえ | 出力潜在空間の次元 |
| **`hidden_dim`** | `256` | いいえ | 隠れ層の次元 |
| **`dropout`** | `0.1` | いいえ | ドロップアウト率 |

### `CnnAdapterConfig` (タイプ: `AdapterType.CNN`)

| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`channels`** | - | **はい** | 入力画像のチャンネル数 |
| **`img_size`** | - | **はい** | 入力画像のサイズ（正方形を想定） |
| **`latent_dim`** | `64` | いいえ | 出力潜在空間の次元 |

### `LstmAdapterConfig` (タイプ: `AdapterType.LSTM`)

| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`input_dim`** | - | **はい** | 各タイムステップの特徴量数 |
| **`seq_len`** | - | **はい** | 時系列シーケンスの長さ |
| **`latent_dim`** | `64` | いいえ | 出力潜在空間の次元 |
| **`hidden_dim`** | `128` | いいえ | 隠れ層の次元 |
| **`num_layers`** | `1` | いいえ | LSTM層の数 |

### `TransformerAdapterConfig` (タイプ: `AdapterType.TRANSFORMER`)

| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`input_dim`** | - | **はい** | 各タイムステップの特徴量数 |
| **`seq_len`** | - | **はい** | 時系列シーケンスの長さ |
| **`latent_dim`** | `64` | いいえ | 出力潜在空間の次元 |
| **`hidden_dim`** | `128` | いいえ | 隠れ層の次元 |
| **`num_layers`** | `2` | いいえ | Transformer Encoder層の数 |
| **`num_heads`** | `4` | いいえ | Multi-head Attentionのヘッド数 |

## 5. Augmenter設定 (`samadhi/configs/augmenter.py`)

入力に対するノイズ付与を制御します。

### `AugmenterConfig`

| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`type`** | `AugmenterType.IDENTITY` | いいえ | Augmenterのタイプ（`IDENTITY`, `GAUSSIAN`） |
| **`max_noise_std`** | `0.3` | いいえ | (GAUSSIAN) 最大ノイズ標準偏差 |

## 6. Vitakka設定 (`samadhi/configs/vitakka.py`)

初期状態（$S_0$）の生成とプローブの管理を制御します。

### `VitakkaConfig`

| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`num_probes`** | `16` | いいえ | 概念プローブの数 |
| **`temperature`** | `0.2` | いいえ | Softmaxの温度パラメータ（低いほど一境性） |
| **`trainable`** | `True` | いいえ | プローブが学習可能かどうか |

## 7. Vicara設定 (`samadhi/configs/vicara.py`)

潜在状態の再帰的精製プロセスを制御します。

### 共通属性

| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`type`** | `VicaraType.STANDARD` | いいえ | Vicaraのタイプ |
| **`beta`** | `0.5` | いいえ | 状態更新の慣性パラメータ: $S_{t+1} = (1-\beta)S_t + \beta\Phi(S_t)$ |

### `StandardVicaraConfig` (タイプ: `VicaraType.STANDARD`)

| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`latent_dim`** | `64` | いいえ | 潜在空間の次元 |
| **`hidden_dim`** | `128` | いいえ | Refiner内の隠れ層の次元 |

### `WeightedVicaraConfig` (タイプ: `VicaraType.WEIGHTED`)

複数Refinerの重み付け合成。

### `ProbeSpecificVicaraConfig` (タイプ: `VicaraType.PROBE_SPECIFIC`)

| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`num_probes`** | `16` | いいえ | プローブの数（プローブごとにRefinerを持つ） |
| **`latent_dim`** | `64` | いいえ | 潜在空間の次元 |

## 8. Sati設定 (`samadhi/configs/sati.py`)

収束判定と停止制御を定義します。

### `SatiConfig`

| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`type`** | `SatiType.FIXED_STEP` | いいえ | Satiのタイプ（`FIXED_STEP`, `THRESHOLD`） |
| **`threshold`** | `1e-4` | いいえ | (THRESHOLD) 収束判定閾値 $\epsilon$ |

## 9. Vipassana設定 (`samadhi/configs/vipassana.py`)

メタ認知モジュールの動作を定義します。

### `StandardVipassanaConfig`

| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`context_dim`** | - | **はい** | 出力文脈ベクトル $V_{ctx}$ の次元 |
| **`latent_dim`** | `64` | いいえ | 入力潜在状態の次元 |
| **`hidden_dim`** | `64` | いいえ | LogEncoder内の隠れ層の次元 |

## 10. デコーダー設定 (`samadhi/configs/decoders.py`)

### `ReconstructionDecoderConfig` (学習補助用)

| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`input_dim`** | - | **はい** | 再構成ターゲットの次元 |
| **`latent_dim`** | `64` | いいえ | 入力潜在状態の次元 |
| **`hidden_dim`** | `128` | いいえ | 隠れ層の次元 |

### `ConditionalDecoderConfig` (推論用)

| 属性 | デフォルト値 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| **`dim`** | - | **はい** | 入力潜在状態 $S^*$ の次元 |
| **`context_dim`** | - | **はい** | 文脈ベクトル $V_{ctx}$ の次元 |
| **`output_dim`** | - | **はい** | 出力の次元 |
| **`hidden_dim`** | `128` | いいえ | 隠れ層の次元 |

**重要:** `ConditionalDecoder` の入力次元は `dim + context_dim` です。

## 11. Config構築例

### Factory関数を使用した構築

```python
from samadhi.configs import SystemConfig, SamathaConfig, VipassanaEngineConfig
from samadhi.configs import create_adapter_config, create_vicara_config
from samadhi.configs import AugmenterConfig, VitakkaConfig, SatiConfig
from samadhi.configs import StandardVipassanaConfig
from samadhi.configs.enums import AugmenterType, SatiType

config = SystemConfig(
    samatha=SamathaConfig(
        latent_dim=64,
        adapter=create_adapter_config("mlp", input_dim=784, latent_dim=64),
        augmenter=AugmenterConfig(type=AugmenterType.GAUSSIAN, max_noise_std=0.3),
        vitakka=VitakkaConfig(num_probes=16, temperature=0.2),
        vicara=create_vicara_config("standard", latent_dim=64),
        sati=SatiConfig(type=SatiType.THRESHOLD, threshold=1e-4),
        max_steps=10,
    ),
    vipassana=VipassanaEngineConfig(
        vipassana=StandardVipassanaConfig(context_dim=32, latent_dim=64),
    ),
    use_label_guidance=True,
    seed=42,
)
```

### 時系列異常検知の例

```python
config = SystemConfig(
    samatha=SamathaConfig(
        latent_dim=128,
        adapter=create_adapter_config(
            "lstm",
            input_dim=10,
            seq_len=50,
            latent_dim=128,
            hidden_dim=256,
        ),
        augmenter=AugmenterConfig(type=AugmenterType.GAUSSIAN, max_noise_std=0.2),
        vitakka=VitakkaConfig(num_probes=8),
        vicara=create_vicara_config("standard", latent_dim=128),
        sati=SatiConfig(type=SatiType.THRESHOLD, threshold=1e-4),
    ),
    vipassana=VipassanaEngineConfig(
        vipassana=StandardVipassanaConfig(context_dim=64, latent_dim=128),
    ),
)
```

## 12. 旧Configパラメータとの対応表 (v3.1 → v4.0)

| 旧`SamadhiConfig`属性 | 新`SystemConfig`属性へのパス |
| :--- | :--- |
| `config.dim` | `config.samatha.latent_dim` |
| `config.adapter.input_dim` | `config.samatha.adapter.input_dim` |
| `config.vitakka.n_probes` | `config.samatha.vitakka.num_probes` |
| `config.vicara.refine_steps` | `config.samatha.max_steps` |
| `config.vicara.inertia` | `config.samatha.vicara.beta` |
| `config.objective.*` | 削除（Trainerで管理） |
| - | `config.vipassana.vipassana.context_dim` (新規) |
| - | `config.samatha.augmenter.*` (新規) |
| - | `config.samatha.sati.*` (新規) |

**注意:** v4.0では、Objective設定はConfigから分離され、`SamadhiV4Trainer`が各Stageに応じて内部で管理します。
