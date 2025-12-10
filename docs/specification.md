# Satipatthana Framework 仕様書

**Version:** 4.0 (The Three Engines & Guided Convergence)
**Status:** Active Specification

-----

## 1. 概要

### 1.1. 本文書の目的

本文書は Satipatthana Framework の**実装仕様書**である。以下を定義する：

* システムアーキテクチャとデータフロー
* コンポーネントのインターフェースと責務
* 数学的定式化（更新則、損失関数）
* 学習カリキュラム
* ハイパーパラメータ

理論的背景と設計思想については [theory.md](theory.md) を参照。

### 1.2. 想定読者

* **MLエンジニア** — フレームワークの実装・拡張
* **研究者** — 実験の再現
* **コードレビュアー** — システム挙動の理解

### 1.3. 前提知識

* 深層学習の基礎（PyTorch）
* Attention機構への理解
* 不動点反復法（あれば望ましい）

-----

## 2. 用語と記号

### 2.1. 主要記号

| 記号 | 型 | 説明 |
|:---|:---|:---|
| $X$ | Tensor (Batch, *) | 生入力データ |
| $z$ | Tensor (Batch, $d$) | Adapter後の潜在ベクトル |
| $S_t$ | Tensor (Batch, $d$) | 反復 $t$ における潜在状態 |
| $S^*$ | Tensor (Batch, $d$) | 収束した不動点状態 |
| $S_0$ | Tensor (Batch, $d$) | Vitakkaからの初期状態 |
| $P_k$ | Tensor ($d$,) | $k$ 番目の概念プローブベクトル |
| $V_{ctx}$ | Tensor (Batch, $c$) | Vipassana文脈ベクトル |
| $\alpha$ | Tensor (Batch, 1) | 信頼度スコア (0.0–1.0) |
| $Y$ | Tensor (Batch, output_dim) | 最終出力 |
| $\mathcal{T}$ | SantanaLog | 思考軌跡 $[S_0, S_1, \dots, S^*]$ |

### 2.2. ハイパーパラメータ記号

| 記号 | 説明 | 推奨範囲 |
|:---|:---|:---|
| $d$ | 潜在空間の次元 | 64–256 |
| $c$ | Vipassana文脈次元 | 32–128 |
| $K$ | 概念プローブ数 | 8–32 |
| $T$ | Vicara最大ステップ数 | 6–20 |
| $\beta$ | 状態更新の慣性 | 0.3–0.7 |
| $\epsilon$ | 収束判定閾値 (Sati) | 1e-4 |
| $\lambda_r$ | 再構成損失の重み | 0.1–0.3 |
| $\lambda_g$ | Guidance損失の重み | 0.1–0.5 |

-----

## 3. システムアーキテクチャ

本フレームワークは、3つの主要エンジン（Samatha, Vipassana, Decoder）と、それらを構成するモジュラーコンポーネント群によって構成される。

### 3.1. データフロー概要

```txt
Raw Input (X)
    ↓
[SamathaEngine]
    Augmenter → Adapter → Vitakka → Vicara loop (w/ Sati) → S*, SantanaLog
    ↓
[VipassanaEngine]
    S* + SantanaLog → V_ctx, α (trust_score)
    ↓
[ConditionalDecoder]
    S* + V_ctx → Output (Y)
```

### 3.2. クラス図

![Class Diagram](diagrams/images/v4_class_diagram.png)

### 3.3. Engine 1: SamathaEngine

**役割:** 世界モデル。いかなる入力も「意味のある点」に収束させる。

**入力:** Raw Data `X` (Batch, *)
**出力:**

* `S*` (Batch, Dim): 収束した潜在状態
* `SantanaLog`: 思考軌跡を記録したオブジェクト
* `severity` (Batch,): ノイズ強度（Vipassanaターゲット用）

**構成コンポーネント:**

| コンポーネント | 役割 |
|:---|:---|
| **Adapter** | 生入力を潜在空間へ投影・正規化 |
| **Augmenter** | 入力にノイズ/摂動を付与（学習時） |
| **Vitakka** | プローブベースの初期状態 $S_0$ 生成 |
| **Vicara** | 1ステップの状態更新 ($S_t \rightarrow S_{t+1}$) |
| **Sati** | 収束判定・停止制御 |

**特徴:** タスクやラベルには依存せず、「構造の抽出」のみを行う。`drunk_mode` フラグにより内部的な摂動制御が可能。

### 3.4. Engine 2: VipassanaEngine

**役割:** メタ認知。Samathaの思考プロセス（ログ）が健全だったか監視する。

**入力:** `S*` (Batch, Dim) + `SantanaLog`
**出力:**

* `V_ctx` (Batch, context_dim): デコーダーへのヒント情報（「迷い」の埋め込み表現）
* `α` (Batch, 1): 信頼度スコア (0.0〜1.0)

**構成:** `StandardVipassana` (LogEncoder + ConfidenceMonitor)

### 3.5. Engine 3: ConditionalDecoder

**役割:** 表現。状態と文脈を統合して、人間にわかる形にする。

**入力:** `S*` (Batch, Dim) + `V_ctx` (Batch, context_dim) → Concatenate → (Batch, Dim + context_dim)
**出力:** `Y` (Batch, output_dim)

**特徴:** 「自信がない時は、自信がないような出力（分散を広げる等）」が可能になり、**謙虚な表現**を実現する。**推論時に使用される唯一のDecoder**。

### 3.6. Reconstruction Heads & AuxHead (学習補助)

学習の安定化を目的とした補助モジュール。**推論時には使用されない。**

* **`adapter_recon_head`** (Stage 0用): Adapterの出力 `z` から元入力を再構成
* **`samatha_recon_head`** (Stage 1用): 収束点 `S*` から元入力を再構成
* **`AuxHead`** (Stage 1用): `S*` (次元: $d$) からタスク予測を行う補助ヘッド

**重要: AuxHead と ConditionalDecoder の関係**

| モジュール | 入力次元 | 用途 | Stage 3での扱い |
|:---|:---|:---|:---|
| `AuxHead` | $d$ (`S*`のみ) | Stage 1のGuidance学習 | **破棄** |
| `ConditionalDecoder` | $d + c$ (`S*` ⊕ `V_ctx`) | Stage 3以降の推論 | 新規学習 |

Stage 1の `AuxHead` と Stage 3の `ConditionalDecoder` は**入力次元が異なるため、物理的に別モジュール**である。`AuxHead` の重みは Stage 3 には転移されず、`ConditionalDecoder` はゼロから学習される。

-----

## 4. コンポーネント詳細

### 4.0. コンポーネント I/O サマリー

| コンポーネント | 入力 | 出力 | インターフェース |
|:---|:---|:---|:---|
| **Adapter** | $X$ (Batch, *) | $z$ (Batch, $d$) | `BaseAdapter` |
| **Augmenter** | $X$ (Batch, *) | $(X_{aug}, severity)$ | `BaseAugmenter` |
| **Vitakka** | $z$ (Batch, $d$) | $(S_0, metadata)$ | `BaseVitakka` |
| **Vicara** | $S_t$ (Batch, $d$), context | $S_{t+1}$ (Batch, $d$) | `BaseVicara` |
| **Sati** | $S_t$, $\mathcal{T}$ | $(should\_stop, info)$ | `BaseSati` |
| **Vipassana** | $S^*$, $\mathcal{T}$ | $(V_{ctx}, \alpha)$ | `BaseVipassana` |
| **ConditionalDecoder** | $S^* \oplus V_{ctx}$ (Batch, $d+c$) | $Y$ (Batch, output\_dim) | `BaseDecoder` |

### 4.1. Adapter

**機能:** 生の外部入力 $X_{raw}$ を潜在空間へ投影・正規化する。

* **Interface:** `BaseAdapter`
* **実装:** `MlpAdapter`, `CnnAdapter`, `LstmAdapter`, `TransformerAdapter`
* **入力:** 生データ $X$ (Batch, *)
* **出力:** 潜在ベクトル $z \in \mathbb{R}^d$

### 4.2. Augmenter

**機能:** 入力に対して環境ノイズや摂動を加える。

* **Interface:** `BaseAugmenter`
* **実装:** `IdentityAugmenter`, `GaussianNoiseAugmenter`
* **入力:** 生データ $X$ (Batch, *)
* **出力:** `(x_augmented, severity)` — severityはサンプルごとのノイズ強度 $\in [0, 1]$

### 4.3. Vitakka

**機能:** 潜在空間内での初期アトラクタ探索。

1. **Active Resonance:** 概念プローブ群 $\mathbf{P}$ と入力の共鳴度を計算
2. **$S_0$ Generation:** 勝者プローブをQueryとして初期状態を生成

* **Interface:** `BaseVitakka`
* **入力:** 潜在ベクトル $z$ (Batch, $d$)
* **出力:** `(s0, metadata)` — metadataには `winner_id`, `probs` 等を含む

### 4.4. Vicara

**機能:** 1ステップの状態更新。

$$S_{t+1} = (1 - \beta) S_t + \beta \Phi(S_t)$$

* **Interface:** `BaseVicara`
* **実装:** `StandardVicara`, `WeightedVicara`, `ProbeSpecificVicara`
* **入力:** 現在の状態 $S_t$ (Batch, $d$)、Vitakkaからのオプショナルなcontext
* **出力:** 次の状態 $S_{t+1}$ (Batch, $d$)
* **責務:** 単一ステップの更新のみ。ループ制御はSamathaEngineに委譲。

**バリエーション:**

| クラス | 説明 |
|:---|:---|
| `StandardVicara` | 単一Refinerで状態更新。最もシンプル |
| `WeightedVicara` | 複数Refinerの重み付け合成 |
| `ProbeSpecificVicara` | Vitakkaの勝者Probe/確率に基づきRefinerを選択 |

### 4.5. Sati

**機能:** 収束判定と停止制御。

* **Interface:** `BaseSati`
* **実装:** `FixedStepSati`, `ThresholdSati`
* **入力:** 現在の状態 $S_t$ (Batch, $d$)、軌跡 $\mathcal{T}$
* **出力:** `(should_stop: bool, info: dict)`
* **Stop Condition:** 状態変化エネルギー $||S_{t+1} - S_t||$ が閾値 $\epsilon$ を下回った時点で停止

### 4.6. Vipassana

**機能:** Samathaの思考ログを監視し、論理的整合性と信頼度を評価するメタ認知モジュール。

* **Interface:** `BaseVipassana`
* **実装:** `StandardVipassana`
* **入力:** 収束状態 $S^*$ (Batch, $d$)、軌跡 $\mathcal{T}$
* **出力:** 文脈ベクトル $V_{ctx}$ (Batch, $c$)、信頼度スコア $\alpha$ (Batch, 1)
* **LogEncoder:** 時系列ログ $\mathcal{T}$ を固定長ベクトルに圧縮
  * **推奨実装:** Bi-LSTM または Transformer Encoder (1-2 layers)。思考の「順序」と「収束の加速度」を捉えるには時系列モデルが必須。
* **ConfidenceMonitor:** 「迷い」や「矛盾」を検知し、信頼度スコア $\alpha$ と文脈ベクトル $V_{ctx}$ を出力

**フォールバック戦略:** 推論時に $\alpha < \text{threshold}$ の場合：

* デフォルト回答（"I don't know"）を出力
* または出力分布の分散（Variance）を最大化
* または検索トリガー/回答拒否を発動

-----

## 5. 数理モデル

### 5.1. Samatha Phase (収束)

**状態更新則:**
$$S_{t+1} = (1 - \beta) S_t + \beta \Phi(S_t)$$

**収束保証:** $\beta \in (0, 1)$ の慣性更新により、写像の実効リプシッツ定数が低減される。$\Phi$ のリプシッツ定数が $L$ の場合、合成写像の実効定数は $L_{eff} = (1 - \beta) + \beta L$ となる。$L < 1$ の場合、または安定性損失が縮小を促進する場合、不動点への収束が促進される。

**停止条件 (Sati):**
$$\text{Stop if } ||S_{t+1} - S_t|| < \epsilon_{sati}$$

### 5.2. Vipassana Phase (内省)

思考ログ $\mathcal{T} = [S_0, \dots, S^*]$ から信頼度を算出する。

$$V_{ctx} = \text{Encoder}(\mathcal{T})$$
$$\alpha = \sigma(\text{Linear}(V_{ctx})) \in [0, 1]$$

* Target ($\hat{\alpha}$): Clean=1.0, Mismatch/Drunk=0.0

### 5.3. Loss Function (Stage-wise)

学習ステージごとに目的関数が切り替わる。

* **Stage 0 (Adapter Pre-training):** Reconstruction Only
    $$\mathcal{L}_0 = \mathcal{L}_{recon}(X, \hat{X}_{adapter})$$

* **Stage 1 (Samatha Training):** Stability + Reconstruction + (Optional) Label Guidance
    $$\mathcal{L}_1 = ||S_T - S_{T-1}||^2 + \lambda_r \mathcal{L}_{recon} + \lambda_g \mathcal{L}_{task}(y, \text{AuxHead}(S^*))$$

* **Stage 2 (Vipassana Training):** Binary Cross Entropy (Contrastive)
    $$\mathcal{L}_2 = \text{BCE}(\alpha, \hat{\alpha})$$

* **Stage 3 (Decoder Fine-tuning):** Task Specific Loss
    $$\mathcal{L}_3 = \mathcal{L}_{task}(y, \text{Decoder}(S^*, V_{ctx}))$$

-----

## 6. データ構造仕様

### 6.1. SantanaLog

収束過程の状態履歴を記録するオブジェクト。

```python
class SantanaLog:
    def add(self, state: Tensor) -> None:
        """状態を軌跡に追加"""

    def to_tensor(self) -> Tensor:
        """軌跡をテンソル化 (Steps, Batch, Dim)"""

    def __len__(self) -> int:
        """記録されたステップ数"""
```

### 6.2. SystemOutput

```python
@dataclass
class SystemOutput:
    output: Tensor        # デコード結果
    s_star: Tensor        # 収束した潜在状態
    v_ctx: Tensor         # Vipassanaの文脈ベクトル
    trust_score: Tensor   # 信頼度スコア (0.0〜1.0)
    santana: SantanaLog   # 思考軌跡
    severity: Tensor      # ノイズ強度
```

-----

## 7. 処理フロー

### 7.1. 推論シーケンス図

![Inference Sequence Diagram](diagrams/images/v4_sequence_diagram_inference.png)

### 7.2. 推論フロー

```python
def inference(x: Tensor) -> SystemOutput:
    # Phase 1: Samatha (収束)
    s_star, santana, severity = samatha_engine(x, run_augmenter=False)

    # Phase 2: Vipassana (内省)
    v_ctx, trust_score = vipassana_engine(s_star, santana)

    # Phase 3: Decode (表現)
    output = conditional_decoder(concat(s_star, v_ctx))

    return SystemOutput(output, s_star, v_ctx, trust_score, santana, severity)
```

### 7.3. SamathaEngine内部フロー

```python
def samatha_forward(x, noise_level=0.0, run_augmenter=True):
    # Augment (学習時のみ)
    if run_augmenter:
        x_aug, severity = augmenter(x, noise_level)
    else:
        x_aug, severity = x, zeros(batch_size)

    # Adapt
    z = adapter(x_aug)

    # Vitakka: 初期状態生成
    s0, metadata = vitakka(z)

    # Vicara loop with Sati
    santana = SantanaLog()
    s_t = s0
    santana.add(s_t)

    for step in range(max_steps):
        s_t = vicara(s_t, context=metadata)
        santana.add(s_t)

        should_stop, _ = sati(s_t, santana)
        if should_stop:
            break

    return s_t, santana, severity
```

-----

## 8. 学習カリキュラム (4-Stage)

### 8.1. 学習ポリシー

| Stage | Name | Train対象 | Freeze対象 | 目的関数 |
|:---|:---|:---|:---|:---|
| **0** | Adapter Pre-training | Adapter, adapter_recon_head | 他すべて | Reconstruction Loss |
| **1** | Samatha Training | Adapter, Vitakka, Vicara, Sati, (samatha_recon_head, AuxHead) | Vipassana, TaskDecoder | Stability + Recon + (Guidance) |
| **2** | Vipassana Training | Vipassana | 他すべて | BCE (Contrastive) |
| **3** | Decoder Fine-tuning | TaskDecoder | 他すべて | Task Specific Loss |

### 8.2. 反復戦略

| モード | 説明 | ユースケース |
|:---|:---|:---|
| **Fixed Steps** | 常に $T$ 回反復 | 学習（勾配の安定性） |
| **Early Stopping** | $\|S_{t+1} - S_t\| < \epsilon$ で停止 | 推論（効率性） |
| **Hybrid** | 最小ステップ数実行後、早期停止を許可 | 安定性と効率のバランス |

**推奨:**

* **学習時:** Fixed steps ($T = 10$) で勾配フローを安定化
* **推論時:** Early stopping ($\epsilon = 10^{-4}$) で効率化
* **切替:** `SatiConfig.mode` で戦略を切り替え

### 8.3. Stage遷移基準

| 遷移 | 基準 | フォールバック |
|:---|:---|:---|
| 0 → 1 | Reconstruction lossがプラトー | 固定エポック (例: 5) |
| 1 → 2 | Stability loss $< 10^{-3}$ | 固定エポック (例: 10) |
| 2 → 3 | Vipassana BCE $< 0.3$ | 固定エポック (例: 5) |

**Early Stopping:** ステージごとに検証損失を監視。`patience` エポック改善がなければ次のステージへ遷移。

### 8.4. Stage 2 ノイズ生成戦略

Vipassanaにメタ認知能力を習得させるための3種類のデータ生成戦略:

1. **Environmental Ambiguity (Augmented Path)**
   * 入力データへのノイズ付与
   * Target: `1.0 - severity`

2. **Internal Dysfunction (Drunk Path)**
   * SamathaEngine内部の摂動（`drunk_mode=True`）
   * 具体的実装: Vicara内のDropout率を上げる、Refinerの重みに一時的ノイズを加算、Vitakkaの温度パラメータを乱す等
   * Target: `0.0`

3. **Logical Inconsistency (Mismatch Path)**
   * バッチ内でS*とSantanaLogをシャッフル
   * Target: `0.0`

**バッチ構成 (推奨):**

| パス | 割合 | 目的 |
|:---|:---|:---|
| Clean | 25% | ベースライン信頼度 |
| Augmented | 25% | 環境的不確実性 |
| Drunk | 25% | 内部機能不全の検知 |
| Mismatch | 25% | 論理的不整合の検知 |

-----

## 9. ハイパーパラメータ

### 9.1. モデルアーキテクチャ

| Key | Symbol | Recommended | Description |
|:---|:---|:---|:---|
| `latent_dim` | $d$ | 64-256 | 潜在空間の次元 |
| `context_dim` | $c$ | 32-128 | Vipassana出力の次元 |
| `num_probes` | $K$ | 8-32 | Vitakkaのプローブ数 |
| `max_steps` | $T$ | 6-20 | Vicaraの最大ステップ数 |

### 9.2. 学習戦略

| Key | Symbol | Recommended | Description |
|:---|:---|:---|:---|
| `sati_threshold` | $\epsilon$ | 1e-4 | 収束判定閾値 |
| `beta` | $\beta$ | 0.3-0.7 | 状態更新の慣性パラメータ |
| `guidance_weight` | $\lambda_g$ | 0.1-0.5 | (Stage 1) Guidance Lossの強さ |
| `recon_weight` | $\lambda_r$ | 0.1-0.3 | Reconstruction Lossの重み |

-----

## 10. 応用と学習戦略

教師ありタスクにおいては **Stage 1 Guidance (AuxHead)** を積極的に使用し、Samathaの収束空間をタスク向けに最適化する。

| 応用タスク | Stage 1 Strategy | Stage 2 Role | Stage 3 Decoder |
|:---|:---|:---|:---|
| **教師あり分類** | Guidance (CE Loss) | Hallucination Check | Classifier (Softmax) |
| **教師あり回帰** | Guidance (MSE Loss) | Uncertainty Est. | Regressor (Linear) |
| **異常検知** | Reconstruction Only | Anomaly Score (最終出力) | Identity |
| **構造発見** | Stability Only | Boundary Detection | None |

-----

## 11. アーキテクチャの拡張性

`SystemConfig` と各種 `ComponentConfig` を使用して、コンポーネントを自由に組み合わせることができる。

### 11.1. タスク別カスタマイズ例

| タスク | Adapter | Augmenter | Vicara | Decoder |
|:---|:---|:---|:---|:---|
| **時系列異常検知** | LSTM | Gaussian | Standard | Reconstruction |
| **画像分類** | CNN | Identity | Standard | Conditional |
| **対話意図推定** | Transformer | Identity | ProbeSpecific | Conditional |
| **ロボット制御** | MLP | Gaussian | Weighted | Conditional |

### 11.2. Config Example

```python
from satipatthana.configs import SystemConfig, SamathaConfig, VipassanaEngineConfig
from satipatthana.configs import create_adapter_config, create_vicara_config

config = SystemConfig(
    samatha=SamathaConfig(
        adapter=create_adapter_config("mlp", input_dim=784, latent_dim=64),
        augmenter=AugmenterConfig(type=AugmenterType.GAUSSIAN, max_noise_std=0.3),
        vitakka=VitakkaConfig(num_probes=16),
        vicara=create_vicara_config("standard", latent_dim=64),
        sati=SatiConfig(type=SatiType.THRESHOLD, threshold=1e-4),
    ),
    vipassana=VipassanaEngineConfig(
        vipassana=StandardVipassanaConfig(context_dim=32),
    ),
    use_label_guidance=True,
)
```
