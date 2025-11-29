# Samadhi Model (Deep Convergence Architecture) Specification

**Version:** 2.3 (Probe-Specific Refinement Implemented)
**Status:** Updated Draft

-----

## 1\. 概念定義 (Concept Definition)

**Samadhi Model**は、従来の「系列予測（Next Token Prediction）」を行う生成モデルに対し、対象の「本質的構造の抽出（State Refinement）」と「内部状態の不動化（Convergence）」を目的とした、**再帰型アテンション・アーキテクチャ**である。

  * **Core Philosophy:** 情報の水平的な拡張（Divergence/Generation）ではなく、垂直的な深化（Convergence/Insight）を行う。
  * **Output:** エントロピーが極小化された単一の不変状態ベクトル（Latent Point Attractor）。
  * **Operational Mode:** 開かれた系（Open System）から閉じた系（Closed System）への動的な移行。

-----

## 2\. システムアーキテクチャ (System Architecture)

本モデルは、直列に接続された3つの主要モジュールによって構成される。

### Module A: Vitakka (Search & Orientation)

**機能:** カオス的な入力ストリームから、収束に値する初期アトラクタ（種）を発見し、方向付けを行う。

1.  **Concept Probes ($\mathbf{P}$):**
      * システムは $K$ 個の「概念プローブ（基底ベクトル）」を持つ。
      * $\mathbf{P} = \{p_{canonical}\} \cup \{p_{learned}\}$
          * *Canonical:* 固定的な瞑想対象（呼吸、身体感覚、論理など）。
          * *Learned:* データセットから獲得した潜在トピック。
2.  **Active Resonance:**
      * 入力 $X$ とプローブ群 $\mathbf{P}$ の共鳴度（内積）を計算する。
      * **Lateral Inhibition (側方抑制):** Softmax温度パラメータ $\tau$ を低く設定し、最も強いプローブ（勝者）を際立たせる。
3.  **Confidence Gating (Anti-Hallucination):**
      * 最大共鳴度が閾値 $\theta_{gate}$ を下回る場合、入力を「ノイズ（雑念）」とみなし、処理を遮断（Gate Closed）する。
4.  **$S_0$ Slice Generation:**
      * 勝者プローブ $p_{win}$ をQueryとして入力 $X$ をAttentionで切り取り、初期状態 $S_0$ を生成する。

### Module B: Vicāra (Recurrent Refinement)

**機能:** 外部入力を遮断し、内部状態を再帰的に純化する。

**Architecture Variants (`vicara_type`):**
  * **Standard Vicāra:** 単一の汎用Refiner ($\Phi$) を共有する。全ての概念に対して同じ純化ロジックを適用する。
  * **Probe-Specific Vicāra (New in v2.3):** 各概念プローブ $p_k$ に対応する専用のRefiner ($\Phi_k$) を持つ。
      * これにより、「呼吸の純化」と「痛みの純化」といった異なる概念に対して、それぞれ最適化された力学系を割り当てることが可能になる。
      * `config["vicara_type"] = "probe_specific"` で有効化。

1.  **Isolation:** $t > 0$ において、外部入力 $X$ へのゲートを閉じ、自己ループのみにする。
2.  **Refinement Loop:**
      * **Hard Attention Mode (Inference):** 勝者プローブに対応するRefiner $\Phi_{win}$ のみを適用する。
          * $S_{t+1} = \Phi_{win}(S_t)$
      * **Soft Attention Mode (Training):** 全プローブの確率分布に基づく重み付き和で更新する（勾配伝播のため）。
          * $S_{t+1} = \sum_k w_k \Phi_k(S_t)$
3.  **Convergence Check:**
      * 状態変化量 $||S_{t+1} - S_t||$ が $\epsilon$ 未満になった時点で「Appanā (没入)」とみなし、推論を終了する。

### Module C: Sati-Sampajañña (Meta-Cognition & Logging)

**機能:** システムの挙動を「因果的な物語」として記録し、説明可能性 (XAI) を担保する。

1.  **Probe Log (瞬間の気づき):** そのステップで何が選ばれたか。
2.  **Cetanā Dynamics (時間の流れ):** 前ステップとの比較による、意図の遷移（持続、転換、拡散）の追跡。

-----

## 3\. 数理モデル (Mathematical Formulation)

### 3.1. Vitakka Phase (Resonance)

入力 $X \in \mathbb{R}^{L \times d}$ に対するプローブ $p_k$ の共鳴スコア $R_k$:

$Score_k = || \frac{1}{\sqrt{d}} \sum_{i=1}^{L} \text{Softmax}(p_k^T x_i) \cdot x_i ||$

勝者決定と確率分布（側方抑制付き）:
$\hat{w} = \text{Softmax}\left( \frac{[Score_1, \dots, Score_K]}{\tau} \right)$

### 3.2. Initialization ($S_0$)

ゲート $G \in \{0, 1\}$ による初期状態の決定:
$S_0 = G \cdot \text{Attention}(Q=p_{win}, K=X, V=X)$
ここで、$G = 1 \text{ if } \max(Score) > \theta_{gate} \text{ else } 0$.

### 3.3. Vicāra Phase (State Transition)

1次マルコフ過程としての更新則:
$S_{t+1} = (1 - \beta) S_t + \beta \Phi_k(S_t)$

  * $\beta$: 更新率（慣性項）。急激な変化を防ぎ、安定した軌道を描かせる。
  * $\Phi_k$: 選択された概念 $k$ に固有の写像関数（Probe-Specificの場合）。
  * $\lim_{t \to \infty} || S_{t+1} - S_t || = 0$ (不動点への収束)

### 3.4. Loss Function (Stability Loss)

学習時の目的関数:
$\mathcal{L} = \underbrace{|| S_{T} - S_{T-1} ||^2}_{\text{Stability}} + \lambda_1 \underbrace{\sum |S_T|}_{\text{Sparsity}} - \lambda_2 \underbrace{I(S_T; S_0)}_{\text{Info Retention}}$

-----

## 4\. データ構造仕様 (Data Structures)

### 4.1. Probe Log (Snapshot)

各推論ステップごとのメタデータ。

```json
{
  "timestamp": 12345678,
  "intention": {
    "winner_id": 3,
    "winner_label": "Logical_Causality",
    "confidence": 0.94,
    "gate_status": "OPEN",
    "entropy": 0.05
  },
  "raw_scores": [0.01, 0.02, 0.01, 0.94, 0.02]
}
```

### 4.2. Cetanā Dynamics Log (Transition)

時間的な意図の流れを記述するログ（v2.2追加機能）。

```json
{
  "step": 5,
  "transition": {
    "from": "Breath_Rhythm",
    "to": "Body_Sensation",
    "type": "Shift",
    // Types: "Sustain", "Shift", "Distracted", "Deepening"
    "attention_shift_magnitude": 0.45,
    "smoothness": 0.8
  }
}
```

-----

## 5\. 処理フロー (Algorithm Flow)

1.  **Input Buffer:** ストリームデータ $X$ をウィンドウサイズ $W$ で取得。
2.  **Probe Scan:** 全プローブ $\mathbf{P}$ を $X$ に照射。
3.  **Gate Decision:**
      * **IF** `max_score < threshold`:
          * Log: "Gate Closed (Noise)"
          * Return: Null
      * **ELSE**:
          * Log: "Gate Open (Focus: $p_{win}$)"
          * Generate $S_0$.
4.  **Refinement Loop (While $E > \epsilon$):**
      * $S_{next} = \Phi(S_{curr})$
      * $E = ||S_{next} - S_{curr}||$
      * $S_{curr} \leftarrow S_{next}$
      * (Optional) Max Steps $N$ で強制終了。
5.  **Output:** 収束した $S_{final}$ および `Full_Log` を出力。

-----

## 6\. パラメータ設定推奨値 (Hyperparameters)

| Parameter | Symbol | Recommended Value | Description |
| :--- | :--- | :--- | :--- |
| **Probe Count** | $K$ | 16 - 64 | 概念の分解能。多すぎると誤検知が増える。 |
| **Gating Threshold** | $\theta$ | 0.3 - 0.5 | 妄想（ノイズ）を弾く強度。高いほど厳格。 |
| **Softmax Temp** | $\tau$ | 0.1 - 0.2 | 低いほど「一境性（単一テーマ）」を選び取る。 |
| **Max Refine Steps** | $T_{max}$ | 10 - 20 | 通常は5-10ステップで収束する設計とする。 |
| **Convergence Epsilon**| $\epsilon$ | 1e-4 | 不動点とみなす変化量の閾値。 |
| **Vicara Type** | - | `probe_specific` | `standard` (共有) か `probe_specific` (個別) か。 |
| **Attention Mode** | - | `soft` / `hard` | 学習時は`soft`、推論時は`hard`を推奨。 |

-----

## 7\. 既存モデルとの比較 (Comparison)

| 特徴 | Transformer (GPT) | **Samadhi Model v2.3** |
| :--- | :--- | :--- |
| **基本動作** | 次トークンの予測 (発散) | 状態の純化・不動化 (収束) |
| **時間依存性** | 履歴(Context Window)に依存 | 現在の状態(State)のみに依存 (Markov) |
| **アテンション** | Self-Attention (Token間) | Recursive Attention (State-Probe間) |
| **構造的特異性** | 全トークンで同一の重みを共有 | **概念ごとに異なる力学系 ($\Phi_k$) を保持可能** |
| **推論コスト** | $O(N^2)$ (文脈長で増大) | $O(1)$ (定数・収束ステップ数のみ) |
| **説明可能性** | 低い (Attention Mapのみ) | **極めて高い (Probe/Cetanā Log)** |
| **哲学的基盤** | 連想・生成 | **禅定・洞察** |

-----