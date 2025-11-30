# Samadhi Framework (Deep Convergence Architecture) Specification

**Version:** 3.0 (Framework Modularization)
**Status:** Active Specification

-----

## 1. 概念定義 (Concept Definition)

**Samadhi Framework**は、従来の「系列予測（Next Token Prediction）」を行う生成モデルに対し、対象の「本質的構造の抽出（State Refinement）」と「内部状態の不動化（Convergence）」を目的とした、**再帰型アテンション・アーキテクチャ**である。

  * **Core Philosophy:** 情報の水平的な拡張（Divergence/Generation）ではなく、垂直的な深化（Convergence/Insight）を行う。
  * **Output:** エントロピーが極小化された単一の不変状態ベクトル（Latent Point Attractor）。
  * **Operational Mode:** 開かれた系（Open System）から閉じた系（Closed System）への動的な移行。

-----

## 2. システムアーキテクチャ (System Architecture)

本フレームワークは、モジュール化されたコンポーネント群（Adapter, Vitakka, Vicara, Decoder）を組み合わせることで構築される。

### Adapter (Manasikāra - Input Adaptation)

**機能:** 異なるモダリティ（画像、時系列、テキストなど）を持つ生の外部入力 $X_{raw}$ を、モデル固有の潜在空間（Samadhi Space）へ投影・正規化する。

*   **Role:** 外界の信号を、モデルが扱える「意味の形式」へと変換する（作意）。
*   **Interface:** `BaseAdapter` (`src/components/adapters/base.py`)
*   **Implementations:**
    *   *MlpAdapter:* タブラーデータやフラットなベクトル用。
    *   *CnnAdapter:* 画像データ用 (Conv2d)。
    *   *LstmAdapter:* 時系列データ用 (LSTM)。
    *   *TransformerAdapter:* 系列データ用 (Transformer Encoder)。
*   **Output:** 潜在ベクトル $X_{adapted} \in \mathbb{R}^d$。

### Vitakka (Search & Orientation)

**機能:** カオス的な入力ストリーム（$X_{adapted}$）から、収束に値する初期アトラクタ（種）を発見し、方向付けを行う。

*   **Interface:** `BaseVitakka` (`src/components/vitakka/base.py`)
1.  **Concept Probes ($\mathbf{P}$):**
      * システムは $K$ 個の「概念プローブ（基底ベクトル）」を持つ。
2.  **Active Resonance:**
      * 入力 $X$ とプローブ群 $\mathbf{P}$ の共鳴度（内積）を計算する。
      * **Lateral Inhibition (側方抑制):** Softmax温度パラメータ $\tau$ を低く設定し、最も強いプローブ（勝者）を際立たせる。
3.  **Confidence Gating (Anti-Hallucination):**
      * 最大共鳴度が閾値 $\theta_{gate}$ を下回る場合、入力を「ノイズ（雑念）」とみなし、処理を遮断（Gate Closed）する。
4.  **$S_0$ Slice Generation:**
      * 勝者プローブ $p_{win}$ をQueryとして入力 $X$ をAttentionで切り取り、初期状態 $S_0$ を生成する。

### Vicāra (Recurrent Refinement)

**機能:** 外部入力を遮断し、内部状態を再帰的に純化する。

*   **Interface:** `BaseVicara` (`src/components/vicara/base.py`)
*   **Implementations:**
    *   *StandardVicara:* 単一の汎用Refiner ($\\Phi$) を共有する。
    *   *WeightedVicara:* 複数のRefinerの重み付き和を使用。
    *   *ProbeSpecificVicara:* 各概念プローブ $p_k$ に対応する専用のRefiner ($\\Phi_k$) を持つ。

1.  **Isolation:** $t > 0$ において、外部入力 $X$ へのゲートを閉じ、自己ループのみにする。
2.  **Refinement Loop:**
      * **Hard Attention Mode (Inference):** 勝者プローブに対応するRefiner $\\Phi_{win}$ のみを適用する。
          * $S_{t+1} = \\Phi_{win}(S_t)$
      * **Soft Attention Mode (Training):** 全プローブの確率分布に基づく重み付き和で更新する（勾配伝播のため）。
          * $S_{t+1} = \sum_k w_k \\Phi_k(S_t)$
3.  **Convergence Check:**
      * 状態変化量 $||S_{t+1} - S_t||$ が $\\epsilon$ 未満になった時点で「Appanā (没入)」とみなし、推論を終了する。

### Refiner (Internal Dynamics)

**機能:** 潜在空間内での状態遷移（力学系）を定義する、Vicāra内部の実行ユニット。

*   **Interface:** `BaseRefiner` (`src/components/refiners/base.py`)
*   **Implementations:**
    *   *MlpRefiner:* 全結合層と活性化関数によるシンプルな状態更新。
    *   *GruRefiner:* (Future) GRUセルを用いた、記憶を持つ状態更新。
    *   *AttentionRefiner:* (Future) Self-Attentionを用いた状態間の関係性整理。

### Sati-Sampajañña (Meta-Cognition & Logging)

**機能:** システムの挙動を「因果的な物語」として記録し、説明可能性 (XAI) を担保する。

1.  **Probe Log (瞬間の気づき):** そのステップで何が選ばれたか。
2.  **Cetanā Dynamics (時間の流れ):** 前ステップとの比較による、意図の遷移（持続、転換、拡散）の追跡。

### Decoder (Expression - Output Reconstruction)

**機能:** 収束・純化された潜在状態 $S_{final}$ を、元の入力形式やターゲット形式に復元・変換する。

*   **Role:** 内的な洞察（Insight）を、外的な表現（Expression）へと戻す。
*   **Interface:** `BaseDecoder` (`src/components/decoders/base.py`)
*   **Implementations:**
    *   *ReconstructionDecoder:* $S_{final}$ を元の入力次元に戻す（Autoencoder用）。
    *   *CnnDecoder:* 画像再構成用。
    *   *LstmDecoder / SimpleSequenceDecoder:* 系列再構成用。
*   **Output:** 再構成データ $\hat{X}$ または予測値 $Y$。

-----

## 3. 数理モデル (Mathematical Formulation)

### 3.1. Vitakka Phase (Resonance)

入力 $X \in \mathbb{R}^{L \times d}$ に対するプローブ $p_k$ の共鳴スコア $R_k$:

$Score_k = || \frac{1}{\\sqrt{d}} \sum_{i=1}^{L} \text{Softmax}(p_k^T x_i) \cdot x_i ||$

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

学習時の目的関数は、選択された `Objective` によって決定される。基本形は以下の通り:
$\mathcal{L} = \underbrace{|| S_{T} - S_{T-1} ||^2}_{Stability} + \lambda_1 \underbrace{\sum |S_T|}_{Sparsity} - \lambda_2 \underbrace{I(S_T; S_0)}_{Info Retention}$

-----

## 4. データ構造仕様 (Data Structures)

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

時間的な意図の流れを記述するログ。

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

## 5. 処理フロー (Algorithm Flow)

1.  **Input:** データ $X$ を取得。
2.  **SamadhiEngine.forward(x, run_vitakka=True, run_vicara=True):**
    *   **Adapter:** $z = \text{Adapter}(x)$
    *   **Vitakka (Optional):** $s_0, \text{meta} = \text{Vitakka}(z)$
        *   Gate Decision (Threshold Check)
    *   **Vicāra (Optional):**
        *   Loop $t=1 \dots N$:
            *   $S_{next} = \Phi(S_{curr})$
            *   Update State with Inertia
    *   **Decoder:** $\text{Output} = \text{Decoder}(S_{final})$
3.  **Output:** 収束した $S_{final}$, デコーダ出力, および `Logs` を出力。

-----

## 6. パラメータ設定推奨値 (Hyperparameters)

コード内の `config` 辞書で使用されるキーと、推奨される設定値の分類。

### モデル・アーキテクチャ (Model Architecture)
| Key | Symbol | Recommended Value | Description |
| :--- | :--- | :--- | :--- |
| **`dim`** | $d$ | 64 - 512 | 潜在状態ベクトルの次元数。 |
| **`input_dim`** | $D_{input}$ | - | 入力データの次元。 |
| **`seq_len`** | $L$ | 10 - 60 | *(時系列モデルのみ)* シーケンス長。 |
| **`n_probes`** | $K$ | 16 - 64 | 概念プローブの数。 |
| **`vicara_type`** | - | `"probe_specific"` | `"standard"` (共有) か `"probe_specific"` (個別) か。 |
| **`probe_trainable`** | - | `True` | プローブ自体を学習するか。 |
| **`adapter_hidden_dim`** | $D_{hidden}$ | 256 | アダプター内の隠れ層の次元。 |

### Vitakka (Search)
| Key | Symbol | Recommended Value | Description |
| :--- | :--- | :--- | :--- |
| **`gate_threshold`** | $\theta$ | 0.3 - 0.5 | 妄想（ノイズ）を弾く強度。 |
| **`softmax_temp`** | $\tau$ | 0.1 - 0.2 | 低いほど「一境性（単一テーマ）」を選び取る。 |

### Vicara (Refinement)
| Key | Symbol | Recommended Value | Description |
| :--- | :--- | :--- | :--- |
| **`refine_steps`** | $T_{max}$ | 5 - 10 | 再帰的精製ステップ数。 |
| **`inertia`** | $\beta$ | 0.7 | 状態更新の慣性。 |

### Training (Objective Params)
| Key | Symbol | Recommended Value | Description |
| :--- | :--- | :--- | :--- |
| **`stability_coeff`** | $\lambda_{stab}$ | 0.01 | 状態の不動化を促す強さ。 |
| **`entropy_coeff`** | $\lambda_{ent}$ | 0.1 | 曖昧な検索結果を罰する強さ。 |
| **`balance_coeff`** | $\lambda_{bal}$ | 0.001 | プローブの使用頻度を均一化する。 |
| **`anomaly_margin`** | `5.0` | - | *(AnomalyObjective)* 異常データのマージン。 |
| **`anomaly_weight`** | `1.0` | - | *(AnomalyObjective)* 異常データに対するペナルティの重み。 |

-----

## 7. 既存モデルとの比較 (Comparison)

| 特徴 | Transformer (GPT) | **Samadhi Framework** |
| :--- | :--- | :--- |
| **基本動作** | 次トークンの予測 (発散) | 状態の純化・不動化 (収束) |
| **時間依存性** | 履歴(Context Window)に依存 | 現在の状態(State)のみに依存 (Markov) |
| **アテンション** | Self-Attention (Token間) | Recursive Attention (State-Probe間) |
| **推論コスト** | $O(N^2)$ (文脈長で増大) | $O(1)$ (定数・収束ステップ数のみ) |
| **説明可能性** | 低い (Attention Mapのみ) | **極めて高い (Probe/Cetanā Log)** |
| **哲学的基盤** | 連想・生成 | **禅定・洞察** |

-----

## 8. 応用と学習戦略 (Applications & Training Strategies)

Samadhi Frameworkは、**学習戦略（Trainer + Objective）**と**デコーダー（Decoder）**の組み合わせにより、異なるタスクに適用可能である。

| 応用タスク | Objective | Decoder Role | 目的関数 (Loss) |
| :--- | :--- | :--- | :--- |
| **構造発見 / クラスタリング**<br>(Unsupervised) | `UnsupervisedObjective` | **Identity** | Stability + Entropy + Sparsity<br>(内部状態の安定化のみを追求) |
| **オートエンコーダ事前学習**<br>(Pre-training) | `AutoencoderObjective` | **Reconstruction** | Reconstruction Loss Only<br>(入力の復元誤差最小化、Vicaraスキップ) |
| **異常検知**<br>(Anomaly Detection) | `AnomalyObjective` | **Reconstruction** | Recon + Stability + Margin<br>(正常データの復元と異常データの排除) |
| **教師ありタスク**<br>(分類/回帰) | `SupervisedObjective` | **Classifier** / **Regressor** | CrossEntropy / MSE + Stability<br>(ターゲット予測) |

*   **Meditation Mode (Unsupervised):** 外界の正解に頼らず、データ内在の構造（Dharma）を見出す。
*   **Expression Mode (Supervised/Anomaly):** 見出した構造を利用して、外界のタスク（分類、検知）を解く。

-----

## 9. 大規模言語モデル (LLM) との連携 (Integration with LLMs)

Samadhiの「収束・安定化」と、LLMの「生成・発散」は補完的である。

*   **LLM (Generator):** 発散的思考、トークン予測、文脈生成を担当。
*   **Samadhi (Stabilizer):** 収束的思考、状態純化、意図の固定を担当。

主な連携:
1.  **意図の安定化 (Intent Stabilization):** LLMの出力をSamadhiで純化し、ブレのない一貫した対話を実現。
2.  **プロンプト強化 (Prompt Refinement):** ユーザー入力を純化し、LLMへの明確な指示ベクトルとする。
3.  **出力検証 (Output Verification):** 収束度（Stability Score）を用いて、LLMの幻覚（Hallucination）を検知。

-----

## 10. アーキテクチャの拡張性 (Architectural Extensibility)

`SamadhiBuilder` や `presets` を使用して、コンポーネントを自由に組み合わせることができる。

### 10.1. Task-Specific Customization Example

| タスク | Adapter | Refiner | Decoder | Objective |
| :--- | :--- | :--- | :--- | :--- |
| **時系列異常検知** | LSTM | MLP | Reconstruction | AnomalyObjective |
| **画像分類** | CNN | MLP | Classification | SupervisedObjective |
| **対話意図推定** | Transformer | Attention | Classification | SupervisedObjective |
| **ロボット制御** | Sensor Fusion | MLP | Action | RL (PPO) |
