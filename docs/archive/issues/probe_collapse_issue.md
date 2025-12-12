# Probe Collapse Issue & Structural Analysis

**Status:** Investigating
**Date:** 2025-11-29

-----

## 1. 現状の構成 (Current Configuration)

現在、Samadhi Model v2.3 では以下の構成で「概念の選択」と「状態の純化」を行っている。

* **Vitakka (Search):**
  * 入力 $X$ と $K$ 個の Probe $P$ の内積（Cosine Similarity）で共鳴度を計算。
  * `Softmax` (with temperature) で確率分布 $w$ を生成。
  * `Hard` モードでは `argmax` で勝者を選択。`Soft` モードでは加重平均。
* **Vicāra (Refinement):**
  * **Standard Vicāra:** 単一の Refiner $\Phi$ を全 Probe で共有。
  * **Probe Vicāra:** Probe ごとに独立した Refiner $\Phi_k$ を保持。
  * 学習時は `Soft` モード、推論時は `Hard` モードへの切り替えを想定。

-----

## 2. 課題点 (The Issue)

**「Probe Collapse（プローブの縮退）」**
特定の単一 Probe（例えば Index 0）のみが常に選択され、他の Probe が死滅（Dead Probes）する現象が発生している。

* 初期化直後から、ある特定の Probe だけが勝率 99% 以上を占める。
* 学習が進んでも、他の Probe に役割が分散されない。
* Probe を固定（Fix）しても、入力に対して適切な Probe が選ばれず、やはり特定の Probe が汎用的に使われ続ける。

結果として、「概念による分業」が行われず、単なる「汎用 Autoencoder」として振る舞ってしまっている。

-----

## 3. 原因分析 (Root Cause Analysis)

なぜこの現象が起きるのか、3つの観点から分析する。

### 3.1. 数学的観点 (Mathematical Perspective)

**「汎用 Refiner の罠 (The Generic Refiner Trap)」**

* **勾配の「楽な道」:** 初期段階で偶然少しだけ反応が良かった Probe A があるとする。Optimizer は Probe A を通るパスの Loss を下げようと、Refiner A（または共有 Refiner）を急速に最適化する。
* **Rich-get-richer:** Refiner A が賢くなると、どんな入力が来ても Probe A を通せばそこそこ復元できるようになる。
* **勾配消失:** 使われない Probe B, C は勾配が流れてこないため、学習が進まず、永久に「使えない」ままとなる。
* **Softmax の排他性:** Temperature が低い場合、少しのスコア差が確率 1.0 vs 0.0 に増幅されるため、一度差がつくと逆転不可能になる。

### 3.2. 計算科学的観点 (Computational Perspective)

**「タスクの単純さとモデル容量の過剰」**

* 現在テストしている MNIST などのタスクは単純であり、単一の Refiner $\Phi$ の表現能力だけで十分に全パターン（0~9）を記憶・復元できてしまう。
* モデルにとって「わざわざ Probe を切り替える」という複雑な制御を獲得するよりも、「一つの強力なネットワークで力押しする」方が、Loss を下げるための最短経路（局所解）となっている。
* **インセンティブの欠如:** 「間違った Probe を選んだら復元できない」という罰則がないため、正しい選択をする動機がない。

### 3.3. 仏教的観点 (Buddhist Philosophy Perspective)

**「痴 (Moha) と 執着 (Upādāna)」**

* **無明 (Avijjā):** システムが「入力の本質的な特徴（相）」を正しく識別できていない。Vitakka が粗雑であるため、対象を区別できず、全てを同一視している。
* **執着 (Upādāna):** 過去に成功した（Lossが下がった）経験（特定のProbe）に固執し、新しい可能性（他のProbe）を探索しようとしない心の癖（習性）。
* **正見 (Sammā-diṭṭhi) の欠如:** 「呼吸は呼吸として、痛みは痛みとして」正しく見るためのフィードバック機構が欠けている。

### 3.4. コード解析からの発見 (Code Analysis Findings: Unsupervised Flaw)

コード（`UnsupervisedSamadhiTrainer`）の確認により、以下の重大な設計ミスが判明した。

* **目的関数の欠陥:** 現在の非教師あり学習は `Stability Loss`（変化しないこと）と `Entropy Loss`（迷わないこと）のみを最小化している。
* **自明な解 (Trivial Solution):** この設定では、「常に入力をそのまま出力する（恒等写像）」かつ「常に同じProbeを選ぶ」ことが、最も簡単にLossを0にする方法となる。
* **学習の放棄:** 入力を復元する（Reconstruction）義務がないため、モデルは「有意義な特徴」を学習する必要性が全くない。これが Probe Collapse を加速、あるいは固定化させている。

-----

## 4. 解決策 (Proposed Solutions)

### Solution A: Load Balancing Loss (負荷分散ロス) 【推奨・即効性高】

Mixture of Experts (MoE) で用いられる手法。Probe の選択確率が「偏る」こと自体にペナルティを与える。

$$ \mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_{balance} \cdot CV(Prob_{batch}) $$
(CV: 変動係数、またはエントロピーの最大化)

* **効果:** 無理やり全ての Probe を使わせるため、死んでいる Probe にも勾配が流れるようになる。
* **実装難易度:** 低。Loss 関数に項を追加するだけ。
* **仏教的解釈:** **「捨 (Upekkhā - 平等心)」**。特定の対象（Probe）に偏ることなく、全ての可能性を公平に扱う心のバランス。

**実験結果 (2025-11-29):**
`test/test_trainer_mnist.py` において、以下の結果が得られた。

* 初期の `balance_coeff=0.001` と `attention_mode="hard"` の組み合わせでは、**Probe Collapse** は解消されず、単一のProbeに集中した利用が見られた。
* `balance_coeff` を `10.0` に**大幅に増加**させ、かつ学習時の `attention_mode` を `"soft"` に変更したところ、**全てのProbeがアクティブ**になり、使用率が約7%〜14%に**均等に分散**され、**Probe Collapse は見られなくなった**。
* ただし、推論時のReconstruction結果は、KMeansによるProbe初期化が意味的な数字ラベルと直接一致しないため、依然として**誤った数字として復元されるケースが確認された**。

### Solution B: Probe-Specific Hard Vicara (強制的な専門化) 【構造改革】

`ProbeVicara` を使用し、学習中もあえて `Hard` (または Gumbel-Softmax のような Sharp な分布) に近づける。

* **ロジック:** 「Probe A を選んだら Refiner A しか使えない」状況を強制する。
* **効果:** Refiner A が「数字の1」しか書けないように初期化されていれば、「8」の画像に Probe A を選ぶと復元に大失敗する（Loss が爆増する）。これにより、モデルは「正しく選ばないと死ぬ」ことを学習する。
* **実装難易度:** 中。`attention_mode` の調整と初期化の工夫が必要。
* **仏教的解釈:** **「精進 (Viriya - 正しい努力)」**。誤った対象（Probe）に努力を費やすのではなく、正しい対象に対してのみ精進するよう強制する。

### Solution C: Vitakka Adapter (入力変換器)

Vitakka 内に軽量な MLP (Adapter) を導入し、入力を Probe 空間に射影してから比較する。

* **ロジック:** 単純なピクセル内積では概念の類似度が測れていない可能性がある。非線形変換を挟むことで、より抽象的なマッチングを可能にする。
* **効果:** Probe 選択の精度向上。
* **実装難易度:** 低。`Vitakka` クラスの修正。
* **仏教的解釈:** **「如理作意 (Yoniso Manasikāra - 適切な注意の向け方)」**。現象の表面（ピクセル）ではなく、その本質（意味空間）に合わせて心を向ける働き。

### Solution D: Orthogonality Constraint (直交性制約)

Probe ベクトル同士が似通らないように強制する。

$$ \mathcal{L}_{ortho} = || P^T P - I ||_F $$

* **効果:** Probe が互いに異なる方向を向くようになり、役割分担を促す。
* **実装難易度:** 低。
* **仏教的解釈:** **「種々性 (Nānatta - 多様性の認識)」**。諸行は無常であり、それぞれが異なる相（特徴）を持っていることを混同せずに認識する。

### Solution E: Fix Unsupervised Objective (非教師あり目的関数の修正) 【必須】

`UnsupervisedSamadhiTrainer` に `Reconstruction Loss` (入力 $X$ 自身の復元) を導入する。

* **ロジック:** モデルに「入力情報を保持する」責任を持たせることで、Stability Loss（手抜き）への逃げ道を塞ぐ。
* **効果:** 有意義な特徴抽出（オートエンコーダとしての機能）の保証。
* **実装難易度:** 低。
* **仏教的解釈:** **「諦 (Sacca - 真理/事実)」**。あるがままの現実（入力 $X$）から目を背けず、それを正しく認識・受容する責任。事実を無視した「空（Stability）」は虚無である。

### Solution F: Noise Injection (Router Stabilization) 【探査促進】

Vitakka のスコア計算時に、ガウシアンノイズを加算してから Softmax/Argmax を適用する。
`logits = raw_scores + noise`, `noise ~ N(0, sigma)`

* **ロジック:** 決定論的な選択を防ぎ、学習初期に全ての Probe が選ばれるチャンスを作る（Exploration）。
* **効果:** 初期収束（Collapse）の回避。MoE で標準的に使用される手法。
* **実装難易度:** 低。`Vitakka` クラスの修正のみ。
* **仏教的解釈:** **「疑 (Vicikicchā) の善用」**。通常は障害だが、ここでは「確信のなさ」を逆手に取り、固定観念（初期の偏り）に囚われず、広く可能性を探求する力として利用する。

### Solution G: Dead Probe Reset (死にプローブの蘇生) 【強力な介入】

VQ-VAE (K-means) で用いられる手法。一定期間（例: 1エポック）使われなかった Probe を検出し、強制的に再初期化する。
再初期化先は「現在のバッチ内のランダムな入力」や「頻出 Probe の摂動」など。

* **ロジック:** 勾配が死んだ Probe を物理的に生き返らせる。
* **効果:** Collapse 状態からの確実な脱出。
* **実装難易度:** 中。Trainer にコールバック的な処理が必要。
* **仏教的解釈:** **「結生 (Paṭisandhi - 再生)」**。機能しなくなったプロセス（死んだProbe）を捨て、新たな縁（入力）を得て生まれ変わらせる循環。

### Solution H: Orthogonal Regularization (直交化)

Probe ベクトル同士が似通らないように強制する。
$$ \mathcal{L}_{ortho} = || P P^T - I ||_F $$

* **ロジック:** 異なる Probe は異なる方向を向くべきという幾何学的制約。
* **効果:** 役割分担の促進。
* **実装難易度:** 低。
* **仏教的解釈:** **「遠離 (Viveka - 分離/孤立)」**。概念同士が癒着・混同せず、それぞれが独立して清浄に保たれている状態。

### Solution I: Expert Network Regularization (エキスパートネットワーク正則化) 【表現の多様性】

各 Probe に紐づく Refiner ネットワーク ($\\Phi_k$) に個別に Weight Decay (L2正則化) や Dropout を適用する。

* **ロジック:** 個々の Refiner が過度に強力になったり、他の Refiner と機能的に重複する表現を学習したりするのを防ぐ。
* **効果:** リファイナー間の機能的分化を促し、表現の縮退 (Representation Collapse) を防ぐ。
* **実装難易度:** 低〜中。各 Refiner (例えば `ProbeVicara` 内の `self.refiners` の要素) に対して直接適用するか、Trainer の最適化ループで正則化項を追加する。
* **仏教的解釈:** **「戒 (Sīla - 自律/規律)」**。過剰な能力や暴走（Overfitting）を抑制し、本来の役割（専門性）に留まるよう自らを律する働き。

-----

## 6. Attention Strategy Analysis (Soft vs Hard vs Hybrid)

学習時の Attention 戦略は、Probe Collapse の発生リスクと専門化の度合いに大きく影響する。

| 戦略 | メリット | デメリット | Collapse リスク |
| :--- | :--- | :--- | :--- |
| **Soft Attention** | ・微分可能で学習が安定。<br>・全プローブに勾配が流れる。 | ・計算コストが高い。<br>・「混ぜる」ことで専門化が阻害されやすい (Representation Collapse)。 | **中**<br>重みの偏りによる「実質的な死」が発生しうる。 |
| **Hard Attention** (Argmax) | ・計算効率が良い。<br>・強い専門化を促す。 | ・微分不可能。<br>・選ばれないプローブが即死する。 | **高**<br>Load Balancing なしでは即座に Collapse する。 |
| **Gumbel-Softmax** | ・微分可能な Hard 近似。<br>・Forward は Hard、Backward は Soft (Straight-Through)。 | ・勾配の分散が大きく不安定になりがち。 | **中〜高**<br>専門化と学習可能性のバランスが良いが、補助ロスは必須。 |
| **Annealing** (Soft→Hard) | ・初期探索 (Soft) と終盤の専門化 (Hard) を両立。 | ・スケジュール調整が難しい。 | **低 (推奨)**<br>最も安全かつ効果的。 |

-----

## 7. 比較と推奨方針

| 手法 | 実装コスト | 期待効果 | 副作用 | 推奨度 |
| :--- | :--- | :--- | :--- | :--- |
| **Load Balancing** | 低 | 高 (強制分散) | ハイパーパラメータ $\lambda$ の調整が必要 | ★★★ |
| **Fix Unsupervised** | 低 | 高 (学習の前提) | なし（必須修正） | ★★★ |
| **Noise Injection** | 低 | 中 (初期探索) | 推論時はオフにする必要あり | ★★★ |
| **Expert Network Reg.** | 低〜中 | 中 (機能分化) | 正則化強度の調整が必要 | ★★★ |
| **Gumbel/Annealing** | 中 | 高 (専門化) | 学習安定性の調整が必要 | ★★☆ |
| **Dead Probe Reset** | 中 | 絶大 (強制蘇生) | 学習曲線が一時的に跳ねる | ★★☆ |
| **Orthogonality** | 低 | 中 (分化促進) | 解の自由度を下げる可能性 | ★☆☆ |
| **Hard Training** | 中 | 高 (構造的必然) | 学習が不安定になりやすい | ★★☆ |
| **Adapter** | 低 | 中 (精度向上) | パラメータ数微増 | ★★☆ |

**結論:**
以下の5点を順次実装・検証する。

1. **Solution A (Load Balancing Loss):** 全Trainer共通で、Probeの偏りを罰する項を追加（最優先）。
2. **Solution E (Fix Objective):** `UnsupervisedSamadhiTrainer` に `Reconstruction Loss` を追加（必須）。
3. **Solution F (Noise Injection):** Vitakka にノイズ注入機能を追加し、学習初期の多様性を確保。
4. **Solution I (Expert Network Regularization):** 各リファイナーに正則化を適用。
5. **Attention Strategy:** `Soft` から開始し、安定したら `Gumbel-Softmax` または `Annealing` への移行を検討する。

これでも改善しない場合、**Solution G (Dead Probe Reset)** の導入を検討する。
