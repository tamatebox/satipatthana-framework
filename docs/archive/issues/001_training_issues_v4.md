# Issue 001: v4 Training Issues (2024-12)

v4アーキテクチャの初期実装で発生した学習問題と対処法のまとめ。

## 概要

4-Stage Curriculum Trainingの実装において、以下の問題が発生した：

1. **Stage 1 (Samatha Training)**: Lossが下がらない
2. **Stage 2 (Vipassana Training)**: Lossが下がらない、Trainableパラメータが0
3. **推論結果**: 全ての入力に対して同じ出力が生成される

---

## Issue 1: Stage 1 StabilityLossの勾配消失

### 症状

- Stage 1のStabilityLossが約1.023から下がらない
- 学習が進んでも収束しない

### 原因

**SantanaLogが状態を`.detach()`して保存していた**

```python
# SantanaLog.add() の実装
def add(self, state: torch.Tensor, ...):
    self.states.append(state.detach().cpu())  # ← 勾配が切れる
```

StabilityLossは `||s_T - s_{T-1}||²` を計算するが、SantanaLogから取得した状態には勾配がないため、backpropagationが不可能だった。

### 根本原因（設計思想）

SantanaLogは**ログ/履歴**として設計されており、以下の理由でdetachが必要：

1. **メモリ効率**: 全ステップの計算グラフを保持するとGPUメモリが爆発する
2. **Vipassana用**: Stage 2でVipassanaが分析する際、Samathaは凍結されているため勾配不要
3. **可視化用**: 軌跡の可視化やデバッグにはdetachされた値で十分

**問題は、StabilityLossもSantanaLogから状態を取得しようとしていたこと**。

### 対処

**stability_pair**を導入し、SantanaLogとは別に勾配付きの状態ペアを保持：

```python
# SamathaOutput (engines.py)
class SamathaOutput(NamedTuple):
    s_star: torch.Tensor
    santana: SantanaLog
    severity: torch.Tensor
    stability_pair: Tuple[torch.Tensor, torch.Tensor]  # (s_T, s_{T-1}) with gradients

# _run_vicara_loop内
s_prev_grad = s_t  # 勾配付きで保持
# ... loop ...
stability_pair = (s_t, s_prev_grad)  # ループ終了時に返す

# StabilityLoss.compute_loss()
def compute_loss(self, stability_pair, santana=None):
    s_T, s_T_1 = stability_pair
    diff = s_T - s_T_1
    stability_loss = torch.norm(diff, dim=1).pow(2).mean()
```

### 教訓

- **ログ用データと学習用データは分離する**
- NamedTupleのような構造体を使い、用途ごとにフィールドを明確に分ける
- `.detach()`が必要な理由を明示的にコメントで残す

---

## Issue 2: HuggingFace Trainerのオプティマイザキャッシュ

### 症状

- Issue 1を修正後もStage 1のLossが下がらない
- Stage遷移時に同じオプティマイザが使い回されている様子

### 原因

HuggingFace Trainerは`train()`呼び出し間でオプティマイザをキャッシュする。
`set_stage()`で`requires_grad`を変更しても、オプティマイザの`param_groups`は古いパラメータを参照したまま。

```python
# 問題のあるフロー
trainer.train_stage(TrainingStage.ADAPTER_PRETRAINING)  # Stage 0
trainer.train_stage(TrainingStage.SAMATHA_TRAINING)     # Stage 1
# ↑ Stage 0で作られたオプティマイザが再利用され、
#   Vitakka/Vicaraのパラメータは最適化対象に含まれていない
```

### 根本原因（設計思想）

HuggingFace Trainerは**単一タスクの学習**を想定しており、マルチステージ学習は標準的なユースケースではない。
オプティマイザのキャッシュは効率化のための設計だが、パラメータの動的な変更には対応していない。

### 対処

`train_stage()`でオプティマイザとスケジューラを明示的にリセット：

```python
def train_stage(self, stage: TrainingStage, num_epochs: int = 1, **kwargs):
    self.set_stage(stage)

    # Reset optimizer and scheduler to use new trainable parameters
    self.optimizer = None
    self.lr_scheduler = None

    # ... train() ...
```

### 教訓

- フレームワークの内部動作を理解する（特にステートフルな部分）
- マルチステージ学習では、各ステージ間でのリソースリセットを明示的に行う
- `requires_grad`の変更だけでは不十分な場合がある

---

## Issue 3: Vipassanaのlazy initialization

### 症状

- Stage 2開始時に`Trainable components: []`と表示される
- Lossが約0.66で停滞（BCEの初期値に近い）
- `vipassana._encoder`が`None`のまま

### 原因

StandardVipassanaは**lazy initialization**を採用しており、初回のforward時にネットワークを構築する：

```python
class StandardVipassana(BaseVipassana):
    def __init__(self, config):
        # ...
        self._encoder = None  # Not built yet
        self._trust_head = None

    def forward(self, s_star, santana):
        if self._encoder is None:
            self._build_networks(s_star.shape[1])  # Build on first call
```

`set_stage(TrainingStage.VIPASSANA_TRAINING)`を呼んだ時点では、まだforwardが実行されていないため、`_encoder`と`_trust_head`は`None`。結果として、`unfreeze_module()`は空のモジュールに対して実行され、学習可能なパラメータが0になる。

### 根本原因（設計思想）

lazy initializationは以下の利点がある：

1. **動的な次元対応**: 入力の次元が実行時まで不明な場合に対応
2. **設定の簡略化**: ユーザーが全ての次元を手動で指定する必要がない
3. **柔軟性**: 同じVipassanaを異なる次元のSamathaと組み合わせられる

**しかし、freeze/unfreeze操作との相性が悪い**。

### 対処

Stage 2開始前にダミーのforward passを実行してネットワークを初期化：

```python
def train_stage(self, stage: TrainingStage, num_epochs: int = 1, **kwargs):
    # For Stage 2, initialize lazy networks first
    if stage == TrainingStage.VIPASSANA_TRAINING:
        self._initialize_vipassana_networks()

    self.set_stage(stage)
    # ...

def _initialize_vipassana_networks(self):
    """Initialize Vipassana's lazy networks by running a dummy forward pass."""
    sample = self.train_dataset[0]
    x = sample["x"].unsqueeze(0).to(self.args.device)

    with torch.no_grad():
        self.model.forward_stage2(x, noise_level=0.0, drunk_mode=False)

    logger.info("Vipassana networks initialized via forward pass")
```

### 教訓

- lazy initializationを使う場合、freeze/unfreezeとの相互作用を考慮する
- 初期化タイミングをドキュメント化し、呼び出し側に明示する
- または、`build(input_dim)`のような明示的な初期化メソッドを提供する

---

## Issue 4: Vitakka gateによる入力情報の消失

### 症状

- 学習は進む（Lossは下がる）が、全ての入力に対して同じ出力画像が生成される
- `s_star`のcosine similarityが全サンプルで1.0（完全に同一）

### 原因

StandardVitakkaの**gate機構**がデフォルトで閉じていた：

```python
@dataclass
class StandardVitakkaConfig(BaseVitakkaConfig):
    gate_threshold: float = 0.6  # ← この閾値
```

gate機構は、入力とプローブの類似度が閾値未満の場合にゲートを閉じ、`s0 = 0`を返す：

```python
def forward(self, z):
    similarities = self._compute_similarities(z)
    gate = (similarities.max(dim=1).values > self.gate_threshold).float()
    # gate = 0 の場合、s0 は 0 または非常に小さな値になる
```

学習初期はプローブがランダムなため、類似度が閾値を超えることがほとんどない。
結果として、ほぼ全てのサンプルで`s0 ≈ 0`となり、Vicaraも`0`から始まって`0`に収束する。

### 根本原因（設計思想）

gate機構の意図：

- **推論時**: 未知のクラスや分布外データを検出するために、「知らない」と判断してゲートを閉じる
- **学習時**: プローブを学習させて、既知のデータには高い類似度を持たせる

**問題は、学習初期にゲートが閉じると勾配が流れず、プローブが学習できないこと**。

### 対処

Notebookで`gate_threshold=-1.0`を設定し、学習中はゲートを常に開放：

```python
vitakka_config = StandardVitakkaConfig(
    dim=64,
    n_probes=10,
    gate_threshold=-1.0,  # 学習中は常にゲートを開く
)
```

デフォルト値（0.6）は推論時のユースケースを考慮して維持。

### 教訓

- 学習時と推論時で異なる振る舞いをするコンポーネントは、その切り替え方法を明確にする
- デフォルト値は「安全側」ではなく「学習が進む側」に設定するべき
- または、`training_gate_threshold`と`inference_gate_threshold`を分離する

---

## 再発防止策

### 1. チェックリスト（新しいStageを追加する時）

- [ ] 対象コンポーネントの初期化タイミングを確認
- [ ] `requires_grad`の変更がオプティマイザに反映されるか確認
- [ ] 勾配が最終的なlossまで流れるかを`grad_fn`で確認
- [ ] 学習初期の状態（ゲート、初期化値など）が学習を阻害しないか確認

### 2. デバッグ手順（Lossが下がらない時）

1. **Trainable componentsの確認**

   ```python
   print(system.get_trainable_params())
   ```

2. **勾配の有無を確認**

   ```python
   loss.backward()
   for name, param in model.named_parameters():
       if param.requires_grad and param.grad is None:
           print(f"No gradient: {name}")
   ```

3. **中間状態の確認**

   ```python
   # s_starが多様か確認
   cos_sim = F.cosine_similarity(s_star[0:1], s_star[1:], dim=1)
   print(f"Cosine similarity: {cos_sim.mean():.4f}")
   ```

4. **各コンポーネントの出力確認**

   ```python
   z = adapter(x)
   s0, meta = vitakka(z)
   print(f"z std: {z.std():.4f}, s0 std: {s0.std():.4f}")
   ```

---

## Issue 5: Stage 2のTrust Scoreがバッチ内で同一値になる

### 症状

- Stage 2学習後、推論時にTrust scoreがバッチ内の全サンプルで同一値（例：全部0.5）
- Clean path と Drunk path で差が出ない

### 原因

当初の実装では、trust_headへの入力に対して正規化を行っていたが、**バッチ全体でのスケール正規化**により、サンプル間の差異が消えていた。

また、trust_headが`s_star`を含む全特徴量（34次元）を入力としていたため、収束品質（velocity, energy）の信号がs_starの次元に埋もれていた。

### 対処

1. **trust_headをs_starから分離**: 入力を`[velocity, avg_energy]`の2次元のみに限定
2. **log1p変換**: スケール正規化しつつper-sample差異を保持

```python
# 修正前
self._trust_head = nn.Sequential(
    nn.Linear(feature_dim, self.hidden_dim // 2),  # feature_dim = state_dim + 2
    ...
)
trust_score = self._trust_head(features)  # features = [s_star, velocity, energy]

# 修正後
trust_feature_dim = 2  # velocity + avg_energy only
self._trust_head = nn.Sequential(
    nn.Linear(trust_feature_dim, self.hidden_dim // 2),
    ...
)
trust_features = torch.cat([
    torch.log1p(velocity),
    torch.log1p(avg_energy_tensor)
], dim=1)
trust_score = self._trust_head(trust_features)
```

### 教訓

- バッチ正規化やLayerNormは、サンプル間の相対的な差異を消す可能性がある
- 「何を測りたいか」に応じて、入力特徴量を適切に分離する
- v_ctx（コンテキスト）とtrust_score（信頼度）は異なる目的を持つため、入力を分けるべき

---

## Issue 6: Stage 2のLossが下がらない（学習率）

### 症状

- Stage 2のBCE Loss が 0.65 → 0.63 程度で停滞
- Trust scoreがClean/Drunk間で差が出ない

### 原因

HuggingFace Trainerのデフォルト学習率（5e-5）がStage 2には低すぎた。

Stage 2では Vipassana のみが学習対象で、パラメータ数が少ない（trust_head: 2→32→1）。
小さなネットワークに対して学習率が低すぎると、収束が遅くなる。

### 対処

Stage 2の学習率を1e-2に引き上げ：

```python
# Notebook側で
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    learning_rate=1e-2,  # Stage 2用に引き上げ
    ...
)
```

### 結果

- Trust score差: Clean 0.31 vs Drunk 0.13 → Clean 0.56 vs Drunk 0.07

### 教訓

- Stage毎に最適な学習率は異なる
- パラメータ数が少ないモジュールには高めの学習率が必要
- マルチステージ学習では、各ステージで`TrainingArguments`を調整することを検討

---

## Issue 7: Stage 2にClean Pathがなかった

### 症状

- Trust scoreの学習が不安定
- 「高信頼」の基準が曖昧

### 原因

training_strategy.mdでは4-way split（Clean 30%, Augmented 40%, Drunk 15%, Mismatch 15%）を推奨しているが、
実装は3-way split（Augmented, Drunk, Mismatchのみ）だった。

Clean path（ノイズなし、target=1.0）がないと、Vipassanaは「高信頼とは何か」の明確な基準を学習できない。

```python
# 修正前: 3-way split
split_size = batch_size // 3
# 1. Augmented (target: 1.0 - severity)
# 2. Drunk (target: 0.0)
# 3. Mismatch (target: 0.0)

# 修正後: 4-way split
split_size = batch_size // 4
# 1. Clean (target: 1.0)  ← 追加
# 2. Augmented (target: 1.0 - severity)
# 3. Drunk (target: 0.0)
# 4. Mismatch (target: 0.0)
```

### 対処

trainer.pyの`_compute_stage2_loss`にClean pathを追加：

```python
# 1. Clean Path (No noise - baseline for high trust)
if sizes[0] > 0:
    x_clean = x_splits[0]
    result_clean = model.forward_stage2(x_clean, noise_level=0.0, drunk_mode=False)
    trust_clean = result_clean["trust_score"]
    target_clean = torch.ones_like(trust_clean)  # target: 1.0

    all_trust_scores.append(trust_clean)
    all_targets.append(target_clean)
```

### 教訓

- ドキュメント（training_strategy.md）と実装の整合性を確認する
- 対照学習では「正例」と「負例」の両方が明確に必要
- Clean pathは「これが高信頼の基準」という教師信号を与える

---

## Issue 8: avg_energyがバッチ全体で同一値だった

### 症状

- Trust scoreがサンプル間で差が出にくい
- デバッグでavg_energyを確認すると全サンプルで同じ値

### 原因

`SantanaLog.get_total_energy()`はバッチ全体の合計エネルギーを返していた。
これをnum_stepsで割ってもバッチ全体の平均であり、**per-sample**のエネルギーではなかった。

```python
# 修正前
total_energy = santana.get_total_energy()  # バッチ全体の合計
avg_energy = total_energy / num_steps      # スカラー値
avg_energy_tensor = torch.full((batch_size, 1), avg_energy, ...)  # 全サンプル同じ値
```

### 対処

SantanaLogのstatesから直接per-sampleエネルギーを計算：

```python
# 修正後
if num_steps >= 2:
    states_tensor = santana.to_tensor()  # (num_steps, batch_size, dim)
    state_diffs = states_tensor[1:] - states_tensor[:-1]  # (num_steps-1, batch, dim)
    per_sample_energy = (state_diffs ** 2).sum(dim=2).sum(dim=0)  # (batch,)
    avg_energy_tensor = (per_sample_energy / (num_steps - 1)).unsqueeze(1)  # (batch, 1)
```

### 教訓

- バッチ処理では「バッチ全体の統計」と「サンプル毎の統計」を区別する
- デバッグ時は中間値のshapeと値を確認する
- ユーティリティメソッドの戻り値の意味を正確に把握する

---

## Issue 9: ConvTranspose2dによるチェッカーボードアーティファクト

### 症状

- 画像再構成時にチェッカーボード状のノイズパターンが出現
- 特にエッジ部分で顕著

### 原因

`ConvTranspose2d`（転置畳み込み）はstride > 1の場合、出力ピクセルの重なり方が不均一になり、
チェッカーボードパターンを生成することが知られている。

### 対処

`Upsample + Conv2d`パターンに変更：

```python
# 修正前
nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),

# 修正後
nn.Upsample(scale_factor=2, mode="nearest"),
nn.Conv2d(256, 128, kernel_size=3, padding=1),
```

### 参考

- [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)

### 教訓

- 画像生成タスクでは`ConvTranspose2d`より`Upsample + Conv2d`が安定
- 視覚的なアーティファクトはモデル出力を必ず目視確認して発見する

---

## Issue 10: Probe Mode Collapse（全サンプルが同一Probeに収束）

### 症状

- 学習後、全サンプルのs_starが類似（cosine similarity ≈ 1.0）
- 特定のProbeにのみ収束し、他のProbeが使われない

### 原因

Vitakkaのプローブが多様性を失い、似たような方向を向いてしまう。
結果として、どの入力も同じProbeに最も近くなり、同じs0から開始してしまう。

### 対処

`ProbeDiversityLoss`を追加し、Probe間のcosine similarityを最小化：

```python
class ProbeDiversityLoss:
    def compute_loss(self, probes: torch.Tensor):
        probes_norm = probes / (probes.norm(dim=1, keepdim=True) + 1e-8)
        similarity_matrix = torch.mm(probes_norm, probes_norm.t())
        mask = ~torch.eye(n_probes, dtype=torch.bool, device=probes.device)
        off_diagonal = similarity_matrix[mask]
        diversity_loss = off_diagonal.mean()  # 小さいほど多様
        return diversity_loss, {...}
```

trainer.pyに`diversity_weight`パラメータを追加：

```python
trainer = SatipatthanaTrainer(
    ...,
    diversity_weight=0.1,  # Probe多様性損失の重み
)
```

### 教訓

- 複数のアンカー（Probe）を使う場合、多様性を明示的に促進する必要がある
- 正則化項として多様性損失を追加することで、mode collapseを防ぐ

---

## Issue 11: Drunk Modeのパラメータがハードコードされていた

### 症状

- Drunk modeの効果が弱い/強すぎる
- チューニングが困難

### 原因

drunk_modeのパラメータ（スキップ確率、摂動の標準偏差）がコード内にハードコードされていた：

```python
# 修正前
if self._drunk_mode and torch.rand(1).item() < 0.2:  # ハードコード
    ...
s_t = s_t + torch.randn_like(s_t) * 0.05  # ハードコード
```

### 対処

SamathaConfigにパラメータを追加し、設定可能に：

```python
# system.py
@dataclass
class SamathaConfig(BaseConfig):
    ...
    drunk_skip_prob: float = 0.3
    drunk_perturbation_std: float = 0.2

# engines.py
skip_prob = getattr(self.config, "drunk_skip_prob", 0.3)
if self._drunk_mode and torch.rand(1).item() < skip_prob:
    ...
perturbation_std = getattr(self.config, "drunk_perturbation_std", 0.2)
s_t = s_t + torch.randn_like(s_t) * perturbation_std
```

### 教訓

- マジックナンバーは設定ファイルに外出しする
- 実験で調整が必要なパラメータは最初からConfigに含める
- デフォルト値は`getattr`で後方互換性を保つ

---

## 再発防止策

### 1. チェックリスト（新しいStageを追加する時）

- [ ] 対象コンポーネントの初期化タイミングを確認
- [ ] `requires_grad`の変更がオプティマイザに反映されるか確認
- [ ] 勾配が最終的なlossまで流れるかを`grad_fn`で確認
- [ ] 学習初期の状態（ゲート、初期化値など）が学習を阻害しないか確認

### 2. デバッグ手順（Lossが下がらない時）

1. **Trainable componentsの確認**

   ```python
   print(system.get_trainable_params())
   ```

2. **勾配の有無を確認**

   ```python
   loss.backward()
   for name, param in model.named_parameters():
       if param.requires_grad and param.grad is None:
           print(f"No gradient: {name}")
   ```

3. **中間状態の確認**

   ```python
   # s_starが多様か確認
   cos_sim = F.cosine_similarity(s_star[0:1], s_star[1:], dim=1)
   print(f"Cosine similarity: {cos_sim.mean():.4f}")
   ```

4. **各コンポーネントの出力確認**

   ```python
   z = adapter(x)
   s0, meta = vitakka(z)
   print(f"z std: {z.std():.4f}, s0 std: {s0.std():.4f}")
   ```

### 3. 設計原則

1. **関心の分離**: ログ用データ（SantanaLog）と学習用データ（stability_pair）は分ける
2. **明示的な初期化**: lazy initializationを使う場合は、明示的な`build()`メソッドも提供
3. **学習フレンドリーなデフォルト**: デフォルト設定は「学習が進む」側に寄せる
4. **ステートのリセット**: マルチステージ学習では各ステージ間でステートをクリアにリセット

---

## 関連ファイル

- [satipatthana/core/engines.py](../../satipatthana/core/engines.py) - SamathaOutput, stability_pair, drunk_mode
- [satipatthana/core/santana.py](../../satipatthana/core/santana.py) - SantanaLog (detach設計)
- [satipatthana/train/trainer.py](../../satipatthana/train/trainer.py) - optimizer reset, lazy init, 4-way split
- [satipatthana/components/objectives/vipassana.py](../../satipatthana/components/objectives/vipassana.py) - StabilityLoss, ProbeDiversityLoss
- [satipatthana/components/vitakka/standard.py](../../satipatthana/components/vitakka/standard.py) - gate機構
- [satipatthana/components/vipassana/standard.py](../../satipatthana/components/vipassana/standard.py) - trust_head分離, log1p
- [satipatthana/components/decoders/vision.py](../../satipatthana/components/decoders/vision.py) - Upsample+Conv2d
- [satipatthana/configs/system.py](../../satipatthana/configs/system.py) - drunk_mode parameters
