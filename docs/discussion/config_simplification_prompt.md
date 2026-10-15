# Satipatthana Framework: Config設計の相談

## 背景

Satipatthana Frameworkは、再帰的アテンションによる収束と自己観察を行う深層学習アーキテクチャです。仏教心理学の概念をエンジニアリングにマッピングしています：

- **Samatha（止）**: 入力を潜在空間で収束させるエンジン
- **Vipassana（観）**: 収束プロセスを観察し、信頼度スコアを出力するメタ認知モジュール
- **4-Stage Curriculum Training**: Adapter事前学習 → Samatha学習 → Vipassana学習 → Decoder微調整

## 現在の問題

### Config地獄

現在、システム構築に必要なConfigクラスが13種類以上あり、階層が深くなっています：

```
SystemConfig
├── SamathaConfig
│   ├── MlpAdapterConfig / CnnAdapterConfig / LstmAdapterConfig / TransformerAdapterConfig
│   ├── AugmenterConfig (Identity / GaussianNoise)
│   ├── StandardVitakkaConfig
│   ├── VicaraConfig (Standard / Weighted / ProbeSpecific)
│   └── SatiConfig (FixedStep / Threshold)
├── VipassanaEngineConfig
│   └── StandardVipassanaConfig
├── ConditionalDecoderConfig (タスク出力用)
├── ReconstructionDecoderConfig (×2: adapter用, samatha用)
└── SimpleAuxHeadConfig (Stage 1補助)
```

### 具体的な問題点

1. **`dim`（潜在次元）の重複指定**
   - `SystemConfig.dim`
   - `SamathaConfig.dim`
   - `MlpAdapterConfig.dim`
   - `StandardVitakkaConfig.dim`
   - `StandardVicaraConfig.dim`
   - `StandardVipassanaConfig.latent_dim`
   - `ConditionalDecoderConfig.dim`

   → 全部同じ値なのに7箇所で指定が必要

2. **システム構築が冗長**（現状20行以上必要）
   ```python
   # 現在の構築方法
   config = SystemConfig(
       dim=64,
       samatha=SamathaConfig(
           dim=64,  # また指定
           adapter=MlpAdapterConfig(input_dim=128, dim=64),  # また指定
           augmenter=IdentityAugmenterConfig(),
           vitakka=StandardVitakkaConfig(dim=64),  # また指定
           vicara=StandardVicaraConfig(dim=64),  # また指定
           sati=FixedStepSatiConfig(),
       ),
       vipassana=VipassanaEngineConfig(
           vipassana=StandardVipassanaConfig(latent_dim=64),  # また指定
       ),
       task_decoder=ConditionalDecoderConfig(dim=64, context_dim=32, output_dim=10),
   )

   # さらにコンポーネントを手動でインスタンス化...
   adapter = MlpAdapter(config.samatha.adapter)
   augmenter = IdentityAugmenter(config.samatha.augmenter)
   # ... 10行以上続く
   ```

3. **Trainerも同様の問題**
   - Stage別のlearning rate設定ができない
   - 14個のパラメータがフラットに並ぶ
   - TrainerConfigを追加すると、また新しいConfig体系が増える

## 検討中の選択肢

### A. Flat Config（完全フラット化）

```python
config = SatipatthanaConfig(
    # 必須
    input_dim=128,
    output_dim=10,

    # 共通（1箇所で指定）
    latent_dim=64,
    context_dim=32,

    # Adapter
    adapter_type="mlp",  # "mlp" | "cnn" | "lstm" | "transformer"
    adapter_hidden_dim=256,

    # Samatha
    n_probes=10,
    max_steps=10,
    vicara_type="standard",
    sati_type="threshold",

    # Vipassana
    gru_hidden_dim=32,

    # Training
    use_label_guidance=False,
    task_type="classification",
)

system = SatipatthanaSystem.from_config(config)
```

**Pros**: シンプル、`dim`の重複なし
**Cons**: パラメータ数が多い（30個以上）、拡張性が低い

### B. Preset + Override（ファクトリ関数 + 上書き）

```python
from satipatthana.presets import create_mlp_system

# 基本：1行で構築
system = create_mlp_system(input_dim=128, output_dim=10)

# カスタマイズ：必要な箇所だけ上書き
system = create_mlp_system(
    input_dim=128,
    output_dim=10,
    latent_dim=128,  # デフォルト64を変更
    n_probes=20,     # デフォルト10を変更
)

# 詳細カスタマイズ：dotted keyで深い階層も指定可能
system = create_mlp_system(
    input_dim=128,
    output_dim=10,
    overrides={
        "vicara.inertia": 0.8,
        "vipassana.gru_hidden_dim": 64,
    }
)
```

**Pros**: 普段はシンプル、必要時だけ詳細指定
**Cons**: overridesの仕組みが複雑、typoに気づきにくい

### C. Builder Pattern（v3で廃止したが再検討）

```python
system = (
    SatipatthanaBuilder(input_dim=128, output_dim=10, latent_dim=64)
    .with_mlp_adapter(hidden_dim=256)
    .with_standard_vicara(inertia=0.8)
    .with_threshold_sati(threshold=1e-4)
    .build()
)
```

**Pros**: メソッドチェーンで可読性が高い、IDEの補完が効く
**Cons**: v4で廃止した理由（内部状態管理の複雑さ）、どこまでカスタマイズ可能にするか曖昧

### D. Hybrid: 2層Config

```python
# Layer 1: ユーザー向けシンプルConfig
@dataclass
class SatipatthanaConfig:
    input_dim: int
    output_dim: int
    latent_dim: int = 64
    adapter_type: str = "mlp"
    # ... よく使うパラメータのみ

# Layer 2: 内部用詳細Config（自動生成）
def build_system(config: SatipatthanaConfig) -> SatipatthanaSystem:
    # SatipatthanaConfigから内部の詳細Configを自動生成
    system_config = SystemConfig(
        dim=config.latent_dim,
        samatha=SamathaConfig(
            dim=config.latent_dim,  # 自動で同じ値
            ...
        ),
        ...
    )
    # コンポーネントも自動構築
    return SatipatthanaSystem(...)
```

**Pros**: ユーザーはシンプル、内部は型安全を維持
**Cons**: 2層のConfigが存在することの認知負荷

## 追加の検討事項

### Trainer設定について

現在のTrainerも14個のパラメータがあり、Stage別設定ができない：

```python
# 現状
trainer = SatipatthanaTrainer(
    model=system,
    args=TrainingArguments(...),
    stage=TrainingStage.SAMATHA_TRAINING,
    noise_level=0.3,
    use_label_guidance=True,
    task_type="classification",
    stability_weight=0.1,
    guidance_weight=1.0,
    recon_weight=1.0,
    # ... まだ続く
)
```

Stage別learning rateの設定方法：

**案1: TrainerConfig追加**（またConfigが増える）
```python
trainer_config = TrainerConfig(
    stage0=StageConfig(epochs=5, lr=1e-3),
    stage1=StageConfig(epochs=10, lr=5e-4),
    ...
)
```

**案2: run_curriculumの引数拡張**
```python
trainer.run_curriculum(
    stage0={"epochs": 5, "lr": 1e-3},
    stage1={"epochs": 10, "lr": 5e-4},
    ...
)
```

**案3: Stage別メソッド**
```python
trainer.train_stage0(epochs=5, lr=1e-3)
trainer.train_stage1(epochs=10, lr=5e-4)
```

## 質問

1. **Config設計として、A〜Dのどれが最もバランスが良いと思いますか？**
   - ユーザビリティ（簡単に使える）
   - 拡張性（将来の機能追加）
   - 保守性（コードの複雑さ）

2. **`dim`のような共通パラメータの伝播について、ベストプラクティスはありますか？**
   - 親から子への自動伝播？
   - 子での明示的な指定を必須？
   - デフォルト値の上書きルール？

3. **Trainer設定について、Config追加 vs 引数拡張 vs メソッド分割、どれが適切でしょうか？**

4. **他のMLフレームワーク（PyTorch Lightning, Keras, Hugging Face Transformers）で参考になる設計パターンはありますか？**

## 参考：現在のディレクトリ構造

```
satipatthana/
├── configs/           # 13種類以上のConfigクラス
│   ├── system.py      # SystemConfig, SamathaConfig
│   ├── adapters.py    # 5種類のAdapterConfig
│   ├── vicara.py      # 3種類のVicaraConfig
│   └── ...
├── components/        # 実装クラス
│   ├── adapters/
│   ├── vicara/
│   └── ...
├── core/
│   ├── system.py      # SatipatthanaSystem
│   └── engines.py     # SamathaEngine, VipassanaEngine
└── train/
    └── trainer.py     # SatipatthanaTrainer (611行)
```
