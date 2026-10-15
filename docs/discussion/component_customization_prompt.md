# Satipatthana Framework: コンポーネントカスタマイズ性の相談

## 背景

前回の相談で、Config設計を「2層構造（ユーザー向けシンプルConfig + 内部詳細Config）」にリファクタリングする方針が決まりました。

```python
# 新しいユーザー向けAPI（予定）
system = create_system("mlp", input_dim=128, output_dim=10)

# または
config = SatipatthanaConfig(
    input_dim=128,
    output_dim=10,
    adapter_type="mlp",
    latent_dim=64,
)
system = config.build()
```

## 新たな懸念

この設計を進める中で、**コンポーネントのカスタマイズ性が失われる**問題に気づきました。

### 現状の内部Config（多態性あり）

```python
@dataclass
class SamathaConfig(BaseConfig):
    # Base型で受けるので、任意の実装を注入可能
    adapter: BaseAdapterConfig = field(default_factory=lambda: MlpAdapterConfig(...))
    augmenter: BaseAugmenterConfig = field(default_factory=IdentityAugmenterConfig)
    vitakka: BaseVitakkaConfig = field(default_factory=StandardVitakkaConfig)
    vicara: BaseVicaraConfig = field(default_factory=StandardVicaraConfig)
    sati: BaseSatiConfig = field(default_factory=FixedStepSatiConfig)

@dataclass
class VipassanaEngineConfig(BaseConfig):
    vipassana: BaseVipassanaConfig = field(default_factory=StandardVipassanaConfig)

@dataclass
class SystemConfig(BaseConfig):
    task_decoder: BaseDecoderConfig = field(default_factory=...)
```

**現状の利点**: `BaseAdapterConfig`を継承したカスタムAdapterを作れば、システムに注入可能

### 新しいユーザー向けConfig（多態性の喪失？）

```python
@dataclass
class SatipatthanaConfig:
    adapter_type: Literal["mlp", "cnn", "lstm", "transformer"] = "mlp"
    # ↑ 列挙型だと、カスタム実装を渡せない
```

## 現在存在するコンポーネントバリエーション

| カテゴリ | 実装 | 用途 |
|----------|------|------|
| **Adapter** | MLP, CNN, LSTM, Transformer, Identity | 入力データ形式に応じて選択 |
| **Augmenter** | Identity, GaussianNoise | データ拡張戦略 |
| **Vitakka** | Standard | （現状1種類のみ） |
| **Vicara** | Standard, Weighted, ProbeSpecific | 収束戦略の違い |
| **Sati** | FixedStep, Threshold | 収束判定の違い |
| **Vipassana** | Standard, LSTM | メタ認知モジュール |
| **Decoder** | Reconstruction, Conditional, CNN, LSTM, SimpleSequence, SimpleAuxHead | 出力形式 |

## 検討中の選択肢

### Option A: 主要なtypeを列挙 + カスタム用エスケープハッチ

```python
@dataclass
class SatipatthanaConfig:
    # 標準的な選択肢を列挙
    adapter_type: Literal["mlp", "cnn", "lstm", "transformer"] = "mlp"
    vicara_type: Literal["standard", "weighted", "probe"] = "standard"
    sati_type: Literal["fixed", "threshold"] = "threshold"
    vipassana_type: Literal["standard", "lstm"] = "standard"
    augmenter_type: Literal["identity", "gaussian"] = "identity"

    # カスタム用エスケープハッチ（これがあればtypeを無視）
    custom_adapter: Optional[BaseAdapterConfig] = None
    custom_vicara: Optional[BaseVicaraConfig] = None
    custom_sati: Optional[BaseSatiConfig] = None
    custom_vipassana: Optional[BaseVipassanaConfig] = None
    custom_augmenter: Optional[BaseAugmenterConfig] = None
```

**Pros:**
- シンプルなケースはシンプルなまま
- カスタムも可能

**Cons:**
- `custom_*`フィールドが多くなる
- `adapter_type`と`custom_adapter`の両方指定時の挙動が曖昧

### Option B: 2つのAPIを明確に分離

```python
# Level 1: シンプルAPI（文字列でtype指定）
system = create_system("mlp", input_dim=128, output_dim=10)

# Level 2: カスタムAPI（Configで詳細指定）
config = SatipatthanaConfig(
    input_dim=128,
    output_dim=10,
    vicara_type="weighted",  # 列挙から選択
)
system = create_system(config)

# Level 3: フルカスタムAPI（内部SystemConfigを直接使用）
system_config = SystemConfig(
    samatha=SamathaConfig(
        adapter=MyCustomAdapterConfig(...),
        vicara=MyCustomVicaraConfig(...),
    ),
    ...
)
system = SatipatthanaSystem.from_config(system_config)
```

**Pros:**
- 責務が明確
- 各レベルのAPIが汚染されない

**Cons:**
- カスタムしたい場合はLevel 3に落ちる必要がある
- Level 2とLevel 3の間のギャップが大きい

### Option C: Union型で柔軟に受ける

```python
@dataclass
class SatipatthanaConfig:
    # 文字列またはConfigオブジェクトを受け付ける
    adapter: Union[Literal["mlp", "cnn", "lstm", "transformer"], BaseAdapterConfig] = "mlp"
    vicara: Union[Literal["standard", "weighted", "probe"], BaseVicaraConfig] = "standard"
    sati: Union[Literal["fixed", "threshold"], BaseSatiConfig] = "threshold"
    vipassana: Union[Literal["standard", "lstm"], BaseVipassanaConfig] = "standard"
    augmenter: Union[Literal["identity", "gaussian"], BaseAugmenterConfig] = "identity"
```

使用例：
```python
# 文字列で指定（シンプル）
config = SatipatthanaConfig(
    input_dim=128,
    output_dim=10,
    adapter="mlp",
    vicara="weighted",
)

# Configオブジェクトで指定（カスタム）
config = SatipatthanaConfig(
    input_dim=128,
    output_dim=10,
    adapter=MyCustomAdapterConfig(input_dim=128, custom_param=0.5),
    vicara=WeightedVicaraConfig(dim=64, custom_weight=0.8),
)
```

**Pros:**
- 1つのAPIで両方のユースケースをカバー
- 柔軟性が高い

**Cons:**
- 型が複雑（`Union[str, Config]`）
- シリアライズ/デシリアライズが複雑になる
- IDEの補完が効きにくい場合がある

### Option D: Builder Pattern（部分的復活）

```python
# 基本はシンプルConfig
config = SatipatthanaConfig(input_dim=128, output_dim=10)

# カスタムが必要な場合はBuilderで上書き
system = (
    SystemBuilder(config)
    .with_vicara(WeightedVicaraConfig(dim=64, custom_weight=0.8))
    .with_custom_adapter(MyCustomAdapterConfig(...))
    .build()
)
```

**Pros:**
- 基本はシンプル、カスタムは明示的
- メソッドチェーンで可読性高い

**Cons:**
- v4で廃止した理由（内部状態管理の複雑さ）
- 2つの構築パターンが混在

## 追加の質問：リファクタリングの順序

現在、以下の2つのタスクがあります：

1. **Config 2層化リファクタリング**（Issue #19）
   - `SatipatthanaConfig`導入
   - `CurriculumConfig`導入
   - `create_system()`ファクトリ関数

2. **コンポーネントカスタマイズ性の設計**（本件）
   - どのOption（A/B/C/D）を採用するか
   - カスタムコンポーネント注入の仕組み

**質問**: これらは同時に設計・実装すべきですか？それとも、まず2層化を完了してから、カスタマイズ性を後から追加すべきですか？

### 同時進行の場合

- 最初から柔軟な設計になる
- 手戻りが少ない
- ただし設計が複雑になり、実装が遅れる可能性

### 段階的進行の場合

- Phase 1: 2層Config（標準コンポーネントのみ対応）
- Phase 2: カスタマイズ性追加

- 早くMVPが動く
- ただし後から設計変更が必要になる可能性

## 質問まとめ

1. **Option A〜Dのどれが最もバランスが良いと思いますか？**
   - シンプルさ
   - カスタマイズ性
   - 型安全性
   - 保守性

2. **現実的に、どの程度のカスタマイズニーズがあると想定すべきですか？**
   - 研究用途では頻繁にコンポーネントを差し替える？
   - 実運用では標準コンポーネントで十分？

3. **リファクタリングの順序について、同時進行 vs 段階的進行、どちらを推奨しますか？**

4. **他のMLフレームワークで参考になるパターンはありますか？**
   - HuggingFace Transformersのカスタムモデル登録
   - PyTorch Lightningのカスタムコンポーネント
   - Kerasのカスタムレイヤー

## 参考：フレームワークの用途

Satipatthana Frameworkは以下の用途を想定しています：

1. **研究用途**: 新しいVicara戦略やVipassanaモジュールの実験
2. **実運用**: 異常検知、分類タスクでの利用
3. **教育用途**: 再帰的アテンションアーキテクチャの学習

研究用途ではカスタマイズ性が重要、実運用では安定性とシンプルさが重要、という異なるニーズがあります。
