# Samadhi Framework v4.1: Presets & Factory Functions - 設計仕様書

## 1. はじめに (Introduction)

本ドキュメントは、Samadhi Framework v4.0 アーキテクチャに対応した **Presets（ファクトリ関数）** の設計仕様を定義します。

v4.0 では旧 `SamadhiBuilder` パターンを廃止し、コンポーネントを直接コンストラクタに渡す設計に移行しました。しかし、毎回すべてのコンポーネントを手動でインスタンス化するのは冗長です。Presets は、よく使われる構成を1行で構築できる便利なファクトリ関数を提供します。

## 2. 目的 (Goals)

1. **簡潔なAPI**: 1行で `SatipatthanaSystem` を構築可能にする
2. **型安全性**: 設定の型チェックを維持
3. **柔軟性**: デフォルト値を使いつつ、必要に応じてカスタマイズ可能
4. **ドメイン別最適化**: Tabular / Vision / Sequence 各ドメインに最適なデフォルト構成を提供

## 3. API設計 (API Design)

### 3.1. 基本シグネチャ

```python
from satipatthana.presets import create_mlp_system, create_conv_system, create_lstm_system

# Tabular データ用
system = create_mlp_system(
    input_dim=128,
    output_dim=10,
    dim=64,                    # Optional: 潜在空間次元 (default: 64)
    context_dim=16,            # Optional: Vipassana context次元 (default: 16)
    n_probes=5,                # Optional: Vitakka probe数 (default: 5)
    max_steps=10,              # Optional: Vicara最大ステップ (default: 10)
)

# Vision データ用
system = create_conv_system(
    img_size=32,
    channels=3,
    output_dim=10,
    dim=128,
)

# Sequence データ用
system = create_lstm_system(
    input_dim=64,
    seq_len=100,
    output_dim=10,
    dim=128,
)
```

### 3.2. 戻り値

すべてのPreset関数は `SatipatthanaSystem` インスタンスを返します。

```python
system = create_mlp_system(input_dim=128, output_dim=10)

# 推論
result = system(x)
print(result.output)       # 予測結果
print(result.trust_score)  # 信頼度

# 学習
from satipatthana.train import SatipatthanaTrainer
trainer = SatipatthanaTrainer(model=system, ...)
trainer.run_curriculum(...)
```

### 3.3. 利用可能なPresets

| 関数名 | ドメイン | Adapter | Decoder | 用途 |
|--------|----------|---------|---------|------|
| `create_mlp_system` | Tabular | MlpAdapter | ConditionalDecoder | 表形式データ、特徴量ベース |
| `create_conv_system` | Vision | CnnAdapter | ConditionalDecoder | 画像分類、異常検知 |
| `create_lstm_system` | Sequence | LstmAdapter | ConditionalDecoder | 時系列予測 |
| `create_transformer_system` | Sequence | TransformerAdapter | ConditionalDecoder | 長期依存のある系列 |

### 3.4. 共通オプション

すべてのPreset関数で共通のオプションパラメータ：

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `dim` | int | 64 | 潜在空間の次元 |
| `context_dim` | int | 16 | Vipassana context vectorの次元 |
| `n_probes` | int | 5 | Vitakkaのprobe数 |
| `max_steps` | int | 10 | Vicaraループの最大ステップ |
| `use_augmenter` | bool | True | Augmenterを使用するか |
| `augmenter_type` | str | "gaussian" | "identity" or "gaussian" |
| `sati_type` | str | "threshold" | "fixed_step" or "threshold" |
| `include_recon_heads` | bool | True | Reconstruction Headsを含めるか（学習用） |
| `include_aux_head` | bool | False | AuxiliaryHeadを含めるか（Stage 1 guidance用） |

## 4. 実装方針 (Implementation Strategy)

### 4.1. Builder を使わない直接構築

```python
def create_mlp_system(
    input_dim: int,
    output_dim: int,
    dim: int = 64,
    context_dim: int = 16,
    n_probes: int = 5,
    max_steps: int = 10,
    **kwargs,
) -> SatipatthanaSystem:
    """MLP-based SatipatthanaSystem for tabular data."""

    # 1. Configs
    adapter_config = MlpAdapterConfig(input_dim=input_dim, dim=dim)
    vitakka_config = StandardVitakkaConfig(dim=dim, n_probes=n_probes)
    vicara_config = StandardVicaraConfig(dim=dim)
    sati_config = ThresholdSatiConfig(max_steps=max_steps)
    augmenter_config = GaussianNoiseAugmenterConfig()
    vipassana_config = StandardVipassanaConfig(context_dim=context_dim)
    decoder_config = ConditionalDecoderConfig(
        dim=dim, context_dim=context_dim, output_dim=output_dim
    )
    samatha_config = SamathaConfig(dim=dim, max_steps=max_steps)
    vipassana_engine_config = VipassanaEngineConfig()
    system_config = SystemConfig(dim=dim)

    # 2. Components
    adapter = MlpAdapter(adapter_config)
    augmenter = GaussianNoiseAugmenter(augmenter_config)
    vitakka = StandardVitakka(vitakka_config)
    refiner = MlpRefiner({"dim": dim})
    vicara = StandardVicara(vicara_config, refiners=refiner)
    sati = ThresholdSati(sati_config)
    vipassana_module = StandardVipassana(vipassana_config)
    decoder = ConditionalDecoder(decoder_config)

    # 3. Engines
    samatha = SamathaEngine(
        config=samatha_config,
        adapter=adapter,
        augmenter=augmenter,
        vitakka=vitakka,
        vicara=vicara,
        sati=sati,
    )
    vipassana = VipassanaEngine(
        config=vipassana_engine_config,
        vipassana=vipassana_module,
    )

    # 4. System
    system = SatipatthanaSystem(
        config=system_config,
        samatha=samatha,
        vipassana=vipassana,
        task_decoder=decoder,
    )

    return system
```

### 4.2. ディレクトリ構造

```
satipatthana/presets/
├── __init__.py          # 全Preset関数をエクスポート
├── tabular.py           # create_mlp_system
├── vision.py            # create_conv_system
└── sequence.py          # create_lstm_system, create_transformer_system
```

## 5. 実装フェーズ (Implementation Roadmap)

### Phase 1: 基本Presets - PLANNED

**Status**: PLANNED

1. `create_mlp_system` - Tabular用基本Preset
2. 単体テスト作成

### Phase 2: ドメイン別Presets - PLANNED

**Status**: PLANNED

1. `create_conv_system` - Vision用
2. `create_lstm_system` - Sequence用（LSTM）
3. `create_transformer_system` - Sequence用（Transformer）

### Phase 3: 高度なオプション - PLANNED

**Status**: PLANNED

1. Reconstruction Heads の自動追加オプション
2. AuxiliaryHead の自動追加オプション
3. カスタムコンポーネント注入のサポート

## 6. テスト方針 (Testing Strategy)

```python
# tests/presets/test_tabular.py
def test_create_mlp_system_basic():
    system = create_mlp_system(input_dim=128, output_dim=10)
    assert isinstance(system, SatipatthanaSystem)

    x = torch.randn(4, 128)
    result = system(x)
    assert result.output.shape == (4, 10)
    assert result.trust_score.shape == (4, 1)

def test_create_mlp_system_custom_dim():
    system = create_mlp_system(input_dim=128, output_dim=10, dim=256)
    assert system.samatha.adapter.config.dim == 256
```

## 7. 移行ガイド (Migration Guide)

### 旧方式（v3.x Builder）

```python
# 削除されたAPI
engine = (
    SamadhiBuilder(config)
    .set_adapter(adapter)
    .set_vitakka()
    .set_vicara(refiner_type="mlp")
    .set_decoder(decoder)
    .build()
)
```

### 新方式（v4.1 Presets）

```python
# 推奨: Preset使用
system = create_mlp_system(input_dim=128, output_dim=10)

# または: 完全なカスタム構築
system = SatipatthanaSystem(
    config=system_config,
    samatha=samatha,
    vipassana=vipassana,
    task_decoder=decoder,
)
```

---

以上が、Samadhi Framework v4.1 Presets の設計仕様書となります。
