# Samadhi Framework Refactoring Plan (v3.1 - Final)

**Version:** 3.1 (Final)
**Status:** Completed
**Goal:** Dict ベースの緩い設定管理から、**継承・Factory・Enumを活用した** Type-Safe (型安全) な Dataclass ベースの設定管理への移行。

## 1. 動機と課題

### 現状 (v3.0)
*   設定情報は `config: Dict[str, Any]` という辞書型でバケツリレーされている。

### 課題
1.  **可読性の欠如**: どのコンポーネントが何の設定を必要とするか不明確。
2.  **エラーの温床**: タイポや型不一致が実行時まで発覚しない。
3.  **Configクラスの肥大化**: 単一Configに全パラメータを詰め込むと保守性が低下する。
4.  **バリデーション不足**: 値の範囲チェックや型変換（list -> tuple）が保証されない。

### 新設計 (v3.1 Final)
*   **BaseConfig & Validation**: 共通基底クラス `BaseConfig` を導入し、`__post_init__` から呼ばれる `validate()` フックで値の検証や型変換を集約する。
*   **Enum Type Safety**: コンポーネントの識別 (`type`) に文字列ではなく `Enum` を使用し、タイポを防ぐ。
*   **Polymorphism & Factory**: コンポーネント毎にConfigを分割し、Factory関数で適切に生成する。
*   **Standard Library Only**: `inspect` 等の標準機能のみで実装し、依存関係を増やさない。

## 2. アーキテクチャ概要

**注意:** 以下のディレクトリ構成とコードスニペットは、計画段階の初期案であり、最終的な実装とは異なります。
最新のConfigシステムの構造については、[`samadhi/configs/README.md`](samadhi/configs/README.md) を参照してください。

### 2.1 設定クラスの階層構造
`BaseConfig` を基点とし、共通のバリデーションやDict変換ロジックを集約する。

```text
BaseConfig (validate, from_dict)
├── SamadhiConfig (Root)
├── BaseAdapterConfig (Abstract)
│   ├── MlpAdapterConfig
│   └── CnnAdapterConfig
# ... その他コンポーネント設定
```

### 2.2 ファイル構成
Enum定義も含める。

```text
# この構成は初期案であり、最終的な実装では samadhi/configs/ 内のファイルがより細かく分割されました。
# 最新のファイル構成は samadhi/configs/README.md を参照してください。
samadhi/configs/
├── __init__.py
├── main.py            # SamadhiConfig
├── base.py            # BaseConfig (共通ロジック)
├── enums.py           # AdapterType, VicaraType 等の定義
├── components.py      # <- このファイルは削除され、より具体的なファイルに分割されました。
└── factory.py         # Factory関数 (Enumで分岐)
```

## 3. 実装詳細

**注意:** 以下のコードスニペットは、計画段階の初期案であり、現在の実装とは異なります。
最新の実装詳細については、各Configファイル（`samadhi/configs/adapters.py`など）および[`docs/config_summary.md`](docs/config_summary.md)を参照してください。

### 3.1 共通基底クラス (`samadhi/configs/base.py`)
全てのConfigの親クラス。`from_dict` による安全な生成と `validate` フックを提供する。

```python
# (中略) 最新のコードは samadhi/configs/base.py を参照
@dataclass
class BaseConfig:
    # ...
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        # ...
        print(f"WARNING: Unknown keys found for {cls.__name__}") # このprintはログ出力に置き換えられました
        # ...
```

### 3.2 Enum定義 (`samadhi/configs/enums.py`)
文字列リテラルのタイポを防ぐ。

```python
# (中略) 最新のコードは samadhi/configs/enums.py を参照
class AdapterType(str, Enum):
    MLP = "mlp"
    CNN = "cnn"
    SEQUENCE = "sequence" # -> 現在は LSTM, TRANSFORMER も含まれます

# ...
```

### 3.3 コンポーネントConfig定義 (`samadhi/configs/components.py`)
**このファイルは削除され、機能は `samadhi/configs/adapters.py`, `samadhi/configs/vicara.py`, `samadhi/configs/vitakka.py`, `samadhi/configs/decoders.py`, `samadhi/configs/objectives.py` に分割されました。**

### 3.4 変換ファクトリ (`samadhi/configs/factory.py`)
Enumを使って分岐し、`from_dict` を呼び出す。非常にシンプルになる。

```python
# (中略) 最新のコードは samadhi/configs/factory.py を参照
# ファクトリ関数は、ObjectiveConfigを含む全てのConfigコンポーネントに対応するようになりました。
# また、型解決のためのより堅牢なロジックが追加されました。
```

### 3.5 全体Config定義 (`samadhi/configs/main.py`)
`BaseConfig` を継承。ネスト構造の復元ロジックも整理。

```python
# (中略) 最新のコードは samadhi/configs/main.py を参照
# SamadhiConfigは objective 設定もネストするようになり、
# from_dict メソッドもフラットな辞書からの objective 関連パラメータの抽出に対応しました。
```

## 4. 移行戦略 (Migration Strategy)

**このセクションで記述された移行作業はすべて完了しました。**

Configシステムのリファクタリングは計画通りに完了し、コードベース全体に適用されました。
これに伴い、[`samadhi/configs/README.md`](samadhi/configs/README.md)と[`docs/config_summary.md`](docs/config_summary.md)が最新のConfigシステムの構造と使い方を記述しています。
