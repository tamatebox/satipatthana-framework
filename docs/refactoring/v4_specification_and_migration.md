# Samadhi Framework v4.0: Introspective Deep Convergence Architecture - 設計仕様書

## 1. はじめに (Introduction)

本ドキュメントは、Samadhi Framework を v4.0 (Introspective Deep Convergence Architecture) へ移行するための最終設計仕様を定義します。v4.0は、従来のブラックボックスAIとは一線を画し、「思考（Samatha）→ 内省（Vipassana）→ 表現（Conditional Decoding）」という3段階の認知プロセスを持つ、アビダルマ的AIシステムです。

このアーキテクチャは、**説明可能性 (Explainability)** と **信頼性 (Reliability)** を飛躍的に向上させることを目的とし、以下の「3つの知性」を同時に提供します。

* **Robustness (堅牢性)**: Samatha による強力なノイズ除去と収束能力。
* **Self-Awareness (自己認識)**: Vipassana による「自分の回答に対する自信」の定量化。
* **Humble Expression (謙虚な表現)**: Conditional Decoder による「自信のなさを加味した」安全なアウトプット。

## 2. システムアーキテクチャ (System Architecture)

Samadhi v4.0は、情報の質的変換を担う3つの主要なエンジンと、それらを構成するモジュラーコンポーネント、そして学習を補助するReconstruction Headによって構成されます。データは一方通行で流れ、各フェーズで情報の「質」が変化します。

### 2.1. 全体クラス構造 (Class Diagram)

確定したクラス構造は以下のPlantUML図を参照してください。
`docs/refactoring/v4_class_diagram.pu`

### 2.2. データフローの概要

1. **Input (X)**: ノイズを含む生データ。
2. **Phase 1 (Samatha)**: 物理法則（収束）に従い、**「不動点 (S\*)」と「思考ログ (SantanaLog)」**を生成。
3. **Phase 2 (Vipassana)**: ログを解析し、その思考の**「文脈ベクトル (V_{ctx})」と「信頼度 (\alpha)」**を生成。
4. **Phase 3 (Decoder)**: 「結論 (S\*)」と「自信のなさ (V_{ctx})」を結合して、タスクに最適な**「出力 (Y)」**を生成。

### 2.3. エンジン別詳細仕様 (Component Specs)

システムは `torch.nn.Module` を継承する3つの主要なエンジンクラス (`SamathaEngine`, `VipassanaEngine`, `ConditionalDecoder` は `SamadhiSystem` 内の `task_decoder` として管理) と、学習補助のReconstruction Headで構成されます。これらは `SamadhiSystem` で統括されます。

#### 2.3.1. Engine 1: SamathaEngine (The Meditator)

* **役割**: 世界モデル。いかなる入力も「意味のある点」に収束させる。
* **入力**: Raw Data `X` (または Augmenter 処理済み `X_aug`) - `(Batch, *)`
* **出力**:
  * `Converged State S*` (Batch, Dim) - `torch.float32`
  * `Thinking Logs (SantanaLog)` (オブジェクト) - `SantanaLog`
  * `Severity` (float, ノイズの強度、Vipassanaターゲット用) - `float`
* **構成**: `Adapter`, `Augmenter`, `Vitakka`, `Vicara`, `Sati`。
* **特徴**: タスクやラベルには依存せず、**「構造の抽出」**のみを行う。`drunk_mode` フラグにより内部的な摂動制御を行う。

#### 2.3.2. Engine 2: VipassanaEngine (The Observer)

* **役割**: メタ認知。Samathaの思考プロセス（ログ）が健全だったか監視する。
* **入力**: `S*` (Batch, Dim) + `SantanaLog` (Batchサイズに対応したオブジェクト)
* **出力**:
  * `Context Vector V_{ctx}` (Batch, C_Dim): デコーダーへのヒント情報（「迷い」の埋め込み表現）。 - `torch.float32`
  * `Trust Score \alpha` (Batch, 1): 外部制御用の信頼度スコア (0.0~1.0)。 - `torch.float32`
* **構成**: `BaseVipassana` を継承する `LogEncoder` と `ConfidenceMonitor`。
* **特徴**: 思考過程の健全性から直接不確実性を推定する。

#### 2.3.3. Engine 3: Conditional Task Decoder (The Speaker)

* **役割**: 表現。状態と文脈を統合して、人間にわかる形にする。
* **入力**: `S*` (Batch, Dim) + `V_{ctx}` (Batch, C_Dim) (Concatenate) -> `(Batch, Dim + C_Dim)`
* **出力**: Final Output `Y` (Class Logits / Value) - `(Batch, OutputDim)` (`torch.float32`)
* **構成**: `BaseDecoder` を継承する `ConditionalDecoder`。
* **特徴**: 「自信がない時は、自信がないような出力（分散を広げる等）」が可能になり、**謙虚な表現**を実現する。**推論時に使用される唯一のDecoder**。

### 2.4. Reconstruction Head (学習補助 Decoder)

Samadhi v4.0では、タスク用の `ConditionalDecoder` とは別に、学習の安定化を目的とした2種類の `Reconstruction Head` (`SimpleReconstructionDecoder`) を使用します。これらは推論時には使用されません。

* **`adapter_recon_head` (Stage 0用)**: Adapter の出力 `z` `(Batch, Dim)` から元の入力 `X` `(Batch, *)` を再構成。Adapter の潜在空間初期化に寄与。
* **`samatha_recon_head` (Stage 1用)**: Samatha の収束点 `S*` `(Batch, Dim)` から元の入力 `X` `(Batch, *)` を再構成。Samatha の力学系が入力情報を保持しているか監視。

### 2.5. コンポーネントの責務分離 (Component Responsibilities)

ノイズ生成と異常モードの責務は以下の通り明確に分離されます。

* **`Augmenter` (Input Only)**:
  * **役割**: 生データ `x` に対して環境ノイズを付与する。
  * **出力**: `(x_augmented: Tensor, severity: Tensor)` - `x_augmented` の Shape は `x` と同じ。`severity` は Shape `(Batch,)` でサンプルごとのノイズ強度を表す。
  * **特徴**: 入力操作に責務を限定し、SamathaEngine の内部構造には依存しない。バッチ処理時の型一貫性のため `severity` は `Tensor` 型とする。
* **`SamathaEngine` (Drunk Mode Control)**:
  * **役割**: Drunk Mode の実装。`drunk_mode` フラグに応じて `Vitakka` / `Vicara` / `Sati` の内部パラメータや挙動を操作し、思考プロセスの混乱をシミュレートする。
  * **特徴**: 内部状態操作は SamathaEngine 内に閉じる。
* **`Trainer` (Mismatch Control)**:
  * **役割**: 論理不整合（Mismatch）の制御。Samatha から生成された `S*` と `SantanaLog` を取得後、バッチ内でシャッフルすることで人為的な不整合を作成する。
  * **特徴**: データ整形を担当し、学習フローを明確にする。

## 3. 学習カリキュラム (The 4-Stage Curriculum)

Samadhi v4.0の学習は、各コンポーネントを段階的に安定させるための4つのステージで構成されます。これにより学習の不安定性を最小限に抑え、各モジュールがそれぞれの役割を効果的に習得できるようになります。

### 3.1. 学習ポリシー (Freeze / Train)

各ステージにおけるコンポーネントの学習 (Train) および凍結 (Freeze) ポリシーは以下の通りです。

| Stage                      | Train 対象                                    | Freeze 対象                                                  | 目的関数 (Loss)                                           | データ               |
| :------------------------- | :-------------------------------------------- | :----------------------------------------------------------- | :-------------------------------------------------------- | :------------------- |
| **0. Adapter Pre-training**<br>(Optional/Recommended) | Adapter, `adapter_recon_head`                 | Samatha Core (Vitakka/Vicara/Sati), Vipassana, TaskDecoder, `samatha_recon_head`, AuxiliaryHead | Reconstruction Loss (MSE)                                 | 大量のラベルなしデータ |
| **1. Samatha Training**    | Adapter, Samatha Core (Vitakka/Vicara/Sati), (Optional) `samatha_recon_head`, (Conditional) AuxiliaryHead | Vipassana, TaskDecoder, `adapter_recon_head`                 | Stability Loss + Reconstruction Loss + (Conditional) `L_{guide}(y, \hat{y}_{aux})` | 大量のラベルなしデータ (+ Conditional Labels) |
| **2. Vipassana Training**  | Vipassana                                     | Samatha Core, Adapter, ReconHeads, TaskDecoder, AuxiliaryHead | Binary Cross Entropy (Contrastive Loss)                   | Samathaの出力 (S*, Santana) + 人工ノイズ (w/ Conditional Label-based Mismatch) |
| **3. Decoder Fine-tuning** | TaskDecoder                                   | Samatha Core, Adapter, Vipassana, ReconHeads, AuxiliaryHead  | Task Specific Loss (CrossEntropy / MSE etc. or Self-Supervised Loss) | 少量のラベル付きデータ (or Self-Supervised Targets) |

### 3.2. 各ステージの詳細フロー (Detailed Sequence Diagrams)

各ステージの詳細な学習フローは、以下のシーケンス図を参照してください。

* **全体概要**: `docs/refactoring/v4_sequence_diagram_training_overview.pu`
* **Stage 0 (Adapter Pre-training)**: `docs/refactoring/v4_sequence_diagram_training_stage0.pu`
* **Stage 1 (Samatha Training)**: `docs/refactoring/v4_sequence_diagram_training_stage1.pu`
* **Stage 2 (Vipassana Training)**: `docs/refactoring/v4_sequence_diagram_training_stage2.pu`
* **Stage 3 (Decoder Fine-tuning)**: `docs/refactoring/v4_sequence_diagram_training_stage3.pu`

### 3.3. ノイズ生成戦略 (Noise Generation Strategy for Stage 2)

Stage 2 の Vipassana Training では、Vipassana にメタ認知能力を習得させるため、以下の3種類の擬似負例/曖昧データ生成戦略をTrainerがオーケストレーションします。詳細は `docs/refactoring/v4_sequence_diagram_noise_generation.pu` を参照してください。

1. **Environmental Ambiguity (Augmented Path)**:
    * **目的**: 入力データ自体の品質低下による判断の迷いをシミュレート（Aleatoric Uncertainty）。
    * **担当**: `Augmenter` が生データにドメイン固有のノイズを付与し、`severity` をTrainerに返す。
    * **Samathaの挙動**: `x_aug` に対して通常通り収束処理を行う。
    * **VipassanaのTarget**: `1.0 - severity_aug`

2. **Internal Dysfunction (Drunk Path)**:
    * **目的**: Samatha内部のパラメータや推論パスが一時的に阻害された状態（脳の混乱）をシミュレート。
    * **担当**: `SamathaEngine` が `drunk_mode=True` フラグを受け取り、内部の `Vitakka` / `Vicara` の挙動を意図的に不安定にする（高Dropout率、プローブシャッフル、ランダム更新スキップなど）。
    * **Samathaの挙動**: カオスな思考軌跡 `SantanaLog` を生成。
    * **VipassanaのTarget**: `0.0`

3. **Logical Inconsistency (Mismatch Path)**:
    * **目的**: 「結論」と「導出過程」の因果関係が破綻している状態（幻覚）をシミュレート。
    * **担当**: `Trainer` が `Samatha` から通常の `S*` と `SantanaLog` を取得後、バッチ内でシャッフルすることで人為的な不整合を作成する。
    * **Samathaの挙動**: 通常通り収束処理を行う。
    * **VipassanaのTarget**: `0.0`

## 4. データフローと推論 (Inference Flow)

本番環境での推論フローは、以下のシーケンス図を参照してください。
`docs/refactoring/v4_sequence_diagram_inference.pu`

### 4.1. 推論時のDecoder使用

* 推論時には、**必ず Stage 3 で学習した `Conditional Task Decoder` のみを使用します**。
* Stage 0 および Stage 1 で使用された `Reconstruction Head` (Adapter用/Samatha用) は、推論時には使用されず破棄されます。

## 5. ディレクトリ構造と移行マップ (Directory Structure & Migration Map)

v3.1の構造をベースに、8コンポーネント制への移行、EngineとSystemの概念導入、およびConfigシステムの階層化を行います。

### 5.1. 主要なディレクトリ再編

* `samadhi/core/`:
  * `system.py`: `SamadhiSystem` クラス (全体統括) を新規作成。
  * `engines.py`: `SamathaEngine`, `VipassanaEngine` クラスを配置 (旧 `engine.py` を分割・改修)。
  * `santana.py`: `SantanaLog` クラス (思考ログ) を新規作成。
* `samadhi/components/`:
  * **`adapters/`**: 既存を改修。
  * **`augmenters/`**: **新規作成**。`BaseAugmenter` および具体的な Augmenter (例: `DomainAugmenter`) を配置。
  * **`vitakka/`**: 既存を改修。
  * **`vicara/`**: 既存を改修。v3.1の `WeightedVicara`, `ProbeSpecificVicara` は v4 でも継続サポート。`Sati` との統合が必要。
  * **`refiners/`**: 既存を改修。
  * **`sati/`**: **新規作成**。`BaseSati` および具体的な Sati (例: `ThresholdSati`, `FixedStepSati`) を配置。
  * **`vipassana/`**: **新規作成**。`BaseVipassana` および具体的な実装 (`LogEncoder`, `ConfidenceMonitor`) を配置。
  * **`decoders/`**: 既存を改修。`ConditionalDecoder`, `SimpleReconstructionDecoder`, `SimpleAuxHead` を配置。
  * **`objectives/`**: 旧 `samadhi/train/objectives/` から**移動**。`VipassanaObjective` を新規作成。

### 5.2. Configシステムの変更

* `samadhi/configs/main.py`: `SystemConfig` (統合設定) を中心に、各Engine/Componentの設定を階層化。
* `samadhi/configs/samatha.py`, `samadhi/configs/vipassana.py`, `samadhi/configs/objectives.py`, `samadhi/configs/augmenters.py`, `samadhi/configs/sati.py`, `samadhi/configs/auxiliary_head.py` など、必要に応じて細分化。

### 5.3. TrainerおよびStrategies

* `samadhi/train/hf_trainer.py`: `SamadhiTrainer` を改修し、4ステージ学習カリキュラムを制御できるようにする。
* `samadhi/train/strategies.py`: 各ステージの具体的な学習ループ定義 (Stage 0, 1, 2, 3) を配置 (新規作成)。

## 6. 実装フェーズ (Implementation Roadmap)

エンジニアチームは以下の詳細な4フェーズ計画に従って実装を進めてください。各フェーズには検証用のテスト項目が含まれており、これをパスすることを完了条件とします。

### Phase 1: 基盤整備 (Foundation) - COMPLETED

**Status**: **COMPLETED** (2024-12-10)

**Goal**: v4アーキテクチャの骨格を作成し、Configシステムを刷新する。

1. **ディレクトリ構造の作成**: DONE

    * `samadhi/components/augmenters/`, `sati/`, `vipassana/` の新規作成。

    * `samadhi/core/system.py`, `engines.py`, `santana.py` の新規作成。

2. **Baseクラス定義**: DONE

    * `BaseAugmenter`, `BaseSati`, `BaseVipassana` を定義 (`torch.nn.Module` ベース)。

    * 各クラスの抽象メソッド (`forward` 等) のシグネチャを確定。

3. **Configシステムの刷新**: DONE

    * `SamadhiConfig` を `SystemConfig` へと再構成。

    * `SystemConfig` に `use_label_guidance: bool` フラグを追加。

    * `samadhi/configs/` 下に `augmenter.py`, `sati.py`, `vipassana.py`, `system.py` 等を追加し、階層構造を定義。

    * `samadhi/configs/enums.py` に `AugmenterType`, `SatiType`, `VipassanaType` を追加。

**Phase 1 Tests**: PASSED (52 tests)

* **Config Test**: ネストされた辞書からConfigオブジェクトが正しく生成されるか。不正な値で初期化エラーになるか。

#### Phase 1 実装詳細

**作成されたファイル:**

| ファイル | 説明 |
|---------|------|
| `samadhi/core/santana.py` | `SantanaLog` クラス - 軌跡ログ |
| `samadhi/core/engines.py` | Engine プレースホルダー (Phase 3 で実装) |
| `samadhi/core/system.py` | System プレースホルダー (Phase 4 で実装) |
| `samadhi/components/augmenters/base.py` | `BaseAugmenter` 抽象クラス |
| `samadhi/components/sati/base.py` | `BaseSati` 抽象クラス |
| `samadhi/components/vipassana/base.py` | `BaseVipassana` 抽象クラス |
| `samadhi/configs/augmenter.py` | Augmenter 設定 |
| `samadhi/configs/sati.py` | Sati 設定 |
| `samadhi/configs/vipassana.py` | Vipassana 設定 |
| `samadhi/configs/system.py` | `SystemConfig`, `SamathaConfig`, `VipassanaEngineConfig` |

**インターフェース定義:**

| クラス | シグネチャ |
|--------|-----------|
| `BaseAugmenter` | `forward(x, noise_level) -> Tuple[Tensor, Tensor]` |
| `BaseSati` | `forward(current_state, santana) -> Tuple[bool, Dict]` |
| `BaseVipassana` | `forward(s_star, santana) -> Tuple[Tensor, float]` |
| `SantanaLog` | `add()`, `to_tensor()`, `__len__()`, etc. |

**テストファイル:**

* `tests/core/test_santana.py` - SantanaLog テスト (12 tests)
* `tests/configs/test_v4_configs.py` - Config テスト (20 tests)
* `tests/components/augmenters/test_base_augmenter.py` - Augmenter テスト (5 tests)
* `tests/components/sati/test_base_sati.py` - Sati テスト (7 tests)
* `tests/components/vipassana/test_base_vipassana.py` - Vipassana テスト (8 tests)

### Phase 2: コンポーネント実装・改修 (Component Implementation) - COMPLETED

**Status**: **COMPLETED** (2024-12-10)

**Goal**: 各単体モジュールの動作を保証する。

1. **Augmenter (New)**: DONE

    * `IdentityAugmenter`: 入力をそのまま通過させる（severity=0）
    * `GaussianNoiseAugmenter`: ガウシアンノイズを付与（max_noise_std設定可能）

2. **Sati (New)**: DONE

    * `FixedStepSati`: 常に`should_stop=False`を返す（ループ制御はEngine側）
    * `ThresholdSati`: エネルギーが閾値以下かつmin_steps到達で停止

3. **Vicara (Refactor)**: DONE

    * `step()` メソッドを追加し、単一ステップ更新インターフェースを実装
    * レガシー `forward()` は内部で `step()` を呼び出す形で後方互換性を維持
    * ループ制御は SamathaEngine に委譲（Phase 3 で統合）

4. **Vipassana (New)**: DONE

    * `StandardVipassana`: SantanaLog の軌跡特徴量をエンコードし、context vector と trust score を出力
    * 遅延初期化により状態次元に自動適応

5. **Decoder (Refactor)**: DONE

    * `ConditionalDecoder`: 入力次元を `dim + context_dim` に拡張し、状態と文脈を結合して処理
    * `SimpleAuxHead`: Stage 1 でのラベルガイダンス用シンプルMLP Head

6. **Objectives (Migration)**: DONE

    * `samadhi/train/objectives/` から `samadhi/components/objectives/` へ移動
    * 後方互換性のため旧パスからの再エクスポートを維持

**Phase 2 Tests**: PASSED (221 tests total)

* **Augmenter Tests** (27 tests): `noise_level=0` で入力不変、Shape維持、severity値の正確性
* **Sati Tests** (25 tests): SantanaLog入力、閾値判定、min_steps制約
* **Vipassana Tests** (23 tests): 可変長ログ入力、バッチ処理、固定長出力
* **Decoder Tests** (22 tests): 結合入力のShape検証、AuxHeadの出力検証
* **Vicara Step Tests** (14 tests): 単一ステップ更新、後方互換性

#### Phase 2 実装詳細

**作成されたファイル:**

| ファイル | 説明 |
|---------|------|
| `samadhi/components/augmenters/identity.py` | `IdentityAugmenter` 実装 |
| `samadhi/components/augmenters/gaussian.py` | `GaussianNoiseAugmenter` 実装 |
| `samadhi/components/sati/fixed_step.py` | `FixedStepSati` 実装 |
| `samadhi/components/sati/threshold.py` | `ThresholdSati` 実装 |
| `samadhi/components/vipassana/standard.py` | `StandardVipassana` 実装 |
| `samadhi/components/decoders/conditional.py` | `ConditionalDecoder` 実装 |
| `samadhi/components/decoders/auxiliary.py` | `SimpleAuxHead` 実装 |
| `samadhi/components/objectives/*` | 全Objective (train/から移動) |
| `samadhi/configs/decoders.py` | `ConditionalDecoderConfig`, `SimpleAuxHeadConfig` 追加 |

**テストファイル:**

* `tests/components/augmenters/test_augmenters.py` - 具象Augmenter テスト
* `tests/components/sati/test_sati.py` - 具象Sati テスト
* `tests/components/vipassana/test_vipassana.py` - 具象Vipassana テスト
* `tests/components/decoders/test_decoders.py` - ConditionalDecoder/AuxHead テスト
* `tests/components/vicara/test_vicara_step.py` - Vicara step インターフェース テスト
* `tests/components/objectives/test_objectives.py` - Objective テスト

### Phase 3: Engine実装・結合 (Engine Integration)

**Goal**: コンポーネントを組み合わせて「思考」と「内省」の機能を実現する。

1. **SamathaEngine (New)**:

    * Adapter, Augmenter, Vitakka, Vicara, Sati を統合。

    * `drunk_mode` 実装: フラグがTrueの時、内部コンポーネントパラメータを一時的に乱す（Dropout率操作など）ロジック。

    * Forward: `x` -> Augment -> Adapter -> Vitakka -> Vicara loop (w/ Sati) -> `S*`, `SantanaLog` (オブジェクト)。

2. **VipassanaEngine (New)**:

    * LogEncoder と ConfidenceMonitor を統合。

    * Forward: `S*`, `SantanaLog` (オブジェクト) -> `V_ctx`, `Trust Score`。

**Phase 3 Tests (Integration Tests)**:

* **Samatha Convergence**: ノイズなし入力で一定の収束動作をするか。

* **Samatha Drunk Mode**: `drunk_mode=True` で出力 (`SantanaLog`) が決定論的でなくなる（または大きく変動する）か。

* **Vipassana Logic**: 収束した `SantanaLog`（良）と発散した `SantanaLog`（悪）のモックデータを入力し、スコアに有意差が出るか。

### Phase 4: System & Trainer実装 (System Integration)

**Goal**: 4ステージ学習と推論フローを完成させる。

1. **SamadhiSystem (New)**:

    * 全Engine (`Samatha`, `Vipassana`) と Decoders (`Task`, `ReconHeads`, `AuxiliaryHead`) を統括。

    * `forward(x, y=Optional, stage=...)` メソッドで、ステージに応じた処理パスのディスパッチを実装。

    * `use_label_guidance` フラグに基づき、Stage 1 で `AuxiliaryHead` をアクティブにするか、Stage 2 で Mismatch 生成ロジックを切り替えるかの制御を行う。

2. **Objectives (Update)**:

    * `VipassanaObjective`: 対比学習（Good `SantanaLog` vs Bad `SantanaLog`）用のLoss実装。

    * `Stage 1 Guidance Loss`: `AuxiliaryHead` の出力と `y` を用いたLossを実装。

3. **SamadhiTrainer (Refactor)**:

    * 4ステージカリキュラムの制御。

    * 各ステージ開始時の `Freeze`/`Unfreeze` 処理の実装。

    * Noise Generation Strategy（Augmenter利用, Drunk Mode切り替え, `use_label_guidance` フラグに基づくMismatch生成）の組み込み。

**Phase 4 Tests (System Tests)**:

* **Stage Switching**: 指定したステージに対応する戻り値（Loss構成など）が返るか。

* **Freeze Verification**: 各ステージで更新されるべきパラメータのみ `requires_grad=True` になっているか。

* **Overfitting Check**: 極小データセットでLossが収束することを確認（Sanity Check）。

* **Label Guidance Impact**: `use_label_guidance=True` の場合と `False` の場合で、Stage 1の潜在空間構造に有意な差が出るか。

## 7. Vicaraバリエーションとインターフェース設計 (Vicara Variants & Interface)

### 7.1. Vicaraバリエーション

v4では以下の3種類のVicaraをサポートします。

| クラス | 説明 |
|--------|------|
| `StandardVicara` | 単一Refinerで状態更新。最もシンプル。 |
| `WeightedVicara` | 複数Refinerの重み付け合成（将来拡張用）。 |
| `ProbeSpecificVicara` | Vitakkaの勝者Probe/確率に基づきRefinerを選択。 |

### 7.2. Vicaraインターフェース設計

v4では**責務の明確な分離**を実現するため、以下の設計方針を採用します。

#### 責務分離の原則

| コンポーネント | 責務 |
|---------------|------|
| **Vicara** | 1ステップの状態更新のみ (`s_t → s_{t+1}`) |
| **SamathaEngine** | ループ制御、SantanaLog記録、context受け渡し |
| **Sati** | 停止判定（SantanaLogを参照） |

#### BaseVicara インターフェース

```python
class BaseVicara(nn.Module, ABC):
    def forward(self, s_t: torch.Tensor, context: Dict[str, Any] = None) -> torch.Tensor:
        """
        1ステップの状態更新を行う。

        Args:
            s_t: 現在の状態 (Batch, Dim)
            context: Vitakkaからのメタ情報（オプション）
                     - ProbeSpecificVicaraで使用: probs, winner_id
                     - Standard/WeightedVicaraでは無視される

        Returns:
            s_{t+1}: 更新後の状態 (Batch, Dim)
        """
        residual = self._refine_step(s_t, context)
        return self.update_state(s_t, residual)

    @abstractmethod
    def _refine_step(self, s_t: torch.Tensor, context: Dict[str, Any] = None) -> torch.Tensor:
        pass
```

#### SamathaEngine のVicaraループ

```python
class SamathaEngine(nn.Module):
    def _run_vicara_loop(self, s0: torch.Tensor, vitakka_meta: Dict) -> Tuple[torch.Tensor, SantanaLog]:
        """
        Vicaraループを実行し、収束状態とログを返す。

        - ループ制御: SamathaEngineが管理
        - 状態更新: Vicaraに委譲
        - 停止判定: Satiに委譲
        - ログ記録: SantanaLogに記録
        """
        santana = SantanaLog()
        s_t = s0
        santana.add(s_t)

        for step in range(self.config.max_steps):
            # Vicaraに1ステップ更新を委譲（ProbeVicara用にcontextも渡す）
            s_t = self.vicara(s_t, context=vitakka_meta)
            santana.add(s_t)

            # Satiに停止判定を委譲
            should_stop, info = self.sati(s_t, santana)
            if should_stop:
                break

        return s_t, santana
```

#### 設計の利点

1. **シンプルなインターフェース**: Standard/WeightedVicaraは`context=None`で動作
2. **拡張性**: ProbeSpecificVicaraはVitakkaのcontextを活用可能
3. **テスト容易性**: Vicaraを単体で1ステップずつテスト可能
4. **責務明確化**: ループ/ログ/停止判定がVicaraから分離

---
以上が、Samadhi Framework v4.0の最終設計仕様書となります。
このドキュメントは、実装チームがV4.0アーキテクチャを正確かつ効率的に構築するためのガイドラインとして機能します。
