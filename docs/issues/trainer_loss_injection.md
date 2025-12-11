# Issue: Trainer への Loss オブジェクト注入機能

## 概要

現状の `SatipatthanaTrainer` は Loss オブジェクトを内部で生成しているが、外部から注入できるようにしたい。

## 現状

```python
# trainer.py
self.vipassana_objective = VipassanaObjective()
self.guidance_loss = GuidanceLoss(task_type=task_type)
self.stability_loss = StabilityLoss()
self.diversity_loss = ProbeDiversityLoss()
```

Loss は内部で固定的に生成され、カスタマイズできない。

## 提案

```python
trainer = SatipatthanaTrainer(
    model=system,
    args=training_args,
    train_dataset=train_dataset,
    # カスタム Loss を渡せるようにする
    diversity_loss=MyCustomDiversityLoss(margin=0.3),
    stability_loss=MyCustomStabilityLoss(),
    # 省略時はデフォルトを使用
)
```

## 実装案

```python
def __init__(
    self,
    ...
    # Loss オブジェクト (None = デフォルト使用)
    vipassana_objective: Optional[VipassanaObjective] = None,
    guidance_loss: Optional[GuidanceLoss] = None,
    stability_loss: Optional[StabilityLoss] = None,
    diversity_loss: Optional[ProbeDiversityLoss] = None,
    ...
):
    # None の場合はデフォルトを生成
    self.vipassana_objective = vipassana_objective or VipassanaObjective()
    self.guidance_loss = guidance_loss or GuidanceLoss(task_type=task_type)
    self.stability_loss = stability_loss or StabilityLoss()
    self.diversity_loss = diversity_loss or ProbeDiversityLoss()
```

## メリット

- カスタム Loss の実験が容易になる
- テスト時にモック Loss を注入できる
- 依存性注入パターンに従った設計

## 懸念点

- 引数がさらに増える
- Loss インターフェースの定義が必要になる可能性

## 優先度

低〜中（現状 weight=0 で無効化できるため）

## 関連

- `ProbeDiversityLoss` 追加時に発生した議論
