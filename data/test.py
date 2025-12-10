import numpy as np
import torch
import matplotlib.pyplot as plt


def generate_meditation_data(batch_size=1, seq_len=64, dim=64):
    """
    Satipatthana Model検証用データジェネレータ
    - Target: 隠された真実 (正弦波の規則性)
    - Distraction: 強力な妄想 (ランダムなスパイクや別の周期)
    - Noise: 背景雑音 (ガウシアンノイズ)
    """
    t = np.linspace(0, 4 * np.pi, dim)  # 時間軸 (次元方向)

    # 1. Target Signal (呼吸/集中対象): ゆっくりした綺麗な波
    # Probeが探すべき対象
    target = np.sin(t)

    # 2. Distraction (雑念): 速くて不規則な波、あるいは矩形波
    distraction = np.sin(3 * t + np.random.rand()) * np.random.choice([0, 1.5], size=dim)

    # 3. Noise (感覚ノイズ)
    noise = np.random.normal(0, 0.3, dim)

    # 入力データ作成: ターゲットは埋もれている
    # Input = 0.3*Target + 0.5*Distraction + 0.2*Noise
    x_input = 0.3 * target + 0.5 * distraction + 0.2 * noise

    # PyTorch Tensor化
    x_tensor = torch.FloatTensor(x_input).unsqueeze(0)  # (Batch, Dim)
    target_tensor = torch.FloatTensor(target).unsqueeze(0)

    return x_tensor, target_tensor, t


# # データの可視化
# x, target, t = generate_meditation_data()

# plt.figure(figsize=(10, 4))
# plt.plot(t, target.numpy().flatten(), label="Target (Essence)", linewidth=3, color="green")
# plt.plot(t, x.numpy().flatten(), label="Input (Chaos)", alpha=0.6, color="gray")
# plt.title("Satipatthana Task: Can you extract the Green line from the Gray noise?")
# plt.legend()
# plt.grid(True)
# plt.show()
