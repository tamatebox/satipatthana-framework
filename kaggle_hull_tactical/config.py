from samadhi.configs.main import SamadhiConfig
from samadhi.configs.enums import AdapterType, DecoderType, VicaraType
from samadhi.configs.objectives import ObjectiveConfig

# ==== ハイパーパラメータ設定 ====
TOTAL_DIM = 95  # 特徴量98 + Lag + Target
LATENT_DIM = 64  # Samadhi内部の思考次元

CONFIG_DICT = {
    "dim": LATENT_DIM,
    "seed": 42,
    # "labels": [str(i) for i in range(10)], # For classification tasks, not directly relevant here
    "adapter": {"type": AdapterType.MLP.value, "input_dim": TOTAL_DIM, "adapter_hidden_dim": 256, "dropout": 0.2},
    "vitakka": {
        "n_probes": 16,
        "gate_threshold": 0.3,
        "softmax_temp": 0.1,
        "training_attention_mode": "soft",
        "prediction_attention_mode": "hard",
    },
    "vicara": {
        "type": VicaraType.PROBE_SPECIFIC.value,  # ProbeVicaraに変更
        "refine_steps": 8,
        "refiner_type": "mlp",
        "inertia": 0.6,
        "n_probes": 16,  # Vitakkaのn_probesと一致させる
    },
    "decoder": {"type": DecoderType.RECONSTRUCTION.value, "input_dim": TOTAL_DIM, "decoder_hidden_dim": 256},
    "objective": {
        "stability_coeff": 0.05,
        "entropy_coeff": 0.01,
        "balance_coeff": 0.9,
        "recon_coeff": 1.0,  # Reconstruction is key for inpainting
    },
}

config = SamadhiConfig.from_dict(CONFIG_DICT)
