from typing import Dict, Tuple, List, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class Vitakka(nn.Module):
    """
    Vitakka (Initial Application/Search) Component.

    役割: 入力ストリームに対して「意図（Probe）」を検索し、初期状態 S0 を形成する。
    思考の粗い段階（Coarse-grained thinking）を担当。
    HardとSoftの両方のモードをこの単一クラス内で切り替えて処理する。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.dim = config["dim"]
        self.n_probes = config["n_probes"]

        # Probe (Concepts) Definition
        self.probes = nn.Parameter(torch.randn(self.n_probes, self.dim))
        self.probes.requires_grad = config.get("probe_trainable", True)
        self._normalize_probes()

        # Adapter (Manasikāra - Attention/Adaptation)
        # 入力をProbe空間に適応させる
        self.adapter = self._build_adapter()

    def _build_adapter(self) -> nn.Module:
        """
        Build the adapter network.
        Identity mapping: assumes input is already normalized externally.
        Strict separation of concerns.
        """
        return nn.Identity()

    def _normalize_probes(self):
        """ProbesをL2正規化"""
        with torch.no_grad():
            self.probes.div_(torch.norm(self.probes, dim=1, keepdim=True))

    def load_probes(self, pretrained_probes: torch.Tensor):
        """外部からProbeをロード"""
        if pretrained_probes.shape != self.probes.shape:
            raise ValueError(f"Shape mismatch: expected {self.probes.shape}, got {pretrained_probes.shape}")
        with torch.no_grad():
            self.probes.copy_(pretrained_probes)
            self._normalize_probes()

    def _generate_hard_s0(
        self, x_adapted: torch.Tensor, probs: torch.Tensor, raw_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """[Hard Mode] 勝者プローブを1つだけ選択し、明確なゲート判定を行います。"""
        # Max Score Selection
        max_raw_score, winner_idx = torch.max(raw_scores, dim=1)
        is_gate_open = max_raw_score > self.config["gate_threshold"]

        # Winner Probe
        winner_probes = self.probes[winner_idx]

        # Mix Input & Probe
        alpha = self.config.get("mix_alpha", 0.5)
        s0_candidate = alpha * x_adapted + (1 - alpha) * winner_probes

        # Gate Application
        gate_mask = is_gate_open.float().unsqueeze(1)
        s0 = s0_candidate * gate_mask

        confidence = probs.gather(1, winner_idx.unsqueeze(1)).squeeze(1)

        return s0, {
            "winner_id": winner_idx,
            "raw_score": max_raw_score,
            "gate_open": is_gate_open,
            "confidence": confidence,
        }

    def _generate_soft_s0(
        self, x_adapted: torch.Tensor, probs: torch.Tensor, raw_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Soft Attention Vitakka.
        確率分布に従ってProbeを重み付き平均してS0とする。学習（微分可能）向け。
        """
        # Weighted Probe Sum
        # (Batch, N) @ (N, Dim) -> (Batch, Dim)
        weighted_probes = torch.matmul(probs, self.probes)

        # Mix Input & Weighted Probe
        alpha = self.config.get("mix_alpha", 0.5)
        s0_candidate = alpha * x_adapted + (1 - alpha) * weighted_probes

        # Soft Gate Logic (Differentiable)
        # 期待スコア (Weighted Average Score) を使用
        avg_score = torch.sum(raw_scores * probs, dim=1)
        gate_logits = (avg_score - self.config["gate_threshold"]) * 10.0
        gate_mask = torch.sigmoid(gate_logits).unsqueeze(1)

        s0 = s0_candidate * gate_mask

        # Log Metadata (for consistency, calculate winner as usual)
        winner_idx = torch.argmax(probs, dim=1)
        max_raw_score = torch.max(raw_scores, dim=1)[0]  # Logging uses max
        confidence = torch.max(probs, dim=1)[0]

        return s0, {
            "winner_id": winner_idx,
            "raw_score": max_raw_score,
            "gate_open": max_raw_score > self.config["gate_threshold"],  # Logging logic
            "confidence": confidence,
        }

    def forward(self, x_input: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Vitakka Process: Search & Select.

        Args:
            x_input: (Batch, Dim)

        Returns:
            s0: (Batch, Dim) - Initial State
            metadata: Log info (winner, confidence, etc.)
        """
        # 1. Adapt Input
        x_adapted = self.adapter(x_input)
        x_norm = F.normalize(x_adapted, p=2, dim=1)

        # 2. Compute Similarity
        # (Batch, Dim) @ (n_probes, Dim).T -> (Batch, n_probes)
        raw_scores = torch.matmul(x_norm, self.probes.T)

        # 3. Compute Probabilities
        temp = self.config.get("softmax_temp", 0.2)
        probs = F.softmax(raw_scores / temp, dim=1)

        # 4. Generate S0 (Mode-based switching)
        mode = self.config.get("attention_mode", "hard")
        if mode == "soft":
            s0, partial_meta = self._generate_soft_s0(x_adapted, probs, raw_scores)
        else:
            s0, partial_meta = self._generate_hard_s0(x_adapted, probs, raw_scores)

        # 5. Metadata Construction
        # Convert indices to labels if on CPU/Single item, otherwise keep tensor
        # Here we just keep tensor logic for batch efficiency

        metadata = {
            "winner_id": partial_meta["winner_id"],
            "confidence": partial_meta["confidence"],
            "raw_score": partial_meta["raw_score"],
            "gate_open": partial_meta["gate_open"],
            "probs": probs,
            "raw_scores": raw_scores,
        }

        return s0, metadata
