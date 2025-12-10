import torch
import torch.nn as nn
from typing import Dict, Any
from satipatthana.components.decoders.base import BaseDecoder
from satipatthana.configs.decoders import LstmDecoderConfig, SimpleSequenceDecoderConfig
from satipatthana.configs.factory import create_decoder_config


class LstmDecoder(BaseDecoder):
    """
    LSTM Decoder.
    Latent Vector (Batch, Dim) -> Output Sequence (Batch, Seq_Len, Input_Dim)
    """

    def __init__(self, config: LstmDecoderConfig):
        if isinstance(config, dict):
            # Ensure type is set so factory creates correct config class
            # Force type if it doesn't match expected type for this class
            if config.get("type") != "lstm":
                config["type"] = "lstm"
            config = create_decoder_config(config)

        super().__init__(config)

        self.output_dim = self.config.output_dim
        self.seq_len = self.config.seq_len
        self.hidden_dim = self.config.decoder_hidden_dim
        self.num_layers = self.config.lstm_layers

        # Note: Original code checked for None. Config has defaults.
        # But if defaults are generic (e.g. 50), and user didn't provide seq_len...
        # The user should provide seq_len.

        # Map latent z back to LSTM initial hidden state
        self.fc_start = nn.Linear(self.dim, self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (Batch, Latent_Dim)
        batch_size = z.size(0)

        # Initialize hidden state from z
        h_0_single = self.fc_start(z).unsqueeze(0)
        h_0 = h_0_single.repeat(self.num_layers, 1, 1)

        # c_0 also needs to match
        c_0 = torch.zeros_like(h_0)

        # Prepare input for LSTM (Repeat z or use zeros/start token)
        # Strategy: Repeat the transformed z as input for every timestep
        lstm_input = self.fc_start(z).unsqueeze(1).repeat(1, self.seq_len, 1)

        # Forward
        output, _ = self.lstm(lstm_input, (h_0, c_0))
        # output: (Batch, Seq_Len, Hidden_Dim)

        # Map to original dimension
        recon_seq = self.fc_out(output)
        # recon_seq: (Batch, Seq_Len, Output_Dim)

        return recon_seq


class SimpleSequenceDecoder(BaseDecoder):
    """
    Simple MLP-based Decoder for Sequences.
    Decodes latent z back to (Batch, Seq, Input)
    Used as default for Transformer Satipatthana for simplicity.
    """

    def __init__(self, config: SimpleSequenceDecoderConfig):
        if isinstance(config, dict):
            if config.get("type") != "simple_sequence":
                config["type"] = "simple_sequence"
            config = create_decoder_config(config)

        super().__init__(config)

        self.output_dim = self.config.output_dim
        self.seq_len = self.config.seq_len
        self.hidden_dim = self.config.decoder_hidden_dim

        # Map latent to flattened sequence size
        self.fc = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.seq_len * self.output_dim)
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: (Batch, Latent)
        out = self.fc(s)  # (Batch, Seq * Output)
        out = out.view(-1, self.seq_len, self.output_dim)
        return out
