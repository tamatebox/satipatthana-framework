from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from src.model.samadhi import SamadhiModel


class LstmSamadhiModel(SamadhiModel):
    """
    LSTM Samadhi Model (Sequence Samadhi).

    Designed for time-series or sequential data.
    Compresses input sequence into a latent vector using an LSTM Adapter,
    performs "search (Vitakka)" and "purification (Vicara)" within that latent space,
    and reconstructs the original sequence using an LSTM Decoder.
    """

    def __init__(self, config: Dict[str, Any]):
        # Load sequence-specific configs
        self.input_dim = config.get("input_dim")  # Number of features per timestep
        self.seq_len = config.get("seq_len")  # Length of the sequence window

        if self.input_dim is None or self.seq_len is None:
            raise ValueError("Config must contain 'input_dim' and 'seq_len' for LstmSamadhiModel.")

        super().__init__(config)

        # Replace Adapters with LSTM versions
        self.vitakka.adapter = self._build_lstm_adapter()
        # Note: self.decoder is built by the base class calling _build_decoder()

    def _build_lstm_adapter(self) -> nn.Module:
        """
        LSTM Encoder for Vitakka.
        Input (Batch, Seq_Len, Input_Dim) -> Latent Vector (Batch, Dim)
        """
        return LstmEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.config.get("adapter_hidden_dim", 128),
            latent_dim=self.dim,
            num_layers=self.config.get("lstm_layers", 1),
        )

    def _build_decoder(self) -> nn.Module:
        """
        LSTM Decoder.
        Latent Vector (Batch, Dim) -> Output Sequence (Batch, Seq_Len, Input_Dim)
        """
        return LstmDecoder(
            latent_dim=self.dim,
            hidden_dim=self.config.get("adapter_hidden_dim", 128),
            output_dim=self.input_dim,
            seq_len=self.seq_len,
            num_layers=self.config.get("lstm_layers", 1),
        )


class LstmEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.Tanh()  # Normalize to [-1, 1] for Samadhi latent space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq_Len, Input_Dim)
        _, (h_n, _) = self.lstm(x)
        # h_n: (Num_Layers, Batch, Hidden_Dim) -> Take last layer
        last_hidden = h_n[-1]
        z = self.fc(last_hidden)
        return self.activation(z)


class LstmDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, seq_len: int, num_layers: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.num_layers = num_layers  # Store num_layers

        # Map latent z back to LSTM initial hidden state
        self.fc_start = nn.Linear(latent_dim, hidden_dim)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (Batch, Latent_Dim)
        batch_size = z.size(0)

        # Initialize hidden state from z
        # (Batch, Hidden_Dim) -> (1, Batch, Hidden_Dim) -> (Num_Layers, Batch, Hidden_Dim)
        h_0_single = self.fc_start(z).unsqueeze(0)
        h_0 = h_0_single.repeat(self.num_layers, 1, 1)

        # c_0 also needs to match (Num_Layers, Batch, Hidden_Dim)
        c_0 = torch.zeros_like(h_0)

        # Prepare input for LSTM (Repeat z or use zeros/start token)
        # Strategy: Repeat the transformed z as input for every timestep
        # (Batch, Seq_Len, Hidden_Dim)
        lstm_input = self.fc_start(z).unsqueeze(1).repeat(1, self.seq_len, 1)

        # Forward
        output, _ = self.lstm(lstm_input, (h_0, c_0))
        # output: (Batch, Seq_Len, Hidden_Dim)

        # Map to original dimension
        recon_seq = self.fc_out(output)
        # recon_seq: (Batch, Seq_Len, Output_Dim)

        return recon_seq
