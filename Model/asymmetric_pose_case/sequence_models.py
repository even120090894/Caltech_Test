import torch
from torch import nn

from .config import SequenceConfig


class LSTMSequenceModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, config: SequenceConfig) -> None:
        super().__init__()
        self.output_dim = hidden_dim * (2 if config.bidirectional else 1)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(sequence)
        return outputs


class GRUSequenceModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, config: SequenceConfig) -> None:
        super().__init__()
        self.output_dim = hidden_dim * (2 if config.bidirectional else 1)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(sequence)
        return outputs


def build_sequence_model(
    model_type: str, input_dim: int, hidden_dim: int, config: SequenceConfig
) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "lstm":
        return LSTMSequenceModel(input_dim, hidden_dim, config)
    if model_type == "gru":
        return GRUSequenceModel(input_dim, hidden_dim, config)
    # moe? KAN?
    raise ValueError(
        f"Unsupported sequence model '{model_type}'. Add a module with the same forward interface."
    )
