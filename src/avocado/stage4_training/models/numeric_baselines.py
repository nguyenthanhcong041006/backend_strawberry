import torch
import torch.nn as nn

class NumericBaselineMLP(nn.Module):
    """
    Numeric baseline using a simple MLP on flattened sequence.
    Expects input: (batch_size, seq_len, num_features)
    """
    def __init__(self, seq_len: int = 24, num_features: int = 3, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        input_dim = seq_len * num_features
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, numeric_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            numeric_seq: (batch_size, seq_len, num_features)
        Returns:
            rul: (batch_size, 1)
        """
        batch_size = numeric_seq.size(0)
        flattened = numeric_seq.view(batch_size, -1)
        return self.net(flattened)


class NumericBaselineGRU(nn.Module):
    """
    Numeric baseline using a GRU over the sequence.
    Expects input: (batch_size, seq_len, num_features)
    """
    def __init__(self, num_features: int = 3, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, numeric_seq: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(numeric_seq)
        last_out = gru_out[:, -1, :]  # Take the last time step
        return self.regressor(last_out)

if __name__ == "__main__":
    print("Testing Numeric Baselines...")
    batch_size = 2
    seq_len = 24
    num_features = 3
    dummy_seq = torch.randn(batch_size, seq_len, num_features)
    
    mlp = NumericBaselineMLP(seq_len=seq_len, num_features=num_features)
    mlp_out = mlp(dummy_seq)
    print(f"MLP Output Shape: {mlp_out.shape} (Expected: [{batch_size}, 1])")
    
    gru = NumericBaselineGRU(num_features=num_features)
    gru_out = gru(dummy_seq)
    print(f"GRU Output Shape: {gru_out.shape} (Expected: [{batch_size}, 1])")
