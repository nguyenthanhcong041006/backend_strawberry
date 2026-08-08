import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Warning: mamba_ssm not found. Falling back to an LSTM sequence encoder. Please install mamba_ssm for the true Mamba temporal model.")

class ViTMambaRULModel(nn.Module):
    """
    Multimodal sequence model:
    ViT (Spatial) -> Mamba (Temporal) -> Regression Head
    """
    def __init__(self, freeze_backbone: bool = True, mamba_d_state: int = 16, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        # ---- 1. Spatial Encoder: ViT ----
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)
        self.feature_dim = self.vit.hidden_dim  # 768
        self.vit.heads = nn.Identity()

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        # ---- 2. Temporal Encoder: Mamba (or LSTM Fallback) ----
        if HAS_MAMBA:
            self.temporal_encoder = Mamba(
                d_model=self.feature_dim,
                d_state=mamba_d_state,
                d_conv=4,
                expand=2
            )
        else:
            # Fallback to GRU if Mamba is missing
            self.temporal_encoder = nn.GRU(
                input_size=self.feature_dim,
                hidden_size=self.feature_dim,
                batch_first=True
            )

        # ---- 3. Regression Head ----
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, images_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images_seq: (batch_size, seq_len, 3, 224, 224)
        Returns:
            rul: (batch_size, 1)
        """
        batch_size, seq_len, C, H, W = images_seq.size()
        
        # Step 1: Spatial features
        images_reshaped = images_seq.view(batch_size * seq_len, C, H, W)
        spatial_features = self.vit(images_reshaped)  # (B*S, 768)
        spatial_features = spatial_features.view(batch_size, seq_len, self.feature_dim)
        
        # Step 2: Temporal modeling
        if HAS_MAMBA:
            temporal_out = self.temporal_encoder(spatial_features) # (B, S, 768)
            last_out = temporal_out[:, -1, :] # Take the last element
        else:
            temporal_out, _ = self.temporal_encoder(spatial_features)
            last_out = temporal_out[:, -1, :]
            
        # Step 3: Regression
        return self.regressor(last_out)

if __name__ == "__main__":
    print("Testing ViT + Mamba Temporal Model...")
    batch_size = 2
    seq_len = 5
    dummy_images = torch.randn(batch_size, seq_len, 3, 224, 224)
    
    model = ViTMambaRULModel()
    out = model(dummy_images)
    print(f"Output Shape: {out.shape} (Expected: [{batch_size}, 1])")
    print(f"Using Mamba: {HAS_MAMBA}")
