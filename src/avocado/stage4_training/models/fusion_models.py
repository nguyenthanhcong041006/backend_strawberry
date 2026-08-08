import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

def get_temporal_encoder(input_dim: int, d_state: int = 16):
    if HAS_MAMBA:
        return Mamba(d_model=input_dim, d_state=d_state, d_conv=4, expand=2)
    else:
        return nn.GRU(input_size=input_dim, hidden_size=input_dim, batch_first=True)

class EarlyFusionModel(nn.Module):
    """
    Concatenates ViT features and numeric features at each hour,
    then processes the sequence.
    """
    def __init__(self, num_features: int = 3, freeze_backbone: bool = True, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)
        self.vit_dim = self.vit.hidden_dim
        self.vit.heads = nn.Identity()

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.fused_dim = self.vit_dim + num_features
        self.temporal_encoder = get_temporal_encoder(self.fused_dim)
        
        self.regressor = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, images_seq: torch.Tensor, numeric_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, C, H, W = images_seq.size()
        
        images_reshaped = images_seq.view(batch_size * seq_len, C, H, W)
        spatial_features = self.vit(images_reshaped).view(batch_size, seq_len, self.vit_dim)
        
        fused_seq = torch.cat([spatial_features, numeric_seq], dim=-1)
        
        if HAS_MAMBA:
            temporal_out = self.temporal_encoder(fused_seq)
            last_out = temporal_out[:, -1, :]
        else:
            temporal_out, _ = self.temporal_encoder(fused_seq)
            last_out = temporal_out[:, -1, :]
            
        return self.regressor(last_out)

class LateFusionModel(nn.Module):
    """
    Separate temporal encoders for images and numeric data, 
    concatenated before regression.
    """
    def __init__(self, num_features: int = 3, freeze_backbone: bool = True, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)
        self.vit_dim = self.vit.hidden_dim
        self.vit.heads = nn.Identity()

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.image_temporal = get_temporal_encoder(self.vit_dim)
        self.numeric_temporal = get_temporal_encoder(num_features)
        
        self.regressor = nn.Sequential(
            nn.Linear(self.vit_dim + num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, images_seq: torch.Tensor, numeric_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, C, H, W = images_seq.size()
        
        # Image stream
        images_reshaped = images_seq.view(batch_size * seq_len, C, H, W)
        spatial_features = self.vit(images_reshaped).view(batch_size, seq_len, self.vit_dim)
        
        if HAS_MAMBA:
            img_temp_out = self.image_temporal(spatial_features)[:, -1, :]
            num_temp_out = self.numeric_temporal(numeric_seq)[:, -1, :]
        else:
            img_temp_out, _ = self.image_temporal(spatial_features)
            img_temp_out = img_temp_out[:, -1, :]
            num_temp_out, _ = self.numeric_temporal(numeric_seq)
            num_temp_out = num_temp_out[:, -1, :]
            
        fused = torch.cat([img_temp_out, num_temp_out], dim=-1)
        return self.regressor(fused)

class MBTFusionModel(nn.Module):
    """
    Multimodal Bottleneck Transformer (MBT) style fusion.
    Uses learnable bottleneck tokens to exchange information between streams.
    Note: Highly simplified approximation using Cross-Attention for demonstration.
    """
    def __init__(self, num_features: int = 3, freeze_backbone: bool = True, hidden_dim: int = 128, num_bottlenecks: int = 4):
        super().__init__()
        # Skeleton implementation
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.vit_dim = self.vit.hidden_dim
        self.vit.heads = nn.Identity()
        if freeze_backbone:
            for p in self.vit.parameters(): p.requires_grad = False
            
        self.num_proj = nn.Linear(num_features, self.vit_dim)
        self.bottlenecks = nn.Parameter(torch.randn(1, num_bottlenecks, self.vit_dim))
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.vit_dim, num_heads=4, batch_first=True)
        self.regressor = nn.Linear(self.vit_dim, 1)

    def forward(self, images_seq: torch.Tensor, numeric_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, C, H, W = images_seq.size()
        
        spatial_features = self.vit(images_seq.view(batch_size * seq_len, C, H, W)).view(batch_size, seq_len, self.vit_dim)
        numeric_features = self.num_proj(numeric_seq)
        
        # Simple cross-modal interaction through bottlenecks
        bottlenecks = self.bottlenecks.expand(batch_size, -1, -1)
        
        # Images to bottlenecks
        bn_img, _ = self.cross_attn(bottlenecks, spatial_features, spatial_features)
        # Numeric to bottlenecks
        bn_num, _ = self.cross_attn(bn_img, numeric_features, numeric_features)
        
        # Regress from pooled bottlenecks
        pooled = bn_num.mean(dim=1)
        return self.regressor(pooled)

if __name__ == "__main__":
    print("Testing Fusion Models...")
    b, s = 2, 5
    d_img = torch.randn(b, s, 3, 224, 224)
    d_num = torch.randn(b, s, 3)
    
    early = EarlyFusionModel()
    print(f"Early Fusion Output: {early(d_img, d_num).shape}")
    
    late = LateFusionModel()
    print(f"Late Fusion Output: {late(d_img, d_num).shape}")
    
    mbt = MBTFusionModel()
    print(f"MBT Fusion Output: {mbt(d_img, d_num).shape}")
