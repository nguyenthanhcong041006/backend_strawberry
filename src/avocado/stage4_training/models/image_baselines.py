import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ImageBaselineViT(nn.Module):
    """
    Image baseline using a pretrained ViT as spatial extractor and 
    a simple temporal pool (e.g., mean over sequence) followed by a regression head.
    Expects input: (batch_size, seq_len, C, H, W)
    """
    def __init__(self, freeze_backbone: bool = True, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        # Load Pretrained ViT
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)
        self.feature_dim = self.vit.hidden_dim  # Usually 768 for vit_b_16
        
        # We only need the feature extraction part of ViT, replace the head with Identity
        self.vit.heads = nn.Identity()

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        # Regression Head over temporally pooled features
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """ Extract spatial features: (N, 3, 224, 224) -> (N, feature_dim) """
        return self.vit(images)

    def forward(self, images_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images_seq: (batch_size, seq_len, 3, 224, 224)
        Returns:
            rul: (batch_size, 1)
        """
        batch_size, seq_len, C, H, W = images_seq.size()
        
        # Reshape to process all frames
        images_reshaped = images_seq.view(batch_size * seq_len, C, H, W)
        
        # Extract features
        spatial_features = self._extract_features(images_reshaped)  # (B*S, 768)
        spatial_features = spatial_features.view(batch_size, seq_len, self.feature_dim)
        
        # Temporal pooling (average over sequence)
        pooled_features = spatial_features.mean(dim=1)  # (B, 768)
        
        # Predict
        return self.regressor(pooled_features)

if __name__ == "__main__":
    print("Testing Image Baseline ViT...")
    batch_size = 2
    seq_len = 5
    dummy_images = torch.randn(batch_size, seq_len, 3, 224, 224)
    
    model = ImageBaselineViT()
    out = model(dummy_images)
    print(f"ViT Output Shape: {out.shape} (Expected: [{batch_size}, 1])")
