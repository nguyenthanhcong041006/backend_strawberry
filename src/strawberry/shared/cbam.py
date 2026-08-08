"""
CBAM: Convolutional Block Attention Module

Paper: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
https://arxiv.org/abs/1807.06521

CBAM sequentially applies:
  1. Channel Attention Module (CAM) — highlights "what" feature channels matter
  2. Spatial Attention Module (SAM) — highlights "where" in the spatial map matters

For the Strawberry RUL prediction pipeline, CBAM sits between the CNN backbone
feature maps and the temporal model, helping the network focus on the most
salient visual features (e.g., mold spots, color changes, texture degradation)
that correlate with remaining shelf life.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.

    Uses both average-pooling and max-pooling to aggregate spatial information,
    then a shared MLP to produce per-channel weights.

    Args:
        in_channels (int): Number of input channels (e.g. 1280 for EfficientNet-B0/MobileNetV2).
        reduction_ratio (int): Reduction ratio for the bottleneck MLP (default: 16).
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(in_channels // reduction_ratio, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map of shape (B, C, H, W).

        Returns:
            Attended feature map of shape (B, C, H, W).
        """
        b, c, _, _ = x.size()

        # Average-pool path
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.mlp(avg_out)

        # Max-pool path
        max_out = self.max_pool(x).view(b, c)
        max_out = self.mlp(max_out)

        # Fuse and broadcast
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)

        return x * channel_att


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    Applies average-pooling and max-pooling along the channel axis, concatenates
    the results, and applies a convolution to produce a 2D spatial attention map.

    Args:
        kernel_size (int): Size of the convolution kernel (default: 7).
    """

    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map of shape (B, C, H, W).

        Returns:
            Attended feature map of shape (B, C, H, W).
        """
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)

        # Concatenate and convolve
        pooled = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        spatial_att = self.sigmoid(self.conv(pooled))  # (B, 1, H, W)

        return x * spatial_att


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Sequentially applies Channel Attention followed by Spatial Attention
    to refine CNN feature maps before temporal aggregation.

    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for channel attention bottleneck.
        kernel_size (int): Kernel size for spatial attention convolution.
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
    ):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map of shape (B, C, H, W).

        Returns:
            Refined feature map of shape (B, C, H, W).
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing CBAM module...")

    # Simulate feature maps from EfficientNet-B0 / MobileNetV2
    # after the final conv layer: 1280 channels, ~7x7 spatial for 224x224 input
    dummy_features = torch.randn(2, 1280, 7, 7)

    cbam = CBAM(in_channels=1280, reduction_ratio=16, kernel_size=7)
    output = cbam(dummy_features)

    print(f"  Input shape:  {dummy_features.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters:   {sum(p.numel() for p in cbam.parameters()):,}")
    print("CBAM test passed!")
