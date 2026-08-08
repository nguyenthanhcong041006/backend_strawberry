import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import math
import sys
from pathlib import Path

# Fix import paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avocado.stage4_training.models.numeric_baselines import NumericBaselineMLP, NumericBaselineGRU
from avocado.stage4_training.models.image_baselines import ImageBaselineViT
from avocado.stage4_training.models.vit_mamba import ViTMambaRULModel
from avocado.stage4_training.models.fusion_models import EarlyFusionModel, LateFusionModel, MBTFusionModel

class DummyAvocadoDataset(Dataset):
    def __init__(self, num_samples: int = 16, seq_len: int = 24):
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        images_seq = torch.randn(self.seq_len, 3, 224, 224)
        numeric_seq = torch.randn(self.seq_len, 3)
        rul = torch.tensor([float(torch.randint(0, 100, (1,)).item())])
        return images_seq, numeric_seq, rul

def get_model(model_type: str, device: torch.device) -> nn.Module:
    if model_type == "numeric_mlp":
        return NumericBaselineMLP().to(device)
    elif model_type == "numeric_gru":
        return NumericBaselineGRU().to(device)
    elif model_type == "image_vit":
        return ImageBaselineViT().to(device)
    elif model_type == "vit_mamba":
        return ViTMambaRULModel().to(device)
    elif model_type == "early_fusion":
        return EarlyFusionModel().to(device)
    elif model_type == "late_fusion":
        return LateFusionModel().to(device)
    elif model_type == "mbt_fusion":
        return MBTFusionModel().to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train(model_type: str, epochs: int, batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{model_type.upper()}] Starting Dummy Training on {device}")
    
    model = get_model(model_type, device)
    
    dataset = DummyAvocadoDataset(num_samples=8, seq_len=24) # match model defaults
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.HuberLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for images, numerics, ruls in dataloader:
            images, numerics, ruls = images.to(device), numerics.to(device), ruls.to(device)
            
            optimizer.zero_grad()
            
            if model_type in ["numeric_mlp", "numeric_gru"]:
                outputs = model(numerics)
            elif model_type in ["image_vit", "vit_mamba"]:
                outputs = model(images)
            else: # fusion
                outputs = model(images, numerics)
                
            loss = criterion(outputs, ruls)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            all_preds.append(outputs.detach().cpu())
            all_targets.append(ruls.cpu())
            
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        
        mae = torch.mean(torch.abs(preds - targets)).item()
        rmse = math.sqrt(torch.mean((preds - targets) ** 2).item())
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    print(f"[{model_type.upper()}] Finished.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                        choices=["numeric_mlp", "numeric_gru", "image_vit", "vit_mamba", "early_fusion", "late_fusion", "mbt_fusion"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    
    train(args.model, args.epochs, args.batch_size)
