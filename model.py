import torch
import torch.nn as nn
import timm
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class FusionDeepfakeDetector(nn.Module):
    def __init__(self, hidden_dim=512, n_heads=8, n_layers=2):
        super().__init__()
        # 1. Load pretrained backbones
        self.xception = timm.create_model('xception', pretrained=True)      # Xception backbone :contentReference[oaicite:3]{index=3}
        self.effnet  = timm.create_model('efficientnet_b0', pretrained=True) # EfficientNet-B0 backbone :contentReference[oaicite:4]{index=4}

        # Remove existing classifier heads
        self.xception.reset_classifier(0)
        self.effnet.reset_classifier(0)

        # 2. Linear projections to common hidden_dim
        self.proj_x = nn.Linear(self.xception.num_features, hidden_dim)
        self.proj_e = nn.Linear(self.effnet.num_features,  hidden_dim)

        # 3. Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='relu'
        )  # standard transformer block :contentReference[oaicite:5]{index=5}

        self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. Classification head: pool + MLP + sigmoid
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # output between 0 and 1 :contentReference[oaicite:6]{index=6}
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        fx = self.xception(x)              # [B, C1]
        fe = self.effnet(x)                # [B, C2]
        px = self.proj_x(fx)               # [B, D]
        pe = self.proj_e(fe)               # [B, D]

        # Stack into a sequence of length=2: shape [S=2, B, D]
        seq = torch.stack([px, pe], dim=0)
        tr_out = self.transformer(seq)     # [2, B, D]

        # Pool across sequence dim → [B, D]
        feat = tr_out.mean(dim=0)

        # Final binary classification
        return self.classifier(feat).squeeze(1)  # → [B]
