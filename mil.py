# mil.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MILHead2D(nn.Module):
    def __init__(self, d_model: int = 128, use_cls_token: bool = True, dropout: float = 0.3):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.d_model = d_model

        self.attn = nn.Linear(d_model, 1, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.mlp1 = nn.LazyLinear(64)
        self.mlp2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)

    def _to_BCTD(self, H: torch.Tensor) -> torch.Tensor:
        if H.dim() == 3:  # (B,T,D) -> (B,1,T,D)
            H = H.unsqueeze(1)
        elif H.dim() != 4:
            raise ValueError(f"Expected (B,T,D) or (B,C,T,D), got {tuple(H.shape)}")
        return H

    def forward(self, token_states: torch.Tensor) -> torch.Tensor:
        H = self._to_BCTD(token_states)
        B, C, T, D = H.shape

        start = 1 if (self.use_cls_token and T > 1) else 0
        H = H[:, :, start:, :]           

        H_ = H.reshape(B*C, T - start, D)        
        a = torch.softmax(self.attn(H_), dim=1)   
        z = (H_ * a).sum(dim=1)                  
        Z = z.view(B, C, D)                      

        Z = self.norm(Z)
        Z = self.dropout(Z)
        Z_flat = Z.reshape(B, -1)                 
        h = F.relu(self.mlp1(Z_flat))
        h = self.dropout(h)
        s = self.mlp2(h).squeeze(-1)             
        return s
