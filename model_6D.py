import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class EnvJointEncoder6D(nn.Module):
    """
    Kinematics-informed + environment-informed joint encoder
    --------------------------------------------------------
    Input:
        env_feat3d: (B, C_env, D, H, W)
        joint_angles: (B, N, 6)
    Output:
        joint_env_features: (B, N, d_model)
    """
    def __init__(self, d_model, joint_dim=6, env_dim=32):
        super().__init__()
        self.d_model = d_model
        self.joint_dim = joint_dim

        self.freq_bands = nn.Parameter(
            torch.exp(torch.linspace(0, 3, d_model // 12)),
            requires_grad=False
        )

        self.link_embed = nn.Parameter(torch.randn(joint_dim, d_model // joint_dim))

        self.env_pool = nn.AdaptiveAvgPool3d(1)      # (B,C,D,H,W)->(B,C,1,1,1)
        self.env_proj = nn.Linear(env_dim, d_model)  # C_env -> d_model

        self.fuse = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, env_feat3d, joint_angles):
        """
        joint_angles: (B, N, D)
        env_feat3d:   (B, C_env, D, H, W)
        """
        B, N, _ = joint_angles.shape
        _, C_env, D, H, W = env_feat3d.shape 

        # joint_angles: (B,N,D)
        angles = joint_angles.unsqueeze(-1) * self.freq_bands  # (B,N,D,F)
        sin_part = torch.sin(angles)
        cos_part = torch.cos(angles)
        periodic_enc = torch.cat([sin_part, cos_part], dim=-1)
        periodic_enc = periodic_enc.reshape(B, N, -1)          # → (B,N,d_model)

        topo_enc = self.link_embed.unsqueeze(0).unsqueeze(0)   # (1,1,D,d_model/D)
        topo_enc = topo_enc.expand(B, N, -1, -1)               # (B,N,D,d_model/D)
        topo_enc = topo_enc.reshape(B, N, -1)                  # → (B,N,d_model)

        pooled = self.env_pool(env_feat3d).view(B, C_env)      # (B,C_env)
        env_global = self.env_proj(pooled)                     # (B,d_model)
        env_global = env_global.unsqueeze(1).expand(B, N, self.d_model)  # (B,N,d_model)

        fused = periodic_enc + topo_enc + env_global           # (B,N,d_model)
        fused = self.fuse(fused)                               # (B,N,d_model)

        return fused

class TransformerNodeSampler6D_PI_KGAN(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, depth, height, width,input_dim=6, dim_feedforward=512):
        super().__init__()

        self.d_model = d_model

        self.env_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
        )

        self.env_joint_encoder = EnvJointEncoder6D(d_model,joint_dim=input_dim, env_dim=32)

        self.input_embedding = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.output_layer = nn.Linear(d_model, input_dim)

    def forward(self, node_positions, env_map):
        """
        node_positions: (B, N, D)
        env_map:        (B, D, H, W)  3D env map
        """
        B, N, _ = node_positions.shape

        env_map = env_map.unsqueeze(1)           # (B,1,D,H,W)
        env_feat3d = self.env_conv(env_map)      # (B,32,D,H,W)

        joint_env_features = self.env_joint_encoder(env_feat3d, node_positions)  # (B,N,d_model)

        joint_embed = self.input_embedding(node_positions)                        # (B,N,d_model)

        fused = joint_embed + joint_env_features                                  # (B,N,d_model)

        trans_out = self.transformer_encoder(fused)                               # (B,N,d_model)

        next_q = self.output_layer(trans_out[:, -1])                              # (B,6)

        return next_q


class LossFunc(nn.Module):
    def __init__(self, delta):
        super(LossFunc, self).__init__()
        self.delta = delta
        
    def forward(self, out, label, target, input_length):
        mse_loss = nn.MSELoss()
        loss_1 = mse_loss(out, label)
        loss_2 = self.delta * mse_loss(out, target) * max(input_length)
        loss = loss_1 + loss_2
        loss = torch.autograd.Variable(loss, requires_grad=True)
        return loss