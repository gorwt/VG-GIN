import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.nn import MultiheadAttention
from torch_geometric.nn import MessagePassing, GINConv, GATConv

import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GatedResidualLayer(nn.Module):
    def __init__(self, hidden_dim, residual_dim=None):
        super().__init__()

        self.residual_proj = nn.Identity() if (residual_dim is None or residual_dim == hidden_dim) else nn.Linear(residual_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

        # Record gate values for loss calculation
        self.last_gate_values = None

    def forward(self, x, residual):
        residual = self.residual_proj(residual)
        # Calculate gate values
        gate_input = torch.cat([x, residual], dim=-1)
        gate = self.gate(gate_input)
        self.last_gate_values = gate
        transformed = self.transform(x)
        return gate * transformed + (1 - gate) * residual

    def compute_consistency_loss(self):
        if self.last_gate_values is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        return torch.mean(-(self.last_gate_values * torch.log(self.last_gate_values + 1e-10) +
                            (1 - self.last_gate_values) * torch.log(1 - self.last_gate_values + 1e-10)))


class AttentionAggregator(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attention = GATConv(
            hidden_dim, hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            add_self_loops=False
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention_weights = None  # Store attention matrices

    def forward(self, x, edge_index):
        # Get attention outputs and weights
        attn_output, (_, attn_weights) = self.attention(
            x, edge_index,
            return_attention_weights=True
        )

        self.attention_weights = attn_weights
        return self.norm(x + attn_output)

    def compute_sparsity_loss(self):
        if self.attention_weights is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # Calculate entropy
        probs = self.attention_weights.mean(dim=1)  # [num_edges]
        entropy = -torch.mean(probs * torch.log(probs + 1e-10))
        return entropy

# Encoder
class GINGRNATTEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, num_heads=4):
        super().__init__()

        def make_gin_mlp(dim_in, dim_out):
            return nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.BatchNorm1d(dim_out),
                nn.ReLU(),
                nn.Linear(dim_out, dim_out),
                nn.BatchNorm1d(dim_out),
                nn.ReLU()
            )

        self.gin_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.grn_layers = nn.ModuleList()

        dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        for i in range(num_layers):
            # GIN Layers
            mlp = make_gin_mlp(dims[i], hidden_dim)
            self.gin_layers.append(GINConv(mlp, eps=1e-5, train_eps=True))

            # Attention Layers
            self.attention_layers.append(AttentionAggregator(hidden_dim, num_heads))
            self.grn_layers.append(GatedResidualLayer(hidden_dim))

        self.mean_lin = nn.Linear(hidden_dim, latent_dim)
        self.logstd_lin = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )
        self.logstd_scale = 5

    def forward(self, x, edge_index):
        h = x
        for i, (gin_layer, attn_layer, grn_layer) in enumerate(zip(
                self.gin_layers, self.attention_layers, self.grn_layers
        )):
            h_prev = gin_layer(h, edge_index)
            h_prev = F.relu(h_prev)

            h_attn = attn_layer(h_prev, edge_index)
            h = grn_layer(h_attn, h_prev)

        mean = self.mean_lin(h)
        logstd = self.logstd_lin(h) * self.logstd_scale
        z = self.reparameterize(mean, logstd)
        return mean, logstd, z

    def reparameterize(self, mean, logstd):
        noise = torch.randn_like(mean)
        # std = torch.exp(logstd.clamp(max=10))
        return mean + noise * torch.exp(logstd)

# Decoder
class DotProductDecoder(nn.Module):
    def __init__(self):
        super(DotProductDecoder, self).__init__()

    def forward(self, z):
        adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_pred

# VG_GIN model
class GIN_VGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super().__init__()
        self.encoder = GINGRNATTEncoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = DotProductDecoder()

        # Loss weighting parameters
        self.gate_loss_weight = 0.05
        self.attn_loss_weight = 0.05

    def forward(self, x, edge_index):
        mean, logstd, z = self.encoder(x, edge_index)
        adj_pred = self.decoder(z)
        return adj_pred, mean, logstd, z

    def compute_auxiliary_losses(self):
        # Calculate auxiliary losses
        gate_loss, attn_loss = 0.0, 0.0

        for layer in self.encoder.grn_layers:
            gate_loss += layer.compute_consistency_loss()

        for layer in self.encoder.attention_layers:
            attn_loss += layer.compute_sparsity_loss()

        total_aux_loss = (
                self.gate_loss_weight * gate_loss +
                self.attn_loss_weight * attn_loss
        )
        return total_aux_loss, {"gate_loss": gate_loss, "attn_loss": attn_loss}