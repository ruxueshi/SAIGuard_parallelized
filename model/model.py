import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    if adj.dim() == 2:
        adj = adj.unsqueeze(0)
        squeeze_back = True
    elif adj.dim() == 3:
        squeeze_back = False
    else:
        raise ValueError("adj is [N,N] or [B,N,N]")

    b, n, _ = adj.shape
    eye = torch.eye(n, device=adj.device).unsqueeze(0).expand(b, -1, -1)
    a_hat = adj + eye

    deg = a_hat.sum(dim=-1).clamp(min=1e-8)
    deg_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt = torch.diag_embed(deg_inv_sqrt)

    a_norm = d_inv_sqrt @ a_hat @ d_inv_sqrt
    return a_norm.squeeze(0) if squeeze_back else a_norm


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F] / adj_norm: [B, N, N]
        return adj_norm @ self.linear(x)


class G_ADM(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads=1, concat=True, edge_dim=None, num_layers=2,
            dropout=0.2, residual=False, aggr_type="mean", add_time_emb=False
    ):
        super().__init__()
        latent_dim = 128
        self.gcn1 = GraphConv(in_channels, hidden_channels)
        self.gcn2 = GraphConv(hidden_channels, latent_dim)
        self.dropout = nn.Dropout(dropout)

        self.attr_decoder = nn.Sequential(
            # nn.Linear(1024, hidden_channels),
            nn.Linear(latent_dim, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, in_channels),
        )

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj_norm = normalize_adj(adj)

        h = self.gcn1(x, adj_norm)
        h = F.gelu(h)
        h = self.dropout(h)

        z = self.gcn2(h, adj_norm)
        z = F.gelu(z)
        return z  # [B, N, D]

    def decode_attr(self, z: torch.Tensor) -> torch.Tensor:
        return self.attr_decoder(z)  # [B, N, F]

    def decode_struct(self, z: torch.Tensor) -> torch.Tensor:
        logits = z @ z.transpose(1, 2)   # [B, N, N]
        return torch.sigmoid(logits)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor):
        """
        x:   [B, N, F]
        adj: [B, N, N]
        """
        graph_state = x.mean(dim=1)
        z = self.encode(x, adj_matrix)

        # z=x
        x_hat = self.decode_attr(z)
        a_hat = self.decode_struct(z)

        graph_state_hat = x_hat.mean(dim=1)

        return x_hat, a_hat, z, graph_state, graph_state_hat


def reconstruction_loss(
    x: torch.Tensor,
    adj: torch.Tensor,
    x_hat: torch.Tensor,
    a_hat: torch.Tensor,
    g_s: torch.Tensor,
    g_s_hat: torch.Tensor,
    attr_weight: float = 0.5,
    graph_weight: float = 0.4,
    struct_weight: float = 0.1,
):

    attr_err = ((x_hat - x) ** 2).mean(dim=-1)   # [B, N]
    graph_err = ((g_s_hat - g_s) ** 2).mean(dim=-1)   # [B, N]

    a_hat = a_hat.clamp(min=1e-6, max=1 - 1e-6)
    bce = F.binary_cross_entropy(a_hat, adj.float(), reduction="none")  # [B, N, N]
    struct_err = bce.mean(dim=-1)  # [B, N]

    node_score = attr_weight * attr_err + struct_weight * struct_err + graph_weight * graph_err.unsqueeze(1)
    total_loss = node_score.mean()

    return total_loss, node_score, attr_err, struct_err, graph_err


def reconstruction_loss2(
    x: torch.Tensor,
    adj: torch.Tensor,
    x_hat: torch.Tensor,
    a_hat: torch.Tensor,
    g_s: torch.Tensor,
    g_s_hat: torch.Tensor,
    attr_weight: float = 0.5,
    graph_weight: float = 0.5,
):

    attr_err = ((x_hat - x) ** 2).mean(dim=-1)   # [B, N]
    graph_err = ((g_s_hat - g_s) ** 2).mean(dim=-1)   # [B, N]

    node_score = attr_err
    total_loss = attr_weight * attr_err + graph_weight * graph_err.unsqueeze(1)
    # total_loss = attr_err
    total_loss = total_loss.mean()

    return total_loss, node_score, attr_err, None, graph_err