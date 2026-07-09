"""Sparse Graph Attention Layer (edge_index based).

Full-graph GAT convolution over the heterogeneous graph. Dense adjacency is
infeasible at ~10k nodes (~400MB float32), so this uses edge_index + scatter.

    h_i' = Σ_{j∈N(i)} α_ij · W h_j
    α_ij = softmax_j( LeakyReLU( a^T [W h_i ‖ W h_j] ) )

Aggregation direction: edge (src -> dst) means src contributes to dst.
For each dst node, attention is softmaxed over its incoming src edges.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_softmax(e: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Segment softmax: for each group id in `index`, softmax over `e`.

    Args:
        e: [E] edge scores
        index: [E] group id (dst node) per edge
        num_nodes: number of groups
    Returns:
        [E] softmaxed scores
    """
    e_max = torch.full((num_nodes,), float('-inf'), device=e.device, dtype=e.dtype)
    e_max.scatter_reduce_(0, index, e, reduce='amax', include_self=True)
    e_exp = (e - e_max[index]).exp()
    e_sum = torch.zeros(num_nodes, device=e.device, dtype=e.dtype)
    e_sum.scatter_add_(0, index, e_exp)
    return e_exp / e_sum[index].clamp(min=1e-12)


class SparseGATLayer(nn.Module):
    """Single-relation sparse GAT layer (multi-head via loop)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        dropout: float = 0.2,
        alpha: float = 0.2,
        concat: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Linear(in_dim, n_heads * out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(alpha)

        out_total = n_heads * out_dim if concat else out_dim
        if in_dim != out_total:
            self.residual_proj = nn.Linear(in_dim, out_total, bias=False)
        else:
            self.residual_proj = None

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_dim]
            edge_index: [2, E] (row0=src, row1=dst)
        Returns:
            [N, out_total]
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        Wh = self.W(x).view(N, self.n_heads, self.out_dim)  # [N, H, d]

        outs = []
        for h in range(self.n_heads):
            wh = Wh[:, h, :]  # [N, d]
            wh_dst = wh[dst]  # [E, d]  (i)
            wh_src = wh[src]  # [E, d]  (j)
            a_in = torch.cat([wh_dst, wh_src], dim=-1)  # [E, 2d]
            e = self.a(a_in).squeeze(-1)  # [E]
            e = self.leaky_relu(e)
            e = scatter_softmax(e, dst, N)  # softmax over j for each i
            if self.training:
                e = F.dropout(e, self.dropout)
            msg = e.unsqueeze(-1) * wh_src  # [E, d]
            out_h = torch.zeros(N, self.out_dim, device=x.device, dtype=x.dtype)
            out_h.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.out_dim), msg)
            outs.append(out_h)

        out = torch.cat(outs, dim=-1) if self.concat else torch.stack(outs).mean(0)
        if self.residual_proj is not None:
            out = out + self.residual_proj(x)
        return out


class SparseGAT(nn.Module):
    """Multi-layer sparse GAT stack for full-graph convolution."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        for i in range(n_layers):
            layers.append(SparseGATLayer(
                in_dim=dims[i], out_dim=dims[i + 1] // n_heads,
                n_heads=n_heads, dropout=dropout, concat=True,
            ))
        self.layers = nn.ModuleList(layers)
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < self.n_layers - 1:
                x = F.elu(x)
        return x
