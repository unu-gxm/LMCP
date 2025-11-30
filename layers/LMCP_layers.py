import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class GraphTransformerBlock(nn.Module):
    def __init__(self, node_num, d_model, n_heads=4, dropout=0.1):
        super(GraphTransformerBlock, self).__init__()
        self.node_num = node_num
        self.d_model = d_model
        self.n_heads = n_heads
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, adj,x ):
        """
        x: [B, N, D]
        adj: [B, ,H, N, N] - 用于 Attention Masking
        """

        B, N, D = x.shape
        H=self.node_num
        # === Multi-head attention with adjacency mask ===
        qkv = self.qkv_proj(x)  # [B, N, 3D]
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # reshape for multi-head: [B, H, N, D//H]
        def reshape_heads(tensor):
            return tensor.view(B, N, self.n_heads, D // self.n_heads).transpose(1, 2)
        q, k, v = map(reshape_heads, (q, k, v))

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D // self.n_heads) ** 0.5  # [B, H, N, N]
        adj = adj.clone()
        eye = torch.eye(adj.shape[-1], device=adj.device).unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        adj = adj + eye  # 给每个节点加上自连接
        adj = (adj > 0).float()  # 保持adj仍然是0/1矩阵
        mask = (adj == 0)  # [B, H, N, N]

        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [B, H, N, D//H]
        # concat heads and project

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        attn_output = self.dropout(self.out_proj(attn_output))

        # === Add & Norm ===
        x = self.norm1(x + attn_output)
        # === FeedForward + Add & Norm ===
        x = self.norm2(x + self.ff(x))

        return x  # [B, N, D]
# 图过滤
class mask_moe(nn.Module):
    def __init__(self, n_vars, top_p=0.5, num_experts=3, in_dim=96):
        super().__init__()
        self.num_experts = num_experts
        self.n_vars = n_vars
        self.in_dim = in_dim

        self.gate = nn.Linear(self.in_dim, num_experts, bias=False)
        self.noise = nn.Linear(self.in_dim, num_experts, bias=False)
        self.noisy_gating = True
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(2)
        self.top_p = top_p

    def cv_squared(self, x):

        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def cross_entropy(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return -torch.mul(x, torch.log(x + eps)).sum(dim=1).mean()

    def noisy_top_k_gating(self, x, is_training, noise_epsilon=1e-2):
        clean_logits = self.gate(x)

        if self.noisy_gating :
            raw_noise = self.noise(x)
            noise_stddev = ((self.softplus(raw_noise) + noise_epsilon))
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        # Convert logits to probabilities
        logits = self.softmax(logits)
        loss_dynamic = self.cross_entropy(logits)

        sorted_probs, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > self.top_p

        threshold_indices = mask.long().argmax(dim=-1)
        threshold_mask = torch.nn.functional.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()
        mask = mask & ~threshold_mask

        top_p_mask = torch.zeros_like(mask)
        zero_indices = (mask == 0).nonzero(as_tuple=True)
        top_p_mask[zero_indices[0], zero_indices[1], sorted_indices[zero_indices[0], zero_indices[1], zero_indices[2]]] = 1

        sorted_probs = torch.where(mask, 0.0, sorted_probs)
        loss_importance = self.cv_squared(sorted_probs.sum(0))
        lambda_2 = 0.1
        loss = loss_importance + lambda_2 * loss_dynamic

        return top_p_mask, loss

    def forward(self, x, masks=None, is_training=None):

        # x [B, H, L, L]
        B, H, L, _ = x.shape
        device = x.device
        dtype = torch.float32

        mask_base = torch.eye(L, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)

        if self.top_p == 0.0:
            return mask_base, 0.0

        x = x.reshape(B * H, L, L)
        print(is_training,"----------------------------------------------------------")
        gates, loss = self.noisy_top_k_gating(x, is_training)

        gates = gates.reshape(B, H, L, -1).float()
        # [B, H, L, 3]

        if masks is None:
            print("Masks is None!------------------------------------------------------------")
            masks = []
            N = L // self.n_vars
            for k in range(L):
                S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(device)
                T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N)).to(dtype).to(device)
                ST = torch.ones(L).to(dtype).to(device) - S - T
                masks.append(torch.stack([S, T, ST], dim=0))
            # [L, 3, L]
            masks = torch.stack(masks, dim=0)

        mask = torch.einsum('bhli,lid->bhld', gates, masks) + mask_base


        return mask, loss

def mask_topk(x, alpha=0.5, largest=False):
    # B, L = x.shape[0], x.shape[-1]
    # x: [B, H, L, L]
    k = int(alpha * x.shape[-1])
    _, topk_indices = torch.topk(x, k, dim=-1, largest=largest)
    mask = torch.ones_like(x, dtype=torch.float32)
    mask.scatter_(-1, topk_indices, 0)  # 1 is topk
    return mask  # [B, H, L, L]

class GCN(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.n_heads = n_heads
    def forward(self, adj, x):
        # adj [B, H, L, L]
        # adj.fill_(1)#全连接
        B, L, D = x.shape
        x = self.proj(x).view(B, L, self.n_heads, -1)  # [B, L, H, D_]
        adj = F.normalize(adj, p=1, dim=-1)
        relship = adj
        # adjout = adj[0, 0, :, :]
        # print(adjout, '------------------------------=========')

        x = torch.einsum("bhij,bjhd->bihd", adj, x).contiguous()  # [B, L, H, D_]
        x = x.view(B, L, -1)
        return x,relship
# 自适应图
class GraphLearner(nn.Module):
    def __init__(self, dim, n_vars, top_p=0.5, in_dim=96):
        super().__init__()
        self.dim = dim
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.n_vars = n_vars
        self.mask_moe = mask_moe(n_vars, top_p=top_p, in_dim=in_dim)

    def forward(self, x, masks=None, alpha=0.5, is_training=False):
        # x: [B, H, L, D]
        adj = F.gelu(torch.einsum('bhid,bhjd->bhij', self.proj_1(x), self.proj_2(x)))
        # adj.fill_(1)

        adj = adj * mask_topk(adj, alpha)  # KNN

        mask, loss = self.mask_moe(adj, masks, is_training)

        adj = adj * mask

        return adj, loss# [B, H, L, L]

class GraphMASK(nn.Module):
    def __init__(self, dim, n_vars, n_heads=4, scale=None, top_p=0.5, dropout=0., in_dim=96):
        super().__init__()
        self.dim = dim
        self.n_vars=n_vars
        self.n_heads = n_heads
        self.scale = dim ** (-0.5) if scale is None else scale
        self.dropout = nn.Dropout(dropout)
        self.graph_learner = GraphLearner(self.dim // self.n_heads, n_vars, top_p, in_dim=in_dim)
        self.graph_conv = GCN(self.dim, self.n_heads)
    def forward(self, x, masks=None, alpha=0.5, is_training=False):

        # x: [B, L, D]
        B, L, D = x.shape
        adj, loss = self.graph_learner(x.reshape(B, L, self.n_heads, -1).permute(0, 2, 1, 3), masks,
                                       alpha, is_training)  # [B, H, L, L]

        adj = torch.softmax(adj, dim=-1)
        relship=adj
        adj = self.dropout(adj)
        out ,  relshipn= self.graph_conv(adj,x)

        return out, loss ,  relship # [B, L, D]

class GraphBlock(nn.Module):
    def __init__(self, dim, n_vars, d_ff=None, n_heads=4, top_p=0.5, dropout=0., in_dim=96):
        super().__init__()
        self.dim = dim
        self.d_ff = dim * 4 if d_ff is None else d_ff
        self.gnn = GraphMASK(self.dim, n_vars, n_heads, top_p=top_p, dropout=dropout, in_dim=in_dim)
        self.norm1 = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, self.dim),
        )
        self.norm2 = nn.LayerNorm(self.dim)
    def forward(self, x, masks=None, alpha=0.5, is_training=False):
        # x: [B, L, D], time_embed: [B, time_embed_dim]
        out, loss ,  relship= self.gnn(self.norm1(x), masks, alpha, is_training)
        x = x + out
        x = x + self.ffn(self.norm2(x))

        return x, loss,  relship

class LMCP_Backbone(nn.Module):
    def __init__(self, hidden_dim, n_vars, d_ff=None, n_heads=4, n_blocks=3, top_p=0.5, dropout=0., in_dim=96):
        super().__init__()
        self.dim = hidden_dim
        self.d_ff = self.dim * 2 if d_ff is None else d_ff
        self.n_heads = n_heads
        # graph blocks
        self.blocks = nn.ModuleList([
            GraphBlock(self.dim, n_vars, self.d_ff, self.n_heads, top_p, dropout, in_dim)
            for _ in range(n_blocks)
        ])
        self.n_blocks = n_blocks

    def forward(self, x, masks=None, alpha=0.5, is_training=False):

        # x: [B, N, T]

        moe_loss = 0.0
        for block in self.blocks:
            x, loss ,  relship= block(x, masks, alpha, is_training)
            moe_loss += loss

        moe_loss /= self.n_blocks

        return x, moe_loss  ,  relship# [B, N, T]
