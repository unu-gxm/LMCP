import time
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import math
# from layers.Layer import Transformer_Layer
# from utils.Other import SparseDispatcher, FourierLayer, series_decomp_multi, MLP
# --------------------------------
import torch.nn.functional as F
import math
from layers.Embed import PositionalEmbedding
from layers.StandardNorm import Normalize
from layers.LMCP_layers import LMCP_Backbone
from utils.Other import SparseDispatcher, FourierLayer, series_decomp_multi, MLP

class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.patch_proj(x)  # [B, N*L, D]
        if self.pos:
            x += self.pe(x)
        return x
# input_size=96, output_size=96, num_experts=4, device=0, num_nodes=1, d_model=32, dynamic=False,
#                  patch_size=[8, 6, 4, 2], noisy_gating=True, k=4, layer_number=1, residual_connection=1, batch_norm=False,
class AMS(nn.Module):
    def __init__(self,
                dim=128, n_vars=7, d_ff=64, n_heads=4,e_layers=None, top_p=0.5, dropout=0, in_dim=96):
        super(AMS, self).__init__()

        self.seq_len=96
        self.pos=0
        # self.patch_len = 2
        self.pred_len=96
        self.e_layers=e_layers
        self.n_vars=n_vars
        self.d_ff=d_ff
        self.n_heads=n_heads
        self.top_p=top_p
        self.dropout=dropout
        self.in_dim=in_dim
        self.dim=dim
        self.patch_size = [2, 6, 4, 8]
        self.k = 2
        self.patch_pos=nn.ParameterList()
        self.patch_embed=nn.ModuleList()
        self.experts = nn.ModuleList()
        self.num_patches_list=[]
        self.linear1=nn.ModuleList()
        for patch_len in self.patch_size:
            self.stride = patch_len
            self.num_patches=int((self.seq_len - patch_len) / self.stride + 1)
            self.num_patches_list.append(self.num_patches)
            self.p_pos = nn.Parameter(torch.randn(1, 1, self.num_patches, self.dim))  # Patch位置嵌入
            self.patch_pos.append(self.p_pos)
            self.embed = PatchEmbed(self.dim, patch_len, self.stride, self.pos)
            self.patch_embed.append(self.embed)
            self.experts.append( LMCP_Backbone(self.dim, self.n_vars, self.d_ff,
                                            self.n_heads, self.e_layers, self.top_p, self.dropout,
                                            self.seq_len * self.n_vars // patch_len))
            print(self.dim, self.n_vars, self.d_ff,
                                            self.n_heads, self.e_layers, self.top_p, self.dropout,
                                            self.seq_len * self.n_vars // patch_len)
            self.linearcur = nn.Linear(self.num_patches, self.pred_len)
            self.linear1.append(self.linearcur)

        self.head=nn.Linear(self.dim * self.pred_len*self.k, self.pred_len)
        self.channel_pos = nn.Parameter(torch.randn(1, self.n_vars, 1, self.dim))  # 通道位置嵌入

        # self.pos_encoding = nn.Parameter(torch.randn(1, self.n_vars, self.dim))

        self.num_experts = len(self.patch_size)
        self.input_size = self.seq_len

        self.start_linear = nn.Linear(in_features=self.n_vars, out_features=1)
        self.seasonality_model = FourierLayer(pred_len=0, k=3)
        self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])
        self.MLPs = nn.ModuleList()  # -----------------
        self.w_noise = nn.Linear(self.seq_len, self.num_experts)
        self.w_gate = nn.Linear(self.seq_len, self.num_experts)

        self.end_MLP = MLP(input_size=self.seq_len, output_size=self.pred_len)
        self.noisy_gating = True
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)

        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def seasonality_and_trend_decompose(self, x):

        _, trend = self.trend_model(x)
        seasonality, _ = self.seasonality_model(x)


        return x + seasonality + trend#trend是空

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):

        x = self.start_linear(x).squeeze(-1)

        # clean_logits = x @ self.w_gate
        clean_logits = self.w_gate(x)
        if self.noisy_gating and train:
            # raw_noise_stddev = x @ self.w_noise

            raw_noise_stddev = self.w_noise(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))



            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, intx,masks=None, alpha=0.5, is_training=False, loss_coef=1e-3):

        new_x = self.seasonality_and_trend_decompose(intx)
        gates, load = self.noisy_top_k_gating(new_x, is_training)
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(intx)
        expert_weight=dispatcher.expert_to_gates()
        nom=sum(expert_weight)
        expert_out=[]
        out_loss=[]
        B, T, C = intx.shape
        # print(expert_inputs[0].shape,expert_inputs[1].shape,expert_inputs[2].shape,expert_inputs[3].shape,)
        for i in range(self.num_experts):

            inx = expert_inputs[i].permute(0, 2, 1)  # [B, C*T]
            inx = self.patch_embed[i](inx)+self.channel_pos+self.patch_pos[i]
            #
            inx = inx.reshape(-1, self.n_vars * self.num_patches_list[i], self.dim)

            if torch.all(inx == 0):
                continue
            out, moe_loss = self.experts[i](inx, masks[i], alpha, is_training)

            out_loss.append(moe_loss*expert_weight[i]/nom)#*w
            out+=inx
            xcur=out.reshape(-1,self.n_vars,self.num_patches_list[i],self.dim)
            xcur = xcur.permute(0, 1, 3, 2)  # [32,7,128,48]
            xcur = self.linear1[i](xcur).reshape(-1, self.n_vars, self.dim, self.pred_len)  # [32,7,128,96]
            xcur = xcur.permute(0, 1, 3, 2)# [32,7,96,128][B, C, pre, D]
            expert_out.append(xcur)
        # # -----------------------------------------------
        out_loss=sum(out_loss)
        expert_out = dispatcher.combine(expert_out,self.k)
        out=expert_out.reshape(-1, self.n_vars, self.pred_len, self.dim*self.k).flatten(start_dim=-2)
        out = self.head(out)
        # ________________________
        return out,0.0
        #+balance_loss
