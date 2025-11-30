import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PositionalEmbedding
from layers.StandardNorm import Normalize
from layers.LMCP_layers import LMCP_Backbone
from layers.AMS import AMS


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
        # x: [B, N, T]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [B, N*L, P]
        x = self.patch_proj(x) # [B, N*L, D]
        if self.pos:
            x += self.pe(x)
        return x


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):

        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.c_out
        self.dim = configs.d_model
        self.d_ff = configs.d_ff
        self.patch_len = configs.patch_len
        self.stride = self.patch_len
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)
        self.alpha = 0.1 if configs.alpha is None else configs.alpha
        self.top_p = 0.5 if configs.top_p is None else configs.top_p
        print(self.top_p)
        print(self.alpha)

        self.patch_embed = PatchEmbed(self.dim, self.patch_len, self.stride, configs.pos)

        self.backbone = LMCP_Backbone(self.dim, self.n_vars, self.d_ff,
                                            configs.n_heads, configs.e_layers, self.top_p, configs.dropout,
                                            self.seq_len * self.n_vars // self.patch_len)
        self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)

        self.use_RevIN = False
        self.norm = Normalize(configs.enc_in, affine=self.use_RevIN)
 
        configs.d_model = configs.T_model
        configs.d_ff = 128
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.nheads = 16
        self.e_layers_time=1
        self.dorp=0.2
        self.d_model=configs.d_model
        self.PositionalEmbedding = PositionalEmbedding(configs.d_model)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=self.dorp,
                                      output_attention='gelu'), configs.d_model, self.nheads, self.dorp),
                    configs.d_model,
                    configs.d_ff,
                    configs.seq_len,
                    dropout=self.dorp,
                    activation=True
                ) for l in range(self.e_layers_time)  # configs.e_layers_time
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        print(configs.factor, self.dorp,False,
              configs.d_model, self.nheads, self.dorp,configs.d_model,
                    configs.d_ff,configs.seq_len,self.dorp,'gelu',
              self.e_layers_time,configs.d_model)
        self.mapping = nn.Parameter(torch.randn(configs.seq_len, configs.d_model))

        self.dropout = nn.Dropout(self.dorp )

        self.seq_pred = nn.Linear(self.seq_len * 16, configs.pred_len, bias=True)
        self.flatten = nn.Flatten(start_dim=-2)
        self.Linear_concat = nn.Linear(configs.d_model * 1, 16)
        self.FC = nn.Linear(configs.pred_len*2 , configs.pred_len)
        self.trend = nn.Linear(configs.seq_len, configs.pred_len)

    def Embedding(self, x_enc):
        enc_out = x_enc.transpose(2, 1).unsqueeze(-1)

        enc_out = enc_out * self.dropout(self.mapping)          

        enc_out_p = enc_out.reshape(-1, self.seq_len, self.d_model)
        enc_out_p = self.PositionalEmbedding(enc_out_p) + enc_out_p     
        enc_out = enc_out_p.reshape(-1, self.n_vars, self.seq_len, self.d_model) 

        enc_out_in = enc_out.transpose(1, 0)            
        enc_out = enc_out_in

        return enc_out_in, enc_out


    def Channel_independence(self, enc_out_in, trend_init):

        enc_out_in = enc_out_in.reshape(-1, self.seq_len, self.d_model)
        enc_out_in = self.encoder(enc_out_in, attn_mask=None)
        enc_out_in = enc_out_in.reshape(self.n_vars, -1, self.seq_len, self.d_model)  

        enc_out = self.dropout(self.Linear_concat(enc_out_in))
        enc_out = enc_out.permute(1, 0, 2, 3)  
        enc_out_in = self.flatten(enc_out)  

        trend_init = self.dropout( self.trend(trend_init.transpose(2,1)) )

        return enc_out_in, trend_init


    def correlation_matrix(self, x):

        reshaped_data = x.transpose(2,1)[-1]

        mean = reshaped_data.mean(dim=1, keepdim=True)

        centered_matrix = reshaped_data - mean

        cov_matrix = (centered_matrix @ centered_matrix.t()) / (reshaped_data.size(1) - 1)

        std_dev = torch.sqrt(torch.diag(cov_matrix))

        std_dev_broadcast = std_dev.unsqueeze(1) * std_dev.unsqueeze(0)

        std_dev_broadcast[std_dev_broadcast == 0] = 1

        correlation_matrix = cov_matrix / std_dev_broadcast

        correlation_matrix = (correlation_matrix + correlation_matrix.t()) / 2

        correlation_matrix.fill_diagonal_(1)
        normalized_tensor = correlation_matrix

        correlation_matrix_mask = normalized_tensor < self.alpha  # N * N

        return correlation_matrix_mask.cuda()


    def forecast(self, x_enc):

        B, _, N = x_enc.shape  # B L N


        ######### TimeDE  ###
        seasonal_init, trend_init = self.decompsition(x_enc)   # batch input_len variate

        enc_out_in, enc_out = self.Embedding(seasonal_init)         # 7 32 96 128
        enc_out_in, trend_init = self.Channel_independence(enc_out_in, trend_init)
        dec_out_time = self.seq_pred(enc_out_in).permute(0, 2, 1)   # 32 96 7
        dec_out_time = dec_out_time + trend_init.transpose(2,1)

        return dec_out_time

    
    def forward(self, x, masks, is_training=False, target=None):

        B, T, C = x.shape
        x = self.norm(x, 'norm')
        x_enc=x
        inx=x
        inx1 = inx.permute(0, 2, 1).reshape(-1, C * T)  # [B, C*T]
        # inx1 = inx.permute(0, 2, 1).reshape(B*C ,  T)  # [B, C*T]
        # ######## patch gcn ##############
        x1 = self.patch_embed(inx1)  # [B, N, D]  N = [C*T / P]
        x1, moe_loss1 ,  relship= self.backbone(x1, masks[0], self.alpha, is_training)
        x1 = self.head(x1.reshape(-1, self.n_vars, self.num_patches, self.dim).flatten(start_dim=-2))  # [B, C, T]
        x1 = x1.permute(0, 2, 1)
        # ######## TimeAttention ##############

        dec_out_time = self.forecast(inx)#
        ######### concat   ##############
        enc_out_concat = torch.cat((dec_out_time.transpose(2,1), x1.transpose(2,1)), dim=-1)    # 7 32 96 256
        dec_out = self.FC(enc_out_concat)
        dec_out_ = self.norm(dec_out.transpose(2,1), 'denorm')
        ######### inv  ##############
        dec_out_time = self.norm(dec_out_time, 'denorm')
        x1 = self.norm(x1, 'denorm')

        return dec_out_,dec_out_time,x1 ,moe_loss1,  relship

