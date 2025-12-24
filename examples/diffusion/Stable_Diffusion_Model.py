import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import math
import numpy as np
import matplotlib.pyplot as plt
from nailorch.optimizers import *
from nailorch import *
import nailorch.functions as F
import nailorch.layers as L
import nailorch.models as M
from nailorch import cuda, Variable

class SinusoidalPositionEmbeddings(L.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        xp = cuda.get_array_module(time)
        half_dim = self.dim // 2
        freq = math.log(10000) / (half_dim - 1)
        freqs = xp.exp(xp.arange(half_dim) * -freq)
        # (batch_size, 1) * (half_dim, ) -> (batch_size, half_dim)
        scaled = time.reshape(-1, 1) * freqs.reshape(1, -1)
        emb = F.concat((F.sin(scaled), F.cos(scaled)), axis=1)
        return emb

# -----------------------------
# TransformerBlock (cross-attention with cls embedding)
# -----------------------------
class TransformerBlock(L.Layer):
    def __init__(self, channel, qsize, vsize, n_embd, fsize=None):
        super().__init__()
        self.w_q = L.Linear(qsize, in_size=channel)   # project per-position features -> Q
        self.w_k = L.Linear(qsize, in_size=n_embd)    # project cls_emb -> K
        self.w_v = L.Linear(vsize, in_size=n_embd)    # project cls_emb -> V
        self.z_linear = L.Linear(channel, in_size=vsize)  # project V-space -> channel
        self.ln1 = L.LayerNorm()
        self.ln2 = L.LayerNorm()
        if fsize is None:
            fsize = channel * 4
        self.fc1 = L.Linear(fsize, in_size=channel)
        self.fc2 = L.Linear(channel, in_size=fsize)

    def forward(self, x, cls_emb):
        xp = cuda.get_array_module(x)
        x_t = x.transpose(0, 2, 3, 1)
        # Q: (batch, h, w, qsize)
        Q = self.w_q(x_t)
        b, h, w, qdim = Q.shape
        Q = Q.reshape(b, h * w, qdim)  # (batch, h * w, qsize)
        # K: (batch, qsize) -> (batch, 1, qsize)
        K = self.w_k(cls_emb)
        K = K.reshape(b, 1, K.shape[1])
        # V: (batch, vsize) -> (batch, 1, vsize)
        V = self.w_v(cls_emb)
        V = V.reshape(b, 1, V.shape[1])
        # Q @ K^T
        K_t = K.transpose(0, 2, 1)  # (batch, qsize, 1)
        attn_scores = F.matmul(Q, K_t) / math.sqrt(float(qdim))  # (batch, h * w, 1)
        attn = F.softmax(attn_scores, axis=-1)

        Z = F.matmul(attn, V)  # (batch, h * w, vsize)
        Z = self.z_linear(Z)  # (batch, h * w, channel)
        Z = Z.reshape(b, h, w, Z.shape[2])  # (batch, h, w, channel)
        Z = self.ln1(Z + x_t)
        out = self.fc1(Z)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.ln2(Z + out)
        # transpose back to (batch, channel, h, w)
        return out.transpose(0, 3, 1, 2)


class Block(L.Layer):
    """通用的卷积块: Conv -> BN -> SiLU"""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = L.Linear(out_ch, in_size=time_emb_dim)
        if up:
            self.conv = L.Deconv2d(out_ch, kernel_size=4, stride=2, pad=1, in_channels=in_ch)
            self.transform = L.Conv2d(out_ch, kernel_size=3, stride=1, pad=1, in_channels=out_ch)
        else:
            self.conv = L.Conv2d(out_ch, kernel_size=3, stride=1, pad=1, in_channels=in_ch)
            self.transform = L.Conv2d(out_ch, kernel_size=3, stride=1, pad=1, in_channels=out_ch)

        self.bn = L.BatchNorm()
        self.up = up

    def forward(self, x, t):
        # 第一次卷积
        h = self.conv(x)
        h = self.bn(h)
        h = F.silu(h)
        time_emb = self.time_mlp(F.silu(t))
        h = h + time_emb.reshape(time_emb.shape[0], time_emb.shape[1], 1, 1)
        # 第二次卷积
        h = self.transform(h)
        return h


class StableUNet(L.Layer):
    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = [32, 64, 128]
        up_channels = [128, 64, 32]
        out_dim = 1
        time_emb_dim = 32

        self.n_vocab = 10
        n_embd = time_emb_dim

        # time embedding
        self.time_mlp = M.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            L.Linear(time_emb_dim, in_size=time_emb_dim)
        )
        self.wte = L.EmbedID(self.n_vocab, n_embd)

        # -------- Down --------
        self.down1 = Block(image_channels, down_channels[0], time_emb_dim)
        self.attn1 = TransformerBlock(down_channels[0], 32, 32, n_embd)
        self.pool1 = lambda x: F.pooling(x, 2, 2)

        self.down2 = Block(down_channels[0], down_channels[1], time_emb_dim)
        self.attn2 = TransformerBlock(down_channels[1], 64, 64, n_embd)
        self.pool2 = lambda x: F.pooling(x, 2, 2)

        self.down3 = Block(down_channels[1], down_channels[2], time_emb_dim)
        self.attn3 = TransformerBlock(down_channels[2], 64, 64, n_embd)

        # -------- Bottleneck --------
        self.mid_attn = TransformerBlock(down_channels[2], 64, 64, n_embd)

        # -------- Up --------
        self.up0 = Block(down_channels[2], up_channels[0], time_emb_dim, up=True)

        self.up1 = Block(up_channels[0] + down_channels[1], up_channels[1], time_emb_dim, up=True)
        self.up_attn1 = TransformerBlock(up_channels[1], 64, 64, n_embd)

        self.up2 = Block(up_channels[1] + down_channels[0], up_channels[2], time_emb_dim)
        self.up_attn2 = TransformerBlock(up_channels[2], 32, 32, n_embd)

        self.output = L.Conv2d(out_dim, kernel_size=1, in_channels=up_channels[2])

    def forward(self, x, t, cls):
        t_emb = self.time_mlp.l0(t)
        t_emb = F.silu(self.time_mlp.l1(t_emb))
        cls_emb = self.wte(cls)

        # ---- Down ----
        x1 = self.down1(x, t_emb)
        x1 = self.attn1(x1, cls_emb)
        x2 = self.pool1(x1)

        x2 = self.down2(x2, t_emb)
        x2 = self.attn2(x2, cls_emb)
        x3 = self.pool2(x2)

        x3 = self.down3(x3, t_emb)
        x3 = self.attn3(x3, cls_emb)

        # ---- Bottleneck ----
        x3 = self.mid_attn(x3, cls_emb)

        # ---- Up ----
        x = self.up0(x3, t_emb)
        x = F.concat((x, x2), axis=1)

        x = self.up1(x, t_emb)
        x = self.up_attn1(x, cls_emb)
        x = F.concat((x, x1), axis=1)

        x = self.up2(x, t_emb)
        x = self.up_attn2(x, cls_emb)

        return self.output(x)


# =============================================================================
# 3. 扩散工具 (Diffusion Logic)
# =============================================================================

class Diffusion:
    def __init__(self, T=300, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = np.linspace(beta_start, beta_end, T)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.append(1., self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - self.alphas_cumprod)

    def get_val(self, val_arr, t, x_shape):
        batch_size = t.shape[0]
        xp = cuda.get_array_module(t)
        if isinstance(t, Variable):
            idx = t.data
        else:
            idx = t
        if xp != np:
            idx = cuda.as_numpy(idx)
        out = val_arr[idx]
        if xp != np:
            out = cuda.as_cupy(out)

        return out.reshape(batch_size, 1, 1, 1)

    def q_sample(self, x_0, t, noise=None):

        xp = cuda.get_array_module(x_0)
        noise = xp.random.randn(*x_0.shape).astype(x_0.dtype)

        sqrt_alphas_cumprod_t = self.get_val(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_val(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        x_t_data = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t_data, noise
