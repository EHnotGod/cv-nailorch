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
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = xp.exp(xp.arange(half_dim) * -embeddings)

        # (batch_size, 1) * (half_dim, ) -> (batch_size, half_dim)
        embeddings = time.reshape(-1, 1) * embeddings.reshape(1, -1)
        embeddings = F.concat((F.sin(embeddings), F.cos(embeddings)), axis=1)
        return embeddings


class Block(Layer):
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


class SimpleUNet(L.Layer):
    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = [32, 64, 128]
        up_channels = [128, 64, 32]
        out_dim = 1
        time_emb_dim = 32

        # 时间编码
        self.time_mlp = M.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            L.Linear(time_emb_dim, in_size=time_emb_dim)
        )

        # 下采样路径
        self.down1 = Block(image_channels, down_channels[0], time_emb_dim)
        self.pool1 = lambda x: F.pooling(x, 2, 2)  # Downsample

        self.down2 = Block(down_channels[0], down_channels[1], time_emb_dim)
        self.pool2 = lambda x: F.pooling(x, 2, 2)

        self.down3 = Block(down_channels[1], down_channels[2], time_emb_dim)

        # 上采样路径
        self.up0 = Block(down_channels[2], up_channels[0], time_emb_dim, up=True)  # 128 -> 128
        self.up1 = Block(up_channels[0] + down_channels[1], up_channels[1], time_emb_dim, up=True)  # 128+64 -> 64
        self.up2 = Block(up_channels[1] + down_channels[0], up_channels[2], time_emb_dim, up=False)  # 64+32 -> 32

        self.output = L.Conv2d(out_dim, kernel_size=1, in_channels=up_channels[2])

    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_mlp.l0(t)
        t_emb = F.silu(self.time_mlp.l1(t_emb))

        # Down
        x1 = self.down1(x, t_emb)  # 28
        x2 = self.pool1(x1)  # 14
        x2 = self.down2(x2, t_emb)  # 14
        x3 = self.pool2(x2)  # 7
        x3 = self.down3(x3, t_emb)  # 7

        # Up
        x_up0 = self.up0(x3, t_emb)  # 14
        x_up0 = F.concat((x_up0, x2), axis=1)

        x_up1 = self.up1(x_up0, t_emb)  # 28
        x_up1 = F.concat((x_up1, x1), axis=1)

        x_up2 = self.up2(x_up1, t_emb)  # 28 (No upsample)

        out = self.output(x_up2)
        return out


# =============================================================================
# 3. 扩散工具 (Diffusion Logic)
# =============================================================================

class Diffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
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
        # t 是 Variable 或 array
        if isinstance(t, Variable):
            idx = t.data
        else:
            idx = t

        # 修复: 使用 as_numpy (转CPU) 和 as_cupy (转GPU)
        if xp != np:
            idx = cuda.as_numpy(idx)

        out = val_arr[idx]

        # 获取值后再转回 GPU
        if xp != np:
            out = cuda.as_cupy(out)

        return out.reshape(batch_size, 1, 1, 1)

    def q_sample(self, x_0, t, noise=None):
        # 统一获取 array module
        xp = cuda.get_array_module(x_0)
        noise = xp.random.randn(*x_0.shape).astype(x_0.dtype)

        sqrt_alphas_cumprod_t = self.get_val(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_val(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        x_t_data = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t_data, noise