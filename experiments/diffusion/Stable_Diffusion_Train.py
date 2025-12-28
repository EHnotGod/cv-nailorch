import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import math
import numpy as np
import matplotlib.pyplot as plt
import nailorch
import nailorch.functions as F
from nailorch import cuda, Variable
from nailorch import optimizers
from nailorch import datasets, dataloaders
from Stable_Diffusion_Model import StableUNet, Diffusion

# ============================================================
# Config
# ============================================================
gpu = cuda.gpu_enable
epochs = 150
batch_size = 256
lr = 1e-4
img_size = 28
sample_batch = 16
T = 300


# ============================================================
# Dataset & DataLoader
# ============================================================
def transform(data):
    return (data - 127.5) / 127.5

train_set = datasets.MNIST(
    train=True,
    transform=transform
)

train_loader = dataloaders.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
)
steps_per_epoch = math.ceil(len(train_set) / batch_size)

if gpu:
    train_loader.to_gpu()
    print("Using GPU")
else:
    print("Using CPU")


# ============================================================
# Model & Optimizer
# ============================================================
model = StableUNet()
optimizer = optimizers.Adam(lr)
optimizer.setup(model)

diffusion = Diffusion(T=T)

if gpu:
    model.to_gpu()


# ============================================================
# Training
# ============================================================
print("Start Training...")
start_time = time.time()

for epoch in range(epochs):
    total_loss = 0.0
    total_samples = 0

    for i, (x, y) in enumerate(train_loader):
        N = x.shape[0]
        xp = cuda.get_array_module(x.data if isinstance(x, Variable) else x)
        t = xp.random.randint(0, diffusion.T, (N,), dtype=np.int64)
        t = Variable(t)
        x_t, noise = diffusion.q_sample(x, t)
        noise_pred = model(x_t, t, y)
        loss = F.mean_squared_error(noise_pred, noise)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        total_loss += float(loss.data) * N
        total_samples += N

        if (i + 1) % 1 == 0:
            elapsed = time.time() - start_time
            iters = epoch * steps_per_epoch + i + 1
            speed = iters / elapsed if elapsed > 0 else 0.0
            print(
                f"Epoch {epoch+1} | Iter {i+1} | "
                f"Loss {float(loss.data):.4f} | "
                f"Speed {speed:.2f} it/s"
            )

    avg_loss = total_loss / total_samples
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
model.save_weights("stable-ddpm3.npz")

# ============================================================
# Sampling (no visualization)
# ============================================================
print("Loading...")
model.load_weights("stable-ddpm3.npz")
if gpu:
    model.to_gpu()
print("Sampling...")
xp = cuda.cupy if gpu else np

x = Variable(
    xp.random.randn(sample_batch, 1, img_size, img_size).astype(np.float32)
)

cls_sample = xp.array(
    [0, 0, 0, 0, 0, 0, 0,   # 7 个 0
    1, 2, 3, 4,
    5, 6, 7, 8, 9],      # 1–9 各一个
    dtype=xp.int64
)


with nailorch.no_grad():
    for i in reversed(range(diffusion.T)):
        t = xp.full((sample_batch,), i, dtype=np.int64)
        t = Variable(t)

        noise_pred = model(x, t, cls_sample)  # pass cls during sampling

        alpha = diffusion.alphas[i]
        alpha_bar = diffusion.alphas_cumprod[i]
        beta = diffusion.betas[i]

        if i > 0:
            z = xp.random.randn(*x.shape).astype(np.float32)
        else:
            z = 0

        # compute x_{t-1} (use .data for Variables)
        x_data = (
            (1 / xp.sqrt(alpha))
            * (x.data - (1 - alpha) / xp.sqrt(1 - alpha_bar) * noise_pred.data)
            + xp.sqrt(beta) * z
        )

        x = Variable(x_data)

print("Sampling finished.")
# 如果在 GPU，转回 CPU
if gpu:
    x.to_cpu()

# 反归一化 [-1, 1] -> [0, 1]
samples = (x.data + 1.0) * 0.5
samples = np.clip(samples, 0.0, 1.0)

# 画 4x4 网格
n = int(np.sqrt(samples.shape[0]))
fig, axes = plt.subplots(n, n, figsize=(6, 6))

idx = 0
for i in range(n):
    for j in range(n):
        axes[i, j].imshow(samples[idx, 0], cmap="gray")
        axes[i, j].axis("off")
        idx += 1

plt.tight_layout()
plt.savefig("ddpm_samples3.png", dpi=200)
plt.close()

print("Saved samples to ddpm_samples3.png")
