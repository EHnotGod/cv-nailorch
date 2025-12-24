import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import matplotlib.pyplot as plt
import nailorch
from nailorch import Model
from nailorch import SeqDataLoader
import nailorch.functions as F
import nailorch.layers as L
from nailorch.datasets import Dataset
import pandas as pd
import cupy as cp

np.random.seed(42)

max_epoch = 3000
batch_size = 128
hidden_size = 100
bptt_length = 30



class Pollution(Dataset):
    def __init__(self, file_path, train=True, split=0.7):
        super().__init__()
        self.train = train
        data = pd.read_csv(file_path)[
            ["pollution", "dew", "temp", "press", "wnd_spd", "snow", "rain"]
        ].to_numpy().astype(np.float32)

        label = pd.read_csv(file_path)["pollution"].to_numpy().astype(np.float32).reshape(-1, 1)
        num_split = int(split * data.shape[0])
        if self.train:
            self.data = data[:num_split]
            self.label = label[:num_split]
        else:
            self.data = data[num_split:]
            self.label = label[num_split:]


train_set = Pollution(file_path="../../data/pollution_train.csv")
dataloader = SeqDataLoader(train_set, batch_size=batch_size, gpu=True)
seqlen = len(train_set)


class BetterRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.LSTM(hidden_size, in_size=7)
        self.fc = L.Linear(out_size, in_size=hidden_size)

    def reset_state(self):
        self.rnn.reset_state()

    def __call__(self, x):
        y = self.rnn(x)
        y = self.fc(y)
        return y


model = BetterRNN(hidden_size, 1)
model.to_gpu()
optimizer = nailorch.optimizers.Adam().setup(model)

# ================
# Training
# ================
loss_history = []  # 记录每个 epoch 的平均 loss（基于每个 mini-batch 的标量 loss）
for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    total_loss = 0.0  # 用来累积每个 mini-batch 的标量 loss（用于统计/绘图）
    for x, t in dataloader:
        y = model(x)
        l = F.mean_squared_error(y, t)
        loss += l
        count += 1

        # --- 累积标量用于统计（不影响反传） ---
        total_loss += float(l.data)

        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
    # 记录并打印（保持你原来的 avg_loss 输出但同时记录更可靠的 avg）
    avg_loss_report = float(loss.data) / count
    epoch_avg_loss = total_loss / count if count > 0 else float('nan')
    loss_history.append(epoch_avg_loss)

    print('| epoch %d | loss_report %f | epoch_avg_loss %f' % (epoch + 1, avg_loss_report, epoch_avg_loss))

model.save_weights("LSTM-3000.npz")

model.load_weights("LSTM-3000.npz")
model.to_gpu()

# ===== 测试集可视化 =====
test_set = Pollution(file_path="../../data/pollution_train.csv", train=False)
test_loader = SeqDataLoader(test_set, batch_size=1, gpu=True)

model.reset_state()
preds = []
trues = []

with nailorch.no_grad():
    for x, t in test_loader:
        y = model(x)
        preds.append(float(cp.asnumpy(y.data).ravel()[0]))
        trues.append(float(cp.asnumpy(t).ravel()[0]))


# =========================
# 计算额外评价指标（MSE, RMSE, MAE, R2）
# =========================
def mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

print("\nEvaluation Metrics:")
print("MSE :", mse(trues, preds))
print("RMSE:", rmse(trues, preds))
print("MAE :", mae(trues, preds))
print("R2  :", r2_score(trues, preds))


# ======================
# Plot prediction (原有) 并保存
# ======================
plt.figure(figsize=(10, 4))
plt.plot(trues, label="True Pollution")
plt.plot(preds, label="Predicted Pollution")
plt.xlabel("Time step")
plt.ylabel("Pollution")
plt.legend()
plt.title("Pollution Prediction on Test Set")
plt.savefig("pollution_prediction.png", dpi=300, bbox_inches="tight")
plt.show()

# ======================
# Plot & save training loss 曲线（每 epoch 的平均 mini-batch loss）
# ======================
plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(loss_history) + 1), loss_history, label="Epoch Avg Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE (per-mini-batch average)")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("training_loss.png", dpi=300, bbox_inches="tight")
plt.show()
