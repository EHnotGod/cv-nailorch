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
np.random.seed(42)

max_epoch = 100
batch_size = 30
hidden_size = 100
bptt_length = 30

train_set = nailorch.datasets.SinCurve(train=True)
dataloader = SeqDataLoader(train_set, batch_size=batch_size)
seqlen = len(train_set)


class BetterRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def __call__(self, x):
        y = self.rnn(x)
        y = self.fc(y)
        return y


model = BetterRNN(hidden_size, 1)
optimizer = nailorch.optimizers.Adam().setup(model)

# ======================
# Training
# ======================
for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in dataloader:
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1

        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

    avg_loss = float(loss.data) / count
    print('| epoch %d | loss %f' % (epoch + 1, avg_loss))


# ======================
# Evaluation + Metrics
# ======================
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()

pred_list = []
true_list = []

with nailorch.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data))
        true_list.append(float(x))

y_true = np.array(true_list)
y_pred = np.array(pred_list)

print("\nEvaluation Metrics:")
print("MSE :", mse(y_true, y_pred))
print("RMSE:", rmse(y_true, y_pred))
print("MAE :", mae(y_true, y_pred))
print("R2  :", r2_score(y_true, y_pred))


# ======================
# Plot
# ======================
plt.plot(np.arange(len(xs)), y_true, label='y=cos(x)')
plt.plot(np.arange(len(xs)), y_pred, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f'R2 = {r2_score(y_true, y_pred):.4f}')
plt.show()