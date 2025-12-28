# RNN/LSTM

RNN/LSTM 算法的最小可运行示例，分别运行在Cos数据集以及Pollution数据集（源自kaggle）上

- **目标**：提供一个**可直接运行**的传统机器学习 / 浅层神经网络基线  
- **定位**：项目示例（非完整教程）

📄 **结果分析**：  
实验结果与分析请见 [`docs/rnn/RNN_LSTM-report-zh.pdf`](../docs/rnn/RNN_LSTM-report-zh.pdf)

📘 **原理说明**：  
RNN/LSTM 的原理与推导请见：[`docs/rnn/RNN.md`](../docs/rnn/RNN.md)

---

## 目录结构

```text
cnn/
├── LSTM.py                 # LSTM模型训练、预测、绘图主程序
├── LSTM-3000.npz           # 已训练的模型数据（训练了3000个epoch，可以通过model.load_weights()导入）
├── rnn.py                  # rnn的demo
├── readme.md               # 本说明文档
└── requirements.txt        # 依赖列表
```
环境要求
`
Python ≥ 3.8
`

安装依赖：
```
pip install -r requirements.txt
```

在 cnn/ 目录下执行：
```
python LSTM.py
```

程序将加载 Pollution 数据并开始训练。