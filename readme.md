# Python 手搓经典计算机视觉与深度学习系统

<p align="center">
  <em>I'm the nailong, I'm the real nailorch!</em>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

本项目是一个**深度学习学习与实践仓库**，包含从传统机器学习到现代生成模型的多种算法实现。所有示例均基于 NumPy/CuPy 从底层实现，旨在帮助理解算法原理。

> 📌 本项目中的 `nailorch` 模块参考了 [DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3)（《深度学习入门3》）的设计与实现。

---

## 📚 算法示例

| 算法 | 描述 | 代码 | 原理文档 |
|------|------|------|----------|
| **KNN** | K-近邻分类算法 | [examples/knn/](examples/knn/) | [KNN.md](docs/knn/KNN.md) |
| **Softmax + MLP** | Softmax 回归与多层感知机 | [examples/softmax/](examples/softmax/) | [softmax.md](docs/softmax/softmax.md) |
| **TLNN** | 两层全连接神经网络 | [examples/tlnn/](examples/tlnn/) | [TLNN.md](docs/tlnn/TLNN.md) |
| **CNN** | 卷积神经网络 | [examples/cnn/](examples/cnn/) | [CNN.md](docs/cnn/CNN.md) |
| **RNN / LSTM** | 循环神经网络与长短期记忆 | [examples/rnn/](examples/rnn/) | [RNN.md](docs/rnn/RNN.md) |
| **DDPM** | 去噪扩散概率模型 | [examples/diffusion/](examples/diffusion/) | - |
| **Pico-GPT** | 轻量级 GPT 语言模型 | [examples/pico-gpt/](examples/pico-gpt/) | - |

更多更详细的原理讲解请关注本项目的飞书文档，以及本目录下的 PDF 教程。

📎 **飞书文档**：[点击访问](https://scnd2n1l49md.feishu.cn/docx/JqkOdUvxfoHxU7x391kcPJBSnpf?from=from_copylink)

---

## 🚀 快速开始

### 环境要求

- Python ≥ 3.8
- NumPy
- Matplotlib
- CuPy（可选，建议，GPU 加速）

### 运行示例

```bash
# 以 CNN 为例
cd examples/cnn
pip install -r requirements.txt
python train.py
```

---

## 📁 项目结构

```
├── examples/                  # 算法示例代码
│   ├── knn/                   # K-近邻
│   ├── softmax/               # Softmax + MLP
│   ├── tlnn/                  # 两层神经网络
│   ├── cnn/                   # 卷积神经网络
│   ├── rnn/                   # RNN / LSTM
│   ├── diffusion/             # 扩散模型
│   └── pico-gpt/              # GPT 语言模型
├── docs/                      # 算法原理文档
├── data/                      # 数据文件
├── nailorch/                  # 底层工具库 (基于 DeZero)
└── llm_logs/                  # 学习日志
```

---

## 📖 文档与报告

- [CNN 实验报告](docs/cnn/cnn_report_zh.md)
- [Softmax 实验报告](docs/softmax/softmax_report_zh.md)
- [TLNN 实验报告](docs/tlnn/tlnn_report_zh.md)

其余实验报告融进了原理讲解内。

---

## 📜 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

## 🙏 致谢

- [DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3) - 本项目的 `nailorch` 模块基于此实现