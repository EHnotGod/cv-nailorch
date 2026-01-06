# Python 手搓经典计算机视觉与深度学习系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

<p align="center">
  <img src="nailong.png" width="200">
</p>

<p align="center">
  <em>I'm the nailong, I'm the real nailorch!</em>
</p>

本项目是一个**深度学习学习与实践仓库**，包含从传统机器学习到现代生成模型的多种算法实现。所有示例均基于 NumPy/CuPy 从底层实现，旨在帮助理解算法原理。

> 📌 本项目中的 `nailorch` 模块参考了 [DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3)（《深度学习入门3》）的设计与实现。

---

## 📚 算法示例

| 算法 | 描述 | 代码 | 原理文档 |
|------|------|------|----------|
| **KNN** | K-近邻分类算法 | [experiments/knn/](experiments/knn/) | [KNN.md](docs/knn/KNN.md) |
| **Softmax + MLP** | Softmax 回归与多层感知机 | [experiments/softmax/](experiments/softmax/) | [softmax.md](docs/softmax/softmax.md) |
| **TLNN** | 两层全连接神经网络 | [experiments/tlnn/](experiments/tlnn/) | [TLNN.md](docs/tlnn/TLNN.md) |
| **CNN** | 卷积神经网络 | [experiments/cnn/](experiments/cnn/) | [CNN.md](docs/cnn/CNN.md) |
| **RNN / LSTM** | 循环神经网络与长短期记忆 | [experiments/rnn/](experiments/rnn/) | [RNN.md](docs/rnn/RNN.md) |
| **DDPM** | 去噪扩散概率模型 | [experiments/diffusion/](experiments/diffusion/) | - |
| **Pico-GPT** | 轻量级 GPT 语言模型 | [experiments/pico-gpt/](experiments/pico-gpt/) | [模型权重](https://huggingface.co/EHnotgod/pico-gpt) |

更多更详细的原理讲解请关注本项目的飞书文档，以及本目录下的 PDF 教程。

📎 **飞书文档**：[点击访问](https://scnd2n1l49md.feishu.cn/wiki/ZDgCwM98XiiZymks2nucxI2ynvd)

---

## 🚀 快速开始

### 环境要求

- Python ≥ 3.8
- NumPy
- Matplotlib
- CuPy（可选，建议，GPU 加速）

不同experiments内部由于不同的数据加载方式会有额外的环境需求，不过仅用于数据加载，非参与训练。

### 运行示例

```bash
# 以 CNN 为例
cd experiments/cnn
pip install -r requirements.txt
python train.py
```

---

## 📁 项目结构

```
├── experiments/               # 算法实验代码
│   ├── knn/                   # K-近邻
│   ├── softmax/               # Softmax
│   ├── tlnn/                  # 两层神经网络
│   ├── cnn/                   # 卷积神经网络
│   ├── rnn/                   # RNN / LSTM
│   ├── diffusion/             # 扩散模型
│   └── pico-gpt/              # GPT 语言模型
├── docs/                      # 算法原理文档
├── data/                      # 部分数据文件
├── nailorch/                  # 自动微分包 (基于 DeZero改造)
├── llm_logs/                  # AI交互日志
└── Textbook.pdf               # 总教材pdf文件
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
- [豆包AI](https://www.doubao.com/) - 奶龙图片由豆包AI生成
