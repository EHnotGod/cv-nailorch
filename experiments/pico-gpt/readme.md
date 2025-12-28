# Pico-GPT (nailorch)

一个基于numpy搭建的自动微分框架实现的轻量级 GPT 风格语言模型（Pico-GPT），  
支持 **BBPE 中文分词**、**Transformer Decoder** 结构，参数量较小，用于高效文本生成，效果普通，教学向。

---

## 📁 项目结构

```shell
├── chinese_bbpe/ # 中文 BBPE tokenizer 目录
│ └── tokenizer.json
├── gpt.py # GPT / Transformer / KV cache 推理代码
├── readme.md # 项目说明文档
└── requirements.txt # 依赖列表
```

## ✨ 特性

- ✅ **GPT Decoder-only 架构**
- ✅ **多头自注意力（Multi-Head Self-Attention）**
- ✅ **Sliding Window（自动截断历史）**
- ✅ **支持 GPU（CuPy）**
- ✅ **中文 BBPE tokenizer**

---

## 🔧 环境依赖

推荐使用 **Python ≥ 3.10**

### 依赖安装

```bash
pip install -r requirements.txt
```

## 🚀 快速开始

```shell
python gpt.py
```

程序将：

加载 tokenizer

加载 pico-gpt.npz 权重

使用 KV cache 进行自回归生成

输出生成文本

## 🧪 已知限制

当前仅支持 batch_size = 1 推理

未实现：

top-k / top-p / temperature

repetition penalty

模型规模较小，生成文本可能出现：

重复

语义跳跃

局部不连贯

这是 Pico-GPT 的设计取舍，而非 bug。

这些功能实现起来并不困难，后续可自由拓展。