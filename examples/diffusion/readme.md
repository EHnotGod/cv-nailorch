# Stable DDPM for MNIST

> 这是一个基于 Nailorch（底层仅为numpy/cupy） 的 DDPM / 条件扩散（Conditional Diffusion）示例工程，目标数据集为 MNIST。  
> 与传统只生成单一数字的 DDPM 不同，本项目的模型支持带类别条件的噪声预测（在采样时可以指定生成 0-9 中任意数字或混合类别）。

---

## 目录结构

```shell
├── diffusion/
├── ddpm.npz
├── DDPMP_Model.py
├── DDPMP_Train.py
├── requirements.txt
├── stable-ddpm3.npz
├── Stable_Diffusion_Model.py
└── Stable_Diffusion_Train.py <-- 当前主训练/采样脚本
├── readme.md
└── ddpm_samples3.png
```

## 依赖（requirements）
在 `diffusion/requirements.txt` 中已有依赖列举，主要依赖包括：

- Python 3.8+
- numpy
- matplotlib
- cupy（如使用 GPU）
- 其他：根据 `requirements.txt` 安装

安装示例：
```bash
pip install -r diffusion/requirements.txt
```
> 如果使用 GPU，请确保已正确安装对应 CUDA 版本的 cupy。

## 如何训练

准备数据，或者直接运行（脚本默认使用 nailorch.datasets.MNIST，会自动下载/加载训练集）。：

```shell
python Stable_Diffusion_Train.py
```

若运行DDPM_Train.py，则会执行普通的DDPM，脚本默认训练“0”这个数字。

## 如何采样（生成图片）

训练完成后，脚本包含了采样流程；也可以单独运行采样部分（或把训练/采样分离成两个脚本）

输出文件（默认）
```
stable-ddpm3.npz — 训练好的模型权重（示例）
ddpm_samples3.png — 采样保存的网格图片
```

该项目为研究/学习用途。若有问题或需要帮助，请在仓库 issue 中描述你的运行环境、完整报错信息和你修改的关键配置。