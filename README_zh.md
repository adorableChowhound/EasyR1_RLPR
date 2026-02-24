# EasyR1-RLPR: 基于 EasyR1 的 RLPR 实现

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-RLPR-purple)](https://arxiv.org/abs/2506.18254)
[![EasyR1](https://img.shields.io/badge/Framework-EasyR1-green)](https://github.com/hiyouga/EasyR1)
[![Original RLPR](https://img.shields.io/badge/Original-RLPR-orange)](https://github.com/OpenBMB/RLPR)

</div>   

## 📖 项目简介

原始的 [RLPR](https://github.com/OpenBMB/RLPR) 项目虽然效果出色，但上手难度较高。本项目基于 [EasyR1](https://github.com/hiyouga/EasyR1) 框架重新实现了 RLPR 算法，旨在提供更简单易用的训练流程。

**RLPR (Reinforcement Learning with Reference Probability Reward)** 是一种使用参考答案生成概率作为奖励信号的强化学习方法，无需外部验证器即可增强大语言模型的推理能力。

### 当前支持

- ✅ 纯文本模型：Qwen2.5-7B
- ✅ LoRA 微调
- ✅ 概率奖励 (Probability Reward)
- ✅ ScoreA 去偏 (Debiasing)
- ✅ EMA 动态过滤

## 🚀 快速开始

### 1. 环境配置

#### 方法一：使用 Docker（推荐）

```bash
# 拉取预构建的 Docker 镜像
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0

# 启动容器
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

#### 方法二：使用 Apptainer

如果你的环境不支持 Docker，可以使用 Apptainer：

```bash
# 拉取镜像
apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0

# 启动容器（根据实际情况修改挂载路径）
apptainer shell --nv --cleanenv --bind /your/data/path:/your/data/path easyr1.sif
```

#### 方法三：手动安装

如果你想手动配置环境，需要安装以下依赖：

Software Requirements
- Python 3.9+
- transformers>=4.54.0
- flash-attn>=2.4.3
- vllm>=0.8.3

Installation
```bash
git clone https://github.com/adorableChowhound/EasyR1_RLPR.git
cd EasyR1_RLPR
pip install -e .
```

### 2. 数据准备

#### 下载训练和测试数据集

使用 Hugging Face CLI 下载 RLPR 官方数据集：

```bash
# 下载训练数据集
huggingface-cli download --repo-type dataset --resume-download openbmb/RLPR-Train-Dataset --local-dir ./datasets/train

# 下载测试数据集
huggingface-cli download --repo-type dataset --resume-download openbmb/RLPR-Evaluation --local-dir ./datasets/test
```

下载完成后，你的数据目录结构应该如下：

```
datasets/
├── test/
│   ├── AIME2024_Avg16.parquet
│   ├── gpqa_diamond_Avg4.parquet
│   ├── Math-500_Avg2.parquet
│   ├── Minerva_Avg4.parquet
│   ├── MMLUPro-1000_Avg2.parquet
│   ├── TheoremQA_Avg2.parquet
│   └── WebInstruct-verified-val_Avg2.parquet
└── train/
    └── rlpr_train.parquet
```

### 3. 准备基座模型

下载 Qwen2.5-7B 基座模型。你可以从 Hugging Face 或 ModelScope 下载：

```bash
# 从 ModelScope 下载（国内推荐）
export USE_MODELSCOPE_HUB=1
# 模型会自动下载到 ~/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B
```

或者手动指定模型路径，修改脚本中的 `MODEL_PATH` 变量。

### 4. 计算 ScoreA

在训练之前，需要先计算训练数据的 ScoreA（用于去偏）：

```bash
sh scripts/compute_scoreA_qwen2_5_7b.sh
```

**脚本说明**：
- 输入文件：`datasets/train/rlpr_train.parquet`
- 输出文件：`datasets/train/qwen2_5_7b_rlpr_train_with_scoreA.parquet`
- 聚合方法：`mean_exp_log_softmax`（推荐）
- 批次大小：8（可根据显存调整）

这一步会使用基座模型计算每个样本的参考答案概率，作为后续训练的去偏基准。

### 5. 开始训练

计算完 ScoreA 后，运行训练脚本：

```bash
bash examples/qwen2_5_7b_rlpr_lora.sh
```

### 6. Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

**训练配置说明**：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| GPU 数量 | 4 | 使用 4 张 GPU（可修改 `CUDA_VISIBLE_DEVICES`） |
| 批次大小 | 96 | 全局批次大小 |
| LoRA Rank | 64 | LoRA 秩 |
| 学习率 | 1e-6 | Actor 学习率 |
| 格式模式 | R1 | 使用 R1 格式（带思考过程） |
| 格式权重 | 0.1 | 格式奖励权重（alpha） |
| 去偏 | true | 使用 ScoreA 去偏 |
| EMA 过滤 | true | 启用动态 EMA 过滤 |

**训练监控**：

训练日志和检查点会保存在：
```
checkpoints/easy_r1/qwen2_5_7b_rlpr_lora/
```

你可以使用 Tensorboard 查看训练曲线：
```bash
tensorboard --logdir checkpoints/easy_r1/qwen2_5_7b_rlpr_lora/
```

## 🔧 高级配置

### 修改训练参数

编辑 `examples/qwen2_5_7b_rlpr_lora.sh` 文件，可以调整以下参数：

```bash
# 格式模式：'R1' 或 'R1_nothink'
FORMAT_MODE=R1

# 格式奖励权重（论文中的 alpha）
FORMAT_WEIGHT=0.1

# 概率聚合方法
AGGREGATION=mean_exp_log_softmax

# 奖励塑形函数
SHAPING_FUNCTION=threshold_0

# 是否使用 ScoreA 去偏
USE_DEBIASING=true

# EMA 过滤配置
ENABLE_EMA_FILTERING=true
FILTER_MODE=ema_std
FILTER_EMA_RATIO=0.99
STD_FILTER_BETA=0.5
```

## 📝 实现细节

本项目在 EasyR1 框架上实现了 RLPR 的核心功能：

1. **概率奖励计算**：`examples/reward_function/rlpr.py`
   - 使用参考答案的生成概率作为奖励信号
   - 支持多种聚合方法（mean_exp_log_softmax 等）

2. **ScoreA 去偏**：`verl/utils/rlpr_helper.py`
   - 计算基座模型的参考概率
   - 在训练时减去 ScoreA 以消除偏差

3. **EMA 动态过滤**：`verl/trainer/core_algos.py`
   - 基于奖励标准差的动态过滤
   - 自动调整过滤阈值

4. **R1 格式奖励**：`verl/workers/reward/rlpr_manager.py`
   - 检查输出是否符合 R1 格式（带 `<think>` 标签）
   - 给予格式奖励以鼓励思考过程

## 🙏 致谢

- [EasyR1](https://github.com/hiyouga/EasyR1)：提供了高效的 RL 训练框架
- [RLPR](https://github.com/OpenBMB/RLPR)：提供了原始算法实现和数据集
- [veRL](https://github.com/volcengine/verl)：EasyR1 的基础框架

## 📧 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

