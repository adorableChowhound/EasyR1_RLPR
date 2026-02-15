# EasyR1-RLPR: RLPR Implementation Based on EasyR1

<h4 align="center">
    <p>
        <a href="README_zh.md">中文</a> | <b>English</b>
    </p>
</h4>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-RLPR-purple)](https://arxiv.org/abs/2506.18254)
[![EasyR1](https://img.shields.io/badge/Framework-EasyR1-green)](https://github.com/hiyouga/EasyR1)
[![Original RLPR](https://img.shields.io/badge/Original-RLPR-orange)](https://github.com/OpenBMB/RLPR)

</div>

## 📖 Project Overview

While the original [RLPR](https://github.com/OpenBMB/RLPR) project demonstrates excellent results, it has a steep learning curve. This project reimplements the RLPR algorithm based on the [EasyR1](https://github.com/hiyouga/EasyR1) framework, aiming to provide a simpler and more user-friendly training pipeline.

**RLPR (Reinforcement Learning with Reference Probability Reward)** is a reinforcement learning method that uses reference answer generation probabilities as reward signals to enhance the reasoning capabilities of large language models without requiring external verifiers.

### Current Support

- ✅ Text-only models: Qwen2.5-7B
- ✅ LoRA fine-tuning
- ✅ Probability Reward
- ✅ ScoreA Debiasing
- ✅ EMA Dynamic Filtering

## 🚀 Quick Start

### 1. Environment Setup

#### Option 1: Using Docker (Recommended)

```bash
# Pull the pre-built Docker image
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0

# Start the container
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

#### Option 2: Using Apptainer

If your environment doesn't support Docker, you can use Apptainer:

```bash
# Pull the image
apptainer pull easyr1.sif docker://hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0

# Start the container (modify mount paths according to your setup)
apptainer shell --nv --cleanenv --bind /your/data/path:/your/data/path easyr1.sif
```

#### Option 3: Manual Installation

If you prefer to set up the environment manually, install the following dependencies:

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

### 2. Data Preparation

#### Download Training and Test Datasets

Use Hugging Face CLI to download the official RLPR datasets:

```bash
# Download training dataset
huggingface-cli download --repo-type dataset --resume-download openbmb/RLPR-Train-Dataset --local-dir ./datasets/train

# Download test dataset
huggingface-cli download --repo-type dataset --resume-download openbmb/RLPR-Evaluation --local-dir ./datasets/test
```

After downloading, your data directory structure should look like this:

```
datasets/
├── train/
│   └── rlpr_train.parquet
└── test/
    ├── MMLUPro-1000_Avg2.parquet
    ├── Math-500_Avg2.parquet
    ├── gpqa_diamond_Avg4.parquet
    ├── AIME2024_Avg16.parquet
    ├── WebInstruct-verified-val_Avg2.parquet
    ├── Minerva_Avg4.parquet
    └── TheoremQA_Avg2.parquet
```

### 3. Prepare Base Model

Download the Qwen2.5-7B base model. You can download from Hugging Face or ModelScope:

```bash
# Download from ModelScope (recommended for users in China)
export USE_MODELSCOPE_HUB=1
# The model will be automatically downloaded to ~/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B
```

Alternatively, manually specify the model path by modifying the `MODEL_PATH` variable in the script.

### 4. Compute ScoreA

Before training, you need to compute ScoreA for the training data (used for debiasing):

```bash
sh scripts/compute_scoreA_qwen2_5_7b.sh
```

**Script Details**:
- Input file: `datasets/train/rlpr_train.parquet`
- Output file: `datasets/train/qwen2_5_7b_rlpr_train_with_scoreA.parquet`
- Aggregation method: `mean_exp_log_softmax` (recommended)
- Batch size: 8 (adjust according to GPU memory)

This step uses the base model to compute reference answer probabilities for each sample, serving as the debiasing baseline for subsequent training.

### 5. Start Training

After computing ScoreA, run the training script:

```bash
bash examples/qwen2_5_7b_rlpr_lora.sh
```

**Training Configuration**:

| Configuration | Default Value | Description |
|---------------|---------------|-------------|
| Number of GPUs | 4 | Uses 4 GPUs (modify `CUDA_VISIBLE_DEVICES`) |
| Batch Size | 96 | Global batch size |
| LoRA Rank | 64 | LoRA rank |
| Learning Rate | 1e-6 | Actor learning rate |
| Format Mode | R1 | Use R1 format (with thinking process) |
| Format Weight | 0.1 | Format reward weight (alpha) |
| Debiasing | true | Use ScoreA debiasing |
| EMA Filtering | true | Enable dynamic EMA filtering |

**Training Monitoring**:

Training logs and checkpoints will be saved in:
```
checkpoints/easy_r1/qwen2_5_7b_rlpr_lora/
```

You can use Tensorboard to view training curves:
```bash
tensorboard --logdir checkpoints/easy_r1/qwen2_5_7b_rlpr_lora/
```

## 🔧 Advanced Configuration

### Modifying Training Parameters

Edit the `examples/qwen2_5_7b_rlpr_lora.sh` file to adjust the following parameters:

```bash
# Format mode: 'R1' or 'R1_nothink'
FORMAT_MODE=R1

# Format reward weight (alpha in the paper)
FORMAT_WEIGHT=0.1

# Probability aggregation method
AGGREGATION=mean_exp_log_softmax

# Reward shaping function
SHAPING_FUNCTION=threshold_0

# Whether to use ScoreA debiasing
USE_DEBIASING=true

# EMA filtering configuration
ENABLE_EMA_FILTERING=true
FILTER_MODE=ema_std
FILTER_EMA_RATIO=0.99
STD_FILTER_BETA=0.5
```

## 📝 Implementation Details

This project implements the core features of RLPR on the EasyR1 framework:

1. **Probability Reward Calculation**: `examples/reward_function/rlpr.py`
   - Uses reference answer generation probabilities as reward signals
   - Supports multiple aggregation methods (mean_exp_log_softmax, etc.)

2. **ScoreA Debiasing**: `verl/utils/rlpr_helper.py`
   - Computes reference probabilities from the base model
   - Subtracts ScoreA during training to eliminate bias

3. **EMA Dynamic Filtering**: `verl/trainer/core_algos.py`
   - Dynamic filtering based on reward standard deviation
   - Automatically adjusts filtering thresholds

4. **R1 Format Reward**: `verl/workers/reward/rlpr_manager.py`
   - Checks if output conforms to R1 format (with `<think>` tags)
   - Provides format rewards to encourage thinking process

## 🙏 Acknowledgments

- [EasyR1](https://github.com/hiyouga/EasyR1): Provides an efficient RL training framework
- [RLPR](https://github.com/OpenBMB/RLPR): Provides the original algorithm implementation and datasets
- [veRL](https://github.com/volcengine/verl): The foundational framework for EasyR1

## 📧 Contact

For questions or suggestions, feel free to submit an Issue or Pull Request.

## 📄 License

This project follows the Apache 2.0 license. The datasets follow the CC BY NC 4.0 license (non-commercial use only).
