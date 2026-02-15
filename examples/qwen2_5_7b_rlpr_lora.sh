#!/bin/bash

set -x

# RLPR Training Script - Pure Text Model (Qwen2.5-7B)
# Full RLPR implementation with probability rewards and EMA-based std filtering

# GPU configuration - use GPUs starting from card 2
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Model configuration - Pure text model (not VL)
MODEL_PATH=~/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B  # replace with your local path

# Data paths (using RLPR original dataset, copied to EasyR1)
TRAIN_FILES=./datasets/train/qwen2_5_7b_rlpr_train_with_scoreA.parquet
VAL_DIR=./datasets/test
VAL_FILES="[${VAL_DIR}/MMLUPro-1000_Avg2.parquet,${VAL_DIR}/Math-500_Avg2.parquet,${VAL_DIR}/gpqa_diamond_Avg4.parquet,${VAL_DIR}/AIME2024_Avg16.parquet,${VAL_DIR}/WebInstruct-verified-val_Avg2.parquet,${VAL_DIR}/Minerva_Avg4.parquet,${VAL_DIR}/TheoremQA_Avg2.parquet]"

# RLPR Configuration (matching RLPR reproduce_qwen.sh)
FORMAT_MODE=R1  # 'R1' or 'R1_nothink'
FORMAT_WEIGHT=0.1  # Weight for format reward (alpha in paper)
AGGREGATION=mean_exp_log_softmax  # Probability aggregation method (recommended)
SHAPING_FUNCTION=threshold_0  # Reward shaping function
USE_DEBIASING=true  # Whether to use scoreA debiasing

# EMA-based std filtering configuration (NEW!)
# Set to true to enable dynamic filtering based on reward standard deviation
ENABLE_EMA_FILTERING=true
FILTER_MODE=ema_std  # 'default' (static threshold) or 'ema_std' (dynamic EMA-based)
FILTER_EMA_RATIO=0.99  # EMA smoothing ratio (higher = more smoothing)
FILTER_EMA_START_STEP=6  # Step to start computing EMA
FILTER_START_STEP=11  # Step to start filtering
STD_FILTER_BETA=0.5  # Beta coefficient: threshold = ema_mean * beta

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.prompt_key=prompt \
    data.train_batch_size=96 \
    data.max_prompt_length=2048 \
    data.max_response_length=3072 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.lora.rank=64 \
    worker.actor.global_batch_size=96 \
    worker.actor.optim.lr=1e-6 \
    trainer.experiment_name=qwen2_5_7b_rlpr_lora \
    trainer.n_gpus_per_node=4 \
    worker.reward.reward_function=examples/reward_function/rlpr.py:compute_score_extended \
    worker.reward.rlpr_format_mode=${FORMAT_MODE} \
    +worker.reward.reward_function_kwargs.format_mode=${FORMAT_MODE} \
    +worker.reward.reward_function_kwargs.format_weight=${FORMAT_WEIGHT} \
    +worker.reward.reward_function_kwargs.aggregation=${AGGREGATION} \
    +worker.reward.reward_function_kwargs.shaping_function_name=${SHAPING_FUNCTION} \
    +worker.reward.reward_function_kwargs.use_debiasing=${USE_DEBIASING} \
    algorithm.online_filtering=${ENABLE_EMA_FILTERING} \
    algorithm.filter_mode=${FILTER_MODE} \
    algorithm.filter_ema_ratio=${FILTER_EMA_RATIO} \
    algorithm.filter_ema_start_step=${FILTER_EMA_START_STEP} \
    algorithm.filter_start_step=${FILTER_START_STEP} \
    algorithm.std_filter_beta=${STD_FILTER_BETA} \
    trainer.total_epochs=1 \
    trainer.val_freq=50 \
    trainer.save_freq=50 \
    "$@"
