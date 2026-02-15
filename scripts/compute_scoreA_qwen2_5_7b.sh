#!/bin/bash

# Compute scoreA for pure text model (Qwen2.5-7B)

export CUDA_VISIBLE_DEVICES=0

python examples/compute_scoreA.py \
      --model_path ~/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B \
      --input_file datasets/train/rlpr_train.parquet \
      --output_file datasets/train/qwen2_5_7b_rlpr_train_with_scoreA.parquet \
      --aggregation mean_exp_log_softmax \
      --batch_size 8
