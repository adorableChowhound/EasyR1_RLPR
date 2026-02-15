# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
RLPR Reward Manager

Implements probability reward computation for RLPR.
"""

import re
from collections import defaultdict
from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ..reward.config import RewardConfig


# ============================================================================
# Shaping Functions (imported from reward function)
# ============================================================================

def sigmoid_k(x: float, k: float = 6) -> float:
    """S-shaped curve mapping [0,1] to [0,1] with steepness k."""
    if k == 0:
        return 1 / (1 + np.exp(-x))
    x_scaled = k * (2 * x - 1)
    sigmoid = 1 / (1 + np.exp(-x_scaled))
    return sigmoid


def threshold_t(x: float, t: float) -> float:
    """Threshold function."""
    return 0.0 if x < t else x


def leaky_relu_like(x: float, threshold: float, alpha: float = 0.01) -> float:
    """Leaky ReLU-like function."""
    return alpha * x if x < threshold else x


def identity(x: float) -> float:
    """Identity function."""
    return x


def get_shaping_function(shaping_function_name: str):
    """Get shaping function by name."""
    if shaping_function_name == 'identity':
        return identity
    elif shaping_function_name.startswith('threshold_'):
        threshold = float(shaping_function_name.split('_')[1])
        return partial(threshold_t, t=threshold)
    elif shaping_function_name.startswith('sigmoid_'):
        k = float(shaping_function_name.split('_')[1])
        return partial(sigmoid_k, k=k)
    elif shaping_function_name.startswith('leaky_'):
        threshold = float(shaping_function_name.split('_')[1])
        return partial(leaky_relu_like, threshold=threshold)
    else:
        raise ValueError(f"Unsupported shaping function: {shaping_function_name}")


def format_reward(response: str, format_mode: str = 'R1') -> float:
    """Check if response follows expected format."""
    def _validate_tags(input_string):
        if format_mode == 'R1':
            tags = ['<think>', '</think>', '<answer>', '</answer>']
        elif format_mode == 'R1_nothink':
            tags = ['<answer>', '</answer>']
        else:
            raise ValueError(f"Unsupported format mode: {format_mode}")
        for tag in tags:
            if input_string.count(tag) != 1:
                return 0.0
        return 1.0

    if _validate_tags(response) == 0.0:
        return 0.0

    if format_mode == 'R1':
        pattern = re.compile(r'<think>.*</think>.*<answer>.*</answer>.*', re.DOTALL)
    elif format_mode == 'R1_nothink':
        pattern = re.compile(r'.*<answer>.*</answer>.*', re.DOTALL)

    return 1.0 if re.fullmatch(pattern, response) else 0.0


def compute_prob_score(
    log_probs: torch.Tensor,
    ground_truth_mask: torch.Tensor,
    aggregation: str = 'mean_exp_log_softmax'
) -> float:
    """Compute probability score from log probs and GT mask."""
    if ground_truth_mask.sum() == 0:
        return 0.0

    gt_log_probs = log_probs[ground_truth_mask.bool()]
    gt_log_probs = gt_log_probs.to(torch.float32)

    if aggregation == 'mean_exp_log_softmax':
        score = torch.mean(torch.exp(gt_log_probs)).item()
    elif aggregation == 'mean_log_softmax':
        score = torch.mean(gt_log_probs).item()
    elif aggregation == 'exp_sum_log_softmax':
        score = torch.exp(torch.sum(gt_log_probs)).item()
    elif aggregation == 'exp_mean_log_softmax':
        score = torch.exp(torch.mean(gt_log_probs)).item()
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    return max(0.0, min(1.0, score))


# ============================================================================
# RLPR Reward Manager
# ============================================================================

class RLPRRewardManager:
    """Reward manager for RLPR (Reinforcement Learning from Probability Reward)."""

    def __init__(
        self,
        config: RewardConfig,
        tokenizer: PreTrainedTokenizer,
        format_mode: str = 'R1',
        format_weight: float = 0.1,
        aggregation: str = 'mean_exp_log_softmax',
        shaping_function_name: str = 'identity',
        use_debiasing: bool = True,
    ):
        """
        Args:
            config: Reward configuration
            tokenizer: Tokenizer for decoding
            format_mode: Format validation mode ('R1' or 'R1_nothink')
            format_weight: Weight for format reward (default 0.1)
            aggregation: Probability aggregation method
            shaping_function_name: Reward shaping function name
            use_debiasing: Whether to use scoreA baseline for debiasing
        """
        self.config = config
        self.tokenizer = tokenizer
        self.format_mode = format_mode
        self.format_weight = format_weight
        self.aggregation = aggregation
        self.use_debiasing = use_debiasing
        self.shaping_fn = get_shaping_function(shaping_function_name)

        print(f"Initialized RLPRRewardManager:")
        print(f"  - format_mode: {format_mode}")
        print(f"  - format_weight: {format_weight}")
        print(f"  - aggregation: {aggregation}")
        print(f"  - shaping_function: {shaping_function_name}")
        print(f"  - use_debiasing: {use_debiasing}")

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """
        Compute RLPR rewards.

        Expects the data to contain:
            - responses: Generated response tokens
            - response_mask: Mask for valid response tokens
            - old_log_probs_pr: Log probabilities for replaced sequences
            - ground_truth_mask_pr: Binary masks for ground truth tokens
            - (optional) scoreA: Pre-computed baseline scores

        Returns:
            Tuple of (reward_tensor, reward_metrics)
        """
        # Check if RLPR-specific data is available
        if 'old_log_probs_pr' not in data.batch or 'ground_truth_mask_pr' not in data.batch:
            print("WARNING: RLPR requires 'old_log_probs_pr' and 'ground_truth_mask_pr' in batch data.")
            print("Falling back to format rewards only.")
            return self._compute_format_only(data)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)

        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        log_probs_pr = data.batch["old_log_probs_pr"]
        gt_masks_pr = data.batch["ground_truth_mask_pr"]

        for i in range(len(data)):
            # Get valid response length and decode
            cur_response_length = int(response_length[i].item())
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids,
                skip_special_tokens=self.config.skip_special_tokens
            )

            # 1. Format reward
            format_score = format_reward(response_str, format_mode=self.format_mode)

            # 2. Probability reward (scoreB)
            prob_score = compute_prob_score(
                log_probs=log_probs_pr[i],
                ground_truth_mask=gt_masks_pr[i],
                aggregation=self.aggregation
            )

            # 3. Debiasing (subtract scoreA if available)
            scoreA = 0.0
            if self.use_debiasing and 'scoreA' in data.non_tensor_batch:
                scoreA = data.non_tensor_batch['scoreA'][i]
                if isinstance(scoreA, torch.Tensor):
                    scoreA = scoreA.item()

            score_delta = prob_score - scoreA
            score_delta = max(0.0, min(1.0, score_delta))

            # 4. Apply shaping function
            shaped_score = self.shaping_fn(score_delta)

            # 5. Combine with format reward
            overall_score = (1 - self.format_weight) * shaped_score + self.format_weight * format_score

            # Assign reward to last token position (sparse reward)
            reward_tensor[i, cur_response_length - 1] = overall_score

            # Collect metrics
            reward_metrics["overall"].append(overall_score)
            reward_metrics["format"].append(format_score)
            reward_metrics["probability"].append(prob_score)
            reward_metrics["scoreA"].append(scoreA)
            reward_metrics["score_delta"].append(score_delta)
            reward_metrics["shaped_score"].append(shaped_score)

        return reward_tensor, reward_metrics

    def _compute_format_only(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Fallback: compute format rewards only."""
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)

        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)

        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids,
                skip_special_tokens=self.config.skip_special_tokens
            )

            format_score = format_reward(response_str, format_mode=self.format_mode)
            reward_tensor[i, cur_response_length - 1] = format_score

            reward_metrics["overall"].append(format_score)
            reward_metrics["format"].append(format_score)
            reward_metrics["probability"].append(0.0)

        return reward_tensor, reward_metrics
