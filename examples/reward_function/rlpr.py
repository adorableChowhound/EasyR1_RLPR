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
RLPR (Reinforcement Learning from Probability Reward) Implementation

Based on paper: https://arxiv.org/abs/2506.18254
"""

import re
from functools import partial
from typing import Any

import numpy as np
import torch


# Metadata
REWARD_NAME = "rlpr"
REWARD_TYPE = "batch"


# ============================================================================
# Shaping Functions
# ============================================================================

def sigmoid_k(x: float, k: float = 6) -> float:
    """
    Maps a value x in [0, 1] to [0, 1] using a sigmoid-like S-shaped curve.

    Parameters:
        x: Input value in the range [0, 1]
        k: Steepness parameter for the sigmoid curve. Higher values make the curve steeper.

    Returns:
        Mapped value in the range [0, 1]
    """
    if k == 0:  # vanilla sigmoid
        return 1 / (1 + np.exp(-x))

    # Shift and scale x to the range [-k, k]
    x_scaled = k * (2 * x - 1)

    # Apply the sigmoid function
    sigmoid = 1 / (1 + np.exp(-x_scaled))

    return sigmoid


def threshold_t(x: float, t: float) -> float:
    """Simple threshold function: returns 0 if x < t, else x."""
    return 0.0 if x < t else x


def leaky_relu_like(x: float, threshold: float, alpha: float = 0.01) -> float:
    """
    Maps a score from [0, 1] to [0, 1] using a Leaky ReLU-like function.

    Parameters:
        x: The input score in the range [0, 1]
        threshold: The threshold below which the score is scaled
        alpha: The slope for scores below the threshold (default is 0.01)

    Returns:
        The transformed score in the range [0, 1]
    """
    if x < threshold:
        return alpha * x
    else:
        return x


def identity(x: float) -> float:
    """Identity function: returns x unchanged."""
    return x


def get_shaping_function(shaping_function_name: str):
    """
    Get the reward shaping function based on the configuration name.

    Supported functions:
        - identity: f(x) = x
        - threshold_t: f(x) = 0 if x < t else x
        - sigmoid_k: S-shaped curve with steepness k
        - leaky_t: Leaky ReLU-like with threshold t
    """
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


# ============================================================================
# Format Reward
# ============================================================================

def format_reward(response: str, format_mode: str = 'R1') -> float:
    """
    Check if the response follows the expected format.

    Format modes:
        - 'R1': Expects <think>...</think><answer>...</answer>
        - 'R1_nothink': Expects <answer>...</answer>

    Returns:
        1.0 if format is valid, 0.0 otherwise
    """
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

    match_result = re.fullmatch(pattern, response)
    return 1.0 if match_result else 0.0


# ============================================================================
# Probability Aggregation Functions
# ============================================================================

def compute_prob_reward(
    log_probs: torch.Tensor,
    ground_truth_mask: torch.Tensor,
    aggregation: str = 'mean_exp_log_softmax'
) -> float:
    """
    Compute probability reward from log probabilities and ground truth mask.

    Parameters:
        log_probs: Log probabilities for each token (response_length,)
        ground_truth_mask: Binary mask indicating ground truth tokens (response_length,)
        aggregation: How to aggregate probabilities
            - 'mean_exp_log_softmax': mean(exp(log_p_i)) - recommended by paper
            - 'mean_log_softmax': mean(log_p_i)
            - 'exp_sum_log_softmax': exp(sum(log_p_i)) - product of probabilities
            - 'exp_mean_log_softmax': exp(mean(log_p_i)) - geometric mean

    Returns:
        Scalar probability reward in [0, 1]
    """
    if ground_truth_mask.sum() == 0:
        return 0.0

    # Extract log probs for ground truth tokens only
    gt_log_probs = log_probs[ground_truth_mask.bool()]

    # Convert to float32 for precision
    gt_log_probs = gt_log_probs.to(torch.float32)

    # Aggregate based on method
    if aggregation == 'mean_exp_log_softmax':
        # Paper's recommended method: mean of probabilities
        score = torch.mean(torch.exp(gt_log_probs)).item()
    elif aggregation == 'mean_log_softmax':
        # Mean of log probabilities
        score = torch.mean(gt_log_probs).item()
    elif aggregation == 'exp_sum_log_softmax':
        # Product of probabilities (sequence probability)
        score = torch.exp(torch.sum(gt_log_probs)).item()
    elif aggregation == 'exp_mean_log_softmax':
        # Geometric mean of probabilities
        score = torch.exp(torch.mean(gt_log_probs)).item()
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")

    # Ensure score is in [0, 1]
    score = max(0.0, min(1.0, score))

    return score


# ============================================================================
# Main Reward Function (Extended Interface)
# ============================================================================

def compute_score_extended(
    reward_inputs: list[dict[str, Any]],
    batch_data: dict[str, torch.Tensor],
    format_weight: float = 0.1,
    format_mode: str = 'R1',
    aggregation: str = 'mean_exp_log_softmax',
    shaping_function_name: str = 'identity',
    use_debiasing: bool = True,
) -> list[dict[str, float]]:
    """
    Compute RLPR rewards with access to batch data.

    This is an EXTENDED interface that requires batch_data containing:
        - 'old_log_probs_pr': Log probabilities for replaced sequences
        - 'ground_truth_mask_pr': Binary masks for ground truth tokens
        - (optional) 'scoreA': Pre-computed baseline scores for debiasing

    Parameters:
        reward_inputs: List of reward input dictionaries (standard interface)
        batch_data: Dictionary containing batch tensors for RLPR
        format_weight: Weight for format reward (default 0.1)
        format_mode: Format validation mode ('R1' or 'R1_nothink')
        aggregation: Probability aggregation method
        shaping_function_name: Reward shaping function name
        use_debiasing: Whether to use scoreA baseline for debiasing

    Returns:
        List of score dictionaries with keys: overall, format, probability
    """
    shaping_fn = get_shaping_function(shaping_function_name)
    scores = []

    # Extract batch tensors
    log_probs_pr = batch_data.get('old_log_probs_pr')  # (batch_size, response_length)
    gt_masks_pr = batch_data.get('ground_truth_mask_pr')  # (batch_size, response_length)
    scoreA_list = batch_data.get('scoreA', None)  # Optional baseline scores

    if log_probs_pr is None or gt_masks_pr is None:
        raise ValueError("RLPR requires 'old_log_probs_pr' and 'ground_truth_mask_pr' in batch_data")

    for i, reward_input in enumerate(reward_inputs):
        response = reward_input["response"]

        # 1. Format reward
        format_score = format_reward(response, format_mode=format_mode)

        # 2. Probability reward (scoreB)
        prob_score = compute_prob_reward(
            log_probs=log_probs_pr[i],
            ground_truth_mask=gt_masks_pr[i],
            aggregation=aggregation
        )

        # 3. Debiasing (subtract scoreA if available)
        if use_debiasing and scoreA_list is not None:
            if isinstance(scoreA_list, (list, tuple)):
                scoreA = scoreA_list[i]
            elif isinstance(scoreA_list, np.ndarray):
                scoreA = float(scoreA_list[i])
            elif torch.is_tensor(scoreA_list):
                scoreA = scoreA_list[i].item()
            else:
                # Assume it's already a scalar
                scoreA = float(scoreA_list)
            prob_score = prob_score - scoreA

        # Clip to [0, 1] after debiasing
        prob_score = max(0.0, min(1.0, prob_score))

        # 4. Apply shaping function
        prob_score = shaping_fn(prob_score)

        # 5. Combine with format reward
        overall_score = (1 - format_weight) * prob_score + format_weight * format_score

        scores.append({
            "overall": overall_score,
            "format": format_score,
            "probability": prob_score,
        })

    return scores


# ============================================================================
# Simplified Interface (Fallback - uses only standard reward inputs)
# ============================================================================

def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    format_mode: str = 'R1',
) -> list[dict[str, float]]:
    """
    Simplified RLPR interface that only validates format.

    This is a fallback when RLPR-specific batch data is not available.
    It only computes format rewards.

    NOTE: For full RLPR functionality, use compute_score_extended() instead.
    """
    print("WARNING: Using simplified RLPR interface without probability rewards.")
    print("To enable full RLPR, ensure batch data contains 'old_log_probs_pr' and 'ground_truth_mask_pr'")

    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        format_score = format_reward(response, format_mode=format_mode)

        scores.append({
            "overall": format_score,
            "format": format_score,
            "probability": 0.0,  # Not available in simplified mode
        })

    return scores
