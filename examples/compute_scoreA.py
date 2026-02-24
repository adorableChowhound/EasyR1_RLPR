"""
Compute scoreA (Baseline Probability) for RLPR

This script computes the baseline probability of ground truth answers
without reasoning steps. scoreA is used for debiasing in RLPR.

Usage:
    python examples/compute_scoreA.py --model_path <path> --input_file <path> --output_file <path>

Input format (JSONL):
    {"prompt": "What is 2 + 2?", "ground_truth": "4"}

Output format (JSONL):
    {"prompt": "What is 2 + 2?", "ground_truth": "4", "scoreA": 0.15}
"""

import argparse
import json
import os
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd


def compute_scoreA_single(
    model,
    tokenizer,
    prompt: str,
    ground_truth: str,
    aggregation: str = 'mean_exp_log_softmax',
    device: str = 'cuda',
) -> float:
    """
    Compute baseline probability for a single example.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt (e.g., "What is 2 + 2? Answer: ")
        ground_truth: Ground truth answer (e.g., "4")
        aggregation: How to aggregate token probabilities
        device: Device to run on

    Returns:
        scoreA: Baseline probability score
    """
    # Tokenize prompt and ground truth separately
    prompt_encoding = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    gt_encoding = tokenizer(ground_truth, return_tensors='pt', add_special_tokens=False)

    prompt_ids = prompt_encoding['input_ids']
    gt_ids = gt_encoding['input_ids']

    # Concatenate
    full_ids = torch.cat([prompt_ids, gt_ids], dim=1).to(device)

    # Get model outputs
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Get log probabilities for ground truth tokens
    # We need logits at positions [len(prompt)-1 : len(prompt)+len(gt)-1]
    prompt_len = prompt_ids.shape[1]
    gt_len = gt_ids.shape[1]

    # Get logits for predicting ground truth tokens
    gt_logits = logits[0, prompt_len-1:prompt_len+gt_len-1, :]  # [gt_len, vocab_size]

    # Get log softmax
    log_probs = torch.log_softmax(gt_logits, dim=-1)  # [gt_len, vocab_size]

    # Extract log probs for actual ground truth tokens
    gt_token_ids = gt_ids[0]  # [gt_len]
    gt_log_probs = log_probs[range(gt_len), gt_token_ids]  # [gt_len]

    # Aggregate according to method
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


def compute_scoreA_batch(
    model,
    tokenizer,
    data: List[Dict],
    batch_size: int = 8,
    aggregation: str = 'mean_exp_log_softmax',
    device: str = 'cuda',
    prompt_suffix: str = " Answer: ",
) -> List[Dict]:
    """
    Compute scoreA for a batch of examples.

    Args:
        model: Language model
        tokenizer: Tokenizer
        data: List of dicts with 'prompt' and 'ground_truth' keys
        batch_size: Batch size for processing
        aggregation: Aggregation method
        device: Device to run on
        prompt_suffix: Suffix to add to prompt (e.g., " Answer: ")

    Returns:
        List of dicts with added 'scoreA' field
    """
    results = []

    for i in tqdm(range(0, len(data), batch_size), desc="Computing scoreA"):
        batch = data[i:i+batch_size]

        for item in batch:
            # Handle different data formats
            prompt_data = item['prompt']

            # Convert numpy array to list if needed
            if hasattr(prompt_data, 'tolist'):
                prompt_data = prompt_data.tolist()

            if isinstance(prompt_data, list):
                # RLPR format: prompt is a conversation array
                # Convert to text using tokenizer's chat template
                prompt = tokenizer.apply_chat_template(
                    prompt_data,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Simple format: prompt is a string
                prompt = str(prompt_data)

            # Extract ground_truth from different locations
            if 'ground_truth' in item:
                ground_truth = item['ground_truth']
            elif 'reward_model' in item and isinstance(item['reward_model'], dict):
                ground_truth = item['reward_model']['ground_truth']
            else:
                print(f"Warning: No ground_truth found for item, skipping...")
                continue

            # Add suffix if not already present
            if not prompt.endswith(prompt_suffix):
                prompt_with_suffix = prompt + prompt_suffix
            else:
                prompt_with_suffix = prompt

            # Compute scoreA
            try:
                scoreA = compute_scoreA_single(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_with_suffix,
                    ground_truth=ground_truth,
                    aggregation=aggregation,
                    device=device,
                )

                # Add scoreA to result
                result = item.copy()
                result['scoreA'] = scoreA
                results.append(result)

            except Exception as e:
                print(f"Error processing item: {item}")
                print(f"Error: {e}")
                # Add with scoreA = 0.0
                result = item.copy()
                result['scoreA'] = 0.0
                results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description='Compute scoreA for RLPR')
    parser.add_argument('--model_path', type=str,
                        default=os.path.expanduser('~/.cache/modelscope/hub/models/Qwen/Qwen3-8B'),
                        help='Path to model (default: ~/.cache/modelscope/hub/models/Qwen/Qwen3-8B)')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input JSONL file with prompt and ground_truth')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSONL file with added scoreA')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing')
    parser.add_argument('--aggregation', type=str, default='mean_exp_log_softmax',
                        choices=['mean_exp_log_softmax', 'mean_log_softmax',
                                'exp_sum_log_softmax', 'exp_mean_log_softmax'],
                        help='Aggregation method for token probabilities')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    parser.add_argument('--prompt_suffix', type=str, default=' Answer: ',
                        help='Suffix to add to prompt for direct answer')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')

    args = parser.parse_args()

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    # Load input data
    print(f"Loading data from {args.input_file}...")
    data = []

    # Support both JSONL and parquet formats
    if args.input_file.endswith('.parquet'):
        df = pd.read_parquet(args.input_file)
        data = df.to_dict('records')
    else:
        # Assume JSONL format
        with open(args.input_file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))

    if args.max_samples:
        data = data[:args.max_samples]
        print(f"Processing first {args.max_samples} samples...")

    print(f"Loaded {len(data)} examples")

    # Compute scoreA
    results = compute_scoreA_batch(
        model=model,
        tokenizer=tokenizer,
        data=data,
        batch_size=args.batch_size,
        aggregation=args.aggregation,
        device=args.device,
        prompt_suffix=args.prompt_suffix,
    )

    # Save results
    print(f"Saving results to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Support both JSONL and parquet formats for output
    if args.output_file.endswith('.parquet'):
        df_output = pd.DataFrame(results)
        df_output.to_parquet(args.output_file, index=False)
    else:
        # Save as JSONL
        with open(args.output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Print statistics
    scoreA_values = [r['scoreA'] for r in results]
    print(f"\nStatistics:")
    print(f"  Total samples: {len(results)}")
    print(f"  Mean scoreA: {sum(scoreA_values) / len(scoreA_values):.4f}")
    print(f"  Min scoreA: {min(scoreA_values):.4f}")
    print(f"  Max scoreA: {max(scoreA_values):.4f}")
    print(f"\nDone!")


if __name__ == '__main__':
    main()
