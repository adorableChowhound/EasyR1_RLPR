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
RLPR Helper for Answer Replacement and Ground Truth Mask Construction
"""

import re
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer


class RLPRHelper:
    """Helper class for RLPR answer replacement and mask construction."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        format_mode: str = 'R1',
        max_prompt_length: int = 2048,
        max_response_length: int = 2048,
    ):
        """
        Args:
            tokenizer: The tokenizer to use
            format_mode: Format mode ('R1' or 'R1_nothink')
            max_prompt_length: Maximum prompt length
            max_response_length: Maximum response length
        """
        self.tokenizer = tokenizer
        self.format_mode = format_mode
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

        # Define special tokens
        self.start_think = "<think>"
        self.end_think = "</think>"
        self.start_answer = "<answer>"
        self.end_answer = "</answer>"
        self.eos_token = tokenizer.eos_token or "</s>"

    def replace_answer_with_ground_truth(
        self,
        input_ids: torch.Tensor,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        ground_truth: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Replace the answer portion of a response with ground truth.

        Args:
            input_ids: Full input IDs (prompt + response)
            prompts: Prompt IDs only
            responses: Response IDs only
            ground_truth: Ground truth answer string

        Returns:
            Tuple of:
                - input_ids_pr: Replaced input IDs
                - attention_mask_pr: Attention mask for replaced sequence
                - position_ids_pr: Position IDs for replaced sequence
                - responses_pr: Replaced response portion
                - ground_truth_mask_pr: Binary mask indicating GT token positions
        """
        # Decode the full input_ids to get the complete and correct text
        # This avoids tokenization boundary issues that occur when splitting by token count
        full_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        prompt_str = self.tokenizer.decode(prompts, skip_special_tokens=False)

        # Extract response from full text
        if full_text.startswith(prompt_str):
            response_str = full_text[len(prompt_str):]
        else:
            # Fallback: decode responses directly (may have boundary issues)
            response_str = self.tokenizer.decode(responses, skip_special_tokens=False)

        # Check if the response has valid format
        is_valid = self._validate_format(response_str)

        if not is_valid:
            # If format is invalid, return zeros (will get 0 reward)
            return self._create_zero_outputs(input_ids.shape[0])

        # Extract the reasoning part (before <answer>)
        if self.format_mode == 'R1':
            # Format: <think>reasoning</think> <answer>answer</answer>
            match = re.search(
                r'(<think>.*?</think>\s*)<answer>.*?</answer>',
                response_str,
                re.DOTALL
            )
            if not match:
                return self._create_zero_outputs(input_ids.shape[0])
            reasoning_part = match.group(1)
        else:  # R1_nothink
            # Format: <answer>answer</answer>
            match = re.search(r'(.*?)<answer>.*?</answer>', response_str, re.DOTALL)
            if not match:
                return self._create_zero_outputs(input_ids.shape[0])
            reasoning_part = match.group(1)

        # Construct new text: prompt + reasoning + <answer>ground_truth</answer>
        new_text = f"{prompt_str}{reasoning_part}{self.start_answer} {ground_truth} {self.end_answer}{self.eos_token}"

        # Tokenize new text with offset mapping to find precise boundaries
        new_encoding = self.tokenizer(
            new_text,
            return_tensors='pt',
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        new_input_ids = new_encoding['input_ids'][0]  # (seq_len,)
        offset_mapping = new_encoding['offset_mapping'][0]  # (seq_len, 2)

        # Find the byte position where prompt ends
        prompt_byte_len = len(prompt_str)

        # Find the split point between prompt and response
        # We want to include any token that overlaps with the response region
        response_start_idx = len(new_input_ids)  # Default to end if not found
        for i, (start, end) in enumerate(offset_mapping):
            # If token ends after prompt boundary, it's part of response
            if end > prompt_byte_len:
                response_start_idx = i
                break

        # Find ground truth token positions using offset mapping
        gt_token_indices = self._locate_ground_truth_tokens(
            new_text,
            ground_truth,
            offset_mapping
        )

        # Split into prompt and response using the correct boundary
        new_prompts = new_input_ids[:response_start_idx]
        new_responses = new_input_ids[response_start_idx:] if response_start_idx < len(new_input_ids) else torch.tensor([], dtype=torch.long)

        # Pad to max lengths
        new_prompts = self._pad_left(new_prompts, self.max_prompt_length, self.tokenizer.pad_token_id)
        new_responses = self._pad_right(new_responses, self.max_response_length, self.tokenizer.pad_token_id)

        # Construct full inputs
        input_ids_pr = torch.cat([new_prompts, new_responses], dim=-1)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask_pr = (input_ids_pr != self.tokenizer.pad_token_id).long()

        # Create position IDs
        position_ids_pr = self._compute_position_ids(attention_mask_pr)

        # Create ground truth mask
        ground_truth_mask_pr = torch.zeros_like(new_responses)
        if gt_token_indices:
            # Adjust indices to response coordinates using the correct boundary
            for idx in gt_token_indices:
                adjusted_idx = idx - response_start_idx
                if adjusted_idx >= 0 and adjusted_idx < len(ground_truth_mask_pr):
                    ground_truth_mask_pr[adjusted_idx] = 1

        return input_ids_pr, attention_mask_pr, position_ids_pr, new_responses, ground_truth_mask_pr

    def _validate_format(self, response: str) -> bool:
        """Validate that response follows expected format."""
        if self.format_mode == 'R1':
            # Must have exactly one of each tag
            tags = [self.start_think, self.end_think, self.start_answer, self.end_answer]
            for tag in tags:
                if response.count(tag) != 1:
                    return False
            # Must match the pattern
            pattern = r'<think>.*</think>.*<answer>.*</answer>.*'
            return bool(re.fullmatch(pattern, response, re.DOTALL))
        else:  # R1_nothink
            tags = [self.start_answer, self.end_answer]
            for tag in tags:
                if response.count(tag) != 1:
                    return False
            pattern = r'.*<answer>.*</answer>.*'
            return bool(re.fullmatch(pattern, response, re.DOTALL))

    def _locate_ground_truth_tokens(
        self,
        full_text: str,
        ground_truth: str,
        offset_mapping: list[tuple[int, int]]
    ) -> list[int]:
        """Locate token indices corresponding to ground truth in the full text."""
        # Find ground truth byte positions
        # Look for ground truth between <answer> and </answer>
        pattern = rf'<answer>\s*{re.escape(ground_truth)}\s*</answer>'
        match = re.search(pattern, full_text)

        if not match:
            # Fallback: find ground truth anywhere
            gt_start = full_text.find(ground_truth)
            if gt_start == -1:
                return []
            gt_end = gt_start + len(ground_truth)
        else:
            # More precise: find GT within the match
            answer_content = match.group(0)
            gt_start_in_match = answer_content.find(ground_truth)
            gt_start = match.start() + gt_start_in_match
            gt_end = gt_start + len(ground_truth)

        # Find tokens that overlap with ground truth byte range
        token_indices = []
        for i, (start, end) in enumerate(offset_mapping):
            if start < gt_end and end > gt_start:
                token_indices.append(i)

        return token_indices

    def _compute_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute position IDs from attention mask."""
        position_ids = torch.zeros_like(attention_mask)
        position = 0
        for i in range(len(attention_mask)):
            if attention_mask[i] == 1:
                position_ids[i] = position
                position += 1
        return position_ids

    def _pad_left(self, tensor: torch.Tensor, target_length: int, pad_value: int) -> torch.Tensor:
        """Pad tensor on the left side."""
        if len(tensor) >= target_length:
            return tensor[-target_length:]  # Truncate from left
        pad_size = target_length - len(tensor)
        return F.pad(tensor, (pad_size, 0), value=pad_value)

    def _pad_right(self, tensor: torch.Tensor, target_length: int, pad_value: int) -> torch.Tensor:
        """Pad tensor on the right side."""
        if len(tensor) >= target_length:
            return tensor[:target_length]  # Truncate from right
        pad_size = target_length - len(tensor)
        return F.pad(tensor, (0, pad_size), value=pad_value)

    def _create_zero_outputs(self, seq_len: int) -> tuple:
        """Create zero tensors when format is invalid.

        Note: This returns 1D tensors to match the normal output format.
        """
        input_ids_pr = torch.zeros(self.max_prompt_length + self.max_response_length, dtype=torch.long)
        attention_mask_pr = torch.zeros_like(input_ids_pr)
        position_ids_pr = torch.zeros_like(input_ids_pr)
        responses_pr = torch.zeros(self.max_response_length, dtype=torch.long)
        ground_truth_mask_pr = torch.zeros_like(responses_pr)

        return input_ids_pr, attention_mask_pr, position_ids_pr, responses_pr, ground_truth_mask_pr

    def batch_replace_answer_with_ground_truth(
        self,
        batch_input_ids: torch.Tensor,
        batch_prompts: torch.Tensor,
        batch_responses: torch.Tensor,
        ground_truths: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        Batch version of replace_answer_with_ground_truth.

        Args:
            batch_input_ids: (batch_size, seq_len)
            batch_prompts: (batch_size, prompt_len)
            batch_responses: (batch_size, response_len)
            ground_truths: List of ground truth strings

        Returns:
            Dictionary with keys: input_ids_pr, attention_mask_pr, position_ids_pr,
                                 responses_pr, ground_truth_mask_pr
        """
        batch_size = len(batch_input_ids)
        results = {
            'input_ids_pr': [],
            'attention_mask_pr': [],
            'position_ids_pr': [],
            'responses_pr': [],
            'ground_truth_mask_pr': [],
        }

        for i in range(batch_size):
            input_ids_pr, attention_mask_pr, position_ids_pr, responses_pr, gt_mask_pr = \
                self.replace_answer_with_ground_truth(
                    input_ids=batch_input_ids[i],
                    prompts=batch_prompts[i],
                    responses=batch_responses[i],
                    ground_truth=ground_truths[i],
                )

            results['input_ids_pr'].append(input_ids_pr)
            results['attention_mask_pr'].append(attention_mask_pr)
            results['position_ids_pr'].append(position_ids_pr)
            results['responses_pr'].append(responses_pr)
            results['ground_truth_mask_pr'].append(gt_mask_pr)

        # Stack all results
        for key in results:
            results[key] = torch.stack(results[key])

        return results
