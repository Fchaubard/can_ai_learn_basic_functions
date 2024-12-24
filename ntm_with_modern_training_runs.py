#!/usr/bin/env python3
"""
Training script to train a Neural Turing Machine (NTM) on classic tasks w/ multiple "modern" optimization techniques (Adam, Cosine LR, Warmup, MeZO, etc.) to see if anything has changed.
May need to: pip install nvidia-ml-py3

Usage Example:
    python ntm_with_modern_training_runs.py \
        --task copy \
        --batch_size 16 \
        --max_iters 50000 \
        --hidden_size 128 \
        --memory_size 128 \
        --head_size 64 \
        --num_heads 1 \
        --optimizer adam \
        --learning_rate 1e-3 \
        --cosine_lr  \
        --warmup_steps 1000 \
        --mezo \
        --wandb_proj "NTM-Experiments" \
        --wandb_run_name "test-run"

Author: Francois Chaubard
"""

import os
os.environ["WANDB_API_KEY"] = ""
import sys
import math
import wandb
import random
import argparse
import numpy as np
from typing import Tuple, List
import pynvml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random

def debug_print_samples(
    x: torch.Tensor, 
    y: torch.Tensor, 
    preds: torch.Tensor, 
    iteration: int, 
    tag: str = "Train", 
    n: int = 3
):
    """
    Print a random subset of examples from the batch:
      - x: [batch_size, seq_len, input_dim]
      - y: [batch_size, seq_len, output_dim]
      - preds: [batch_size, seq_len, output_dim]
      - iteration: current iteration number
      - tag: 'Train' or 'Val' to label
      - n: how many samples to print

    We'll show a short portion of each sequence for readability, or entire if short.
    """
    # Make sure inputs are on CPU for printing
    x_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu()
    p_cpu = preds.detach().cpu()
    
    batch_size = x_cpu.size(0)
    seq_len_in = x_cpu.size(1)
    seq_len_out = y_cpu.size(1)

    print(f"\n[DEBUG] {tag} Samples at iteration {iteration}:")
    
    # Choose 'n' random indices from the batch
    indices = random.sample(range(batch_size), min(n, batch_size))
    for idx in indices:
        # Convert to Python lists (just for printing clarity)
        input_seq = x_cpu[idx].tolist()
        target_seq = y_cpu[idx].tolist()
        pred_seq = p_cpu[idx].tolist()

        # For tasks like Copy/Repeat/Associative Recall, seq_len_in and seq_len_out might differ
        # We'll unify them if we want to print them side by side
        max_seq_len = max(seq_len_in, seq_len_out)
        
        # Print just the first few timesteps if it’s too long
        T_print = min(max_seq_len, 10)  # limit to 10 for brevity
        print(f"  Sample idx={idx} (showing up to first {T_print} timesteps):")
        for t in range(T_print):
            # Safely index
            input_t = input_seq[t] if t < seq_len_in else "[no-input]"
            target_t = target_seq[t] if t < seq_len_out else "[no-target]"
            pred_t   = pred_seq[t] if t < seq_len_out else "[no-pred]"

            print(f"    t={t} | x={input_t} | y={target_t} | pred={pred_t}")
    print("[END DEBUG]\n")


def compute_batch_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes bitwise accuracy for a batch, assuming binary classification (0 or 1) on each bit.
      - logits: [batch_size, seq_len, bits], raw outputs
      - targets: [batch_size, seq_len, bits], 0/1 ground truth
    """
    with torch.no_grad():
        preds = torch.sigmoid(logits) > 0.5   # boolean tensor
        correct = (preds == (targets > 0.5))  # elementwise comparison
        return correct.float().mean().item()  # average over entire batch & all bits

def compute_weight_decay_term(model, weight_decay: float) -> float:
    """
    If you're applying weight decay in optimizer, you might want to log
    the L2 penalty term = 0.5 * wd * sum of squares of all parameters.
    (This is purely for logging; the actual WD is applied in the optimizer step.)
    """
    if weight_decay <= 0.0:
        return 0.0
    sum_of_squares = 0.0
    with torch.no_grad():
        for p in model.parameters():
            sum_of_squares += p.pow(2).sum().item()
    return 0.5 * weight_decay * sum_of_squares


def pick_gpu_with_most_free_mem() -> int:
    """
    Queries all available GPUs and returns the index of the one
    with the most free memory.
    """
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    best_index = 0
    best_free_mem = 0
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if mem_info.free > best_free_mem:
            best_free_mem = mem_info.free
            best_index = i

    pynvml.nvmlShutdown()
    return best_index


##############################################################################
# Data Generation for Classical NTM Tasks
##############################################################################
def generate_copy_task(batch_size: int, seq_len: int, bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The copy task:
      - Input is a random sequence of bits plus an end-of-sequence marker.
      - Model should output the same sequence after the marker.
    Returns:
      x: [batch_size, 2*seq_len + 1, bits+1]  (the input, incl. EOS marker)
      y: [batch_size, 2*seq_len + 1, bits]    (the target)
    """
    # 1) Generate random bit sequences [batch_size, seq_len, bits].
    seq = torch.randint(0, 2, (batch_size, seq_len, bits), dtype=torch.float32)
    
    # 2) Convert to [batch_size, seq_len, bits+1] by appending a zero column for the marker slot.
    zero_col = torch.zeros(batch_size, seq_len, 1, dtype=torch.float32)
    seq_input = torch.cat([seq, zero_col], dim=-1)  # now shape is [batch_size, seq_len, bits+1]

    # 3) Create an EOS marker of shape [batch_size, 1, bits+1], where the last entry is 1.
    eos = torch.zeros(batch_size, 1, bits + 1, dtype=torch.float32)
    eos[..., -1] = 1.0

    # 4) Concatenate the sequence and the EOS marker along the time dimension => [batch_size, seq_len+1, bits+1]
    seq_with_eos = torch.cat([seq_input, eos], dim=1)

    # 5) For the output phase, we add an extra seq_len "time steps" of zeros to the input:
    pad_input = torch.zeros(batch_size, seq_len, bits + 1)
    x = torch.cat([seq_with_eos, pad_input], dim=1)
    # x shape: [batch_size, 2*seq_len + 1, bits+1]

    # 6) Build the target:
    #    - During the first seq_len+1 timesteps (input + EOS), the target is zero (or we don't penalize).
    #    - After the EOS, we want the model to reproduce the original seq bits.
    # So we create zero output for seq_len+1 steps, then the seq bits for the next seq_len steps.
    pad_out = torch.zeros(batch_size, seq_len + 1, bits)
    y_copy = seq  # shape [batch_size, seq_len, bits]
    y = torch.cat([pad_out, y_copy], dim=1)
    # y shape: [batch_size, 2*seq_len + 1, bits]

    return x, y



def generate_repeat_copy_task(
    batch_size: int,
    seq_len: int,
    bits: int = 8,
    repeat_min: int = 1,
    repeat_max: int = 3
):
    """
    The repeat-copy task:
      - We feed a sequence of bits, followed by a row encoding the repeat count,
        then a marker row, then a period for output generation.
      - The model must output the original sequence repeated 'count' times.

    Requirements:
      - Input dimension must be (bits + 2).
      - Output dimension is 'bits'.
    """
    # 1) Generate random bit sequences: shape [B, seq_len, bits]
    seq = torch.randint(0, 2, (batch_size, seq_len, bits), dtype=torch.float32)

    # 2) Convert to shape [B, seq_len, bits+2] by adding two extra columns for
    #    repeat-count/marker usage:
    extra_cols = torch.zeros(batch_size, seq_len, 2, dtype=torch.float32)
    seq_input = torch.cat([seq, extra_cols], dim=-1)  # [B, seq_len, bits+2]

    # 3) Sample repeat counts
    repeat_counts = torch.randint(repeat_min, repeat_max + 1, (batch_size,))

    # 4) Create a single row for the repeat count (second-last column)
    #    and a single row for the marker (last column).
    #    count_row: [B, 1, bits+2]
    count_row = torch.zeros(batch_size, 1, bits + 2, dtype=torch.float32)
    for i in range(batch_size):
        c = float(repeat_counts[i].item())
        count_row[i, 0, -2] = c  # store the count in the second-last column

    #    marker_row: [B, 1, bits+2], where the last column is 1
    marker_row = torch.zeros(batch_size, 1, bits + 2, dtype=torch.float32)
    marker_row[..., -1] = 1.0

    # 5) Concatenate sequence + count row + marker row
    seq_with_count_marker = torch.cat([seq_input, count_row, marker_row], dim=1)
    # shape: [B, seq_len + 2, bits+2]

    # 6) Pad the input for the "output phase".
    #    We assume the maximum repeated output can be seq_len * repeat_max.
    pad_input = torch.zeros(batch_size, seq_len * repeat_max, bits + 2, dtype=torch.float32)

    # 7) Final input: [B, (seq_len+2) + seq_len*repeat_max, bits+2]
    x = torch.cat([seq_with_count_marker, pad_input], dim=1)

    # ------------------------------
    # Construct the Target (outputs)
    # ------------------------------

    # 8) We want the first (seq_len+2) steps to produce zeros (or be ignored in loss),
    #    then the repeated sequence. We'll pad to length = seq_len * repeat_max.
    total_out_len = (seq_len + 2) + (seq_len * repeat_max)
    zero_out = torch.zeros(batch_size, seq_len + 2, bits, dtype=torch.float32)

    # 9) Build the repeated sequence for each sample, then pad it to seq_len*repeat_max
    repeated_outs = []
    for i in range(batch_size):
        c = repeat_counts[i].item()
        repeated_seq = seq[i].repeat((int(c), 1))  # shape: [seq_len * c, bits]
        # pad to seq_len * repeat_max
        pad_len = seq_len * repeat_max - repeated_seq.size(0)
        out_pad = torch.zeros(pad_len, bits, dtype=torch.float32)
        repeated_padded = torch.cat([repeated_seq, out_pad], dim=0)  # [seq_len*repeat_max, bits]
        repeated_outs.append(repeated_padded)

    out_rep = torch.stack(repeated_outs, dim=0)  # [B, seq_len*repeat_max, bits]

    # 10) Concatenate zeros (for the input & marker phase) + repeated portion
    y = torch.cat([zero_out, out_rep], dim=1)  # [B, total_out_len, bits]

    return x, y


def generate_associative_recall_task(
    batch_size: int,
    item_len: int = 3,
    num_items: int = 3,
    bits: int = 8
):
    """
    The associative-recall task (simplified):
      - We have `num_items` items, each of length `item_len` bits.
      - We present all items (flattened), then a marker row, then a 'query' item.
      - The model must output the item that follows the query in the original list.

    Requirements:
      - Input dimension must be (bits + 1).
      - Output dimension is 'bits'.
    """
    # 1) Generate random items: [B, num_items, item_len, bits]
    items = torch.randint(0, 2, (batch_size, num_items, item_len, bits), dtype=torch.float32)

    # 2) Randomly choose an index in [0, num_items-2], so there's always a "next" item
    query_indices = torch.randint(0, num_items - 1, (batch_size,))

    # 3) Gather the queries and their "next" (answers)
    queries = []
    answers = []
    for i in range(batch_size):
        q_idx = query_indices[i].item()
        queries.append(items[i, q_idx])       # shape [item_len, bits]
        answers.append(items[i, q_idx + 1])   # shape [item_len, bits]

    queries = torch.stack(queries, dim=0)   # [B, item_len, bits]
    answers = torch.stack(answers, dim=0)   # [B, item_len, bits]

    # 4) Flatten the full set of items: [B, num_items * item_len, bits]
    flattened = items.view(batch_size, num_items * item_len, bits)

    # 5) Convert flattened items to shape [B, num_items*item_len, bits+1] by adding
    #    one extra column for the marker dimension (initialized to zero).
    extra_col = torch.zeros(batch_size, num_items * item_len, 1, dtype=torch.float32)
    flattened_in = torch.cat([flattened, extra_col], dim=-1)  # [B, num_items*item_len, bits+1]

    # 6) Create a marker row: [B, 1, bits+1]
    marker_row = torch.zeros(batch_size, 1, bits + 1, dtype=torch.float32)
    marker_row[..., -1] = 1.0

    # 7) Convert the query to shape [B, item_len, bits+1] (add zero col)
    zero_col_query = torch.zeros(batch_size, item_len, 1, dtype=torch.float32)
    query_in = torch.cat([queries, zero_col_query], dim=-1)  # [B, item_len, bits+1]

    # 8) Final input x = [flattened items, marker row, query]
    #    shape => [B, (num_items*item_len + 1 + item_len), bits+1]
    x = torch.cat([flattened_in, marker_row, query_in], dim=1)

    # ------------------------------
    # Construct the Target (outputs)
    # ------------------------------

    # 9) The target is the item after the query => [B, item_len, bits].
    #    We'll pad it to match the same time dimension as x.
    T_in = x.size(1)  # total input length
    item_out_len = answers.size(1)  # usually item_len
    pad_len = T_in - item_out_len

    pad_zeros = torch.zeros(batch_size, pad_len, bits, dtype=torch.float32)
    y = torch.cat([answers, pad_zeros], dim=1)  # [B, T_in, bits]

    return x, y

##############################################################################
# NTM Model Definition
##############################################################################
class NTM(nn.Module):
    """
    A simple Neural Turing Machine architecture:
      - RNN controller (LSTM)
      - Memory with read/write heads
      - For demonstration purposes, this is simplified compared to the original
        Alex Graves version.
    """
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        hidden_size: int, 
        memory_size: int, 
        head_size: int, 
        num_heads: int,
    ):
        super(NTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.memory_size = memory_size
        self.head_size = head_size
        
        # Controller
        self.controller = nn.LSTM(input_size + num_heads * head_size, hidden_size, batch_first=True)
        
        # Heads (read/write) parameters for each head
        # We'll just do a single set of parameters for read and write, for simplicity.
        self.fc_head = nn.Linear(hidden_size, num_heads * (head_size + memory_size + 3))
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size + num_heads * head_size, output_size)
        
        # Initialize memory as a buffer
        # Memory: [batch_size, memory_size, head_size] (we'll create it dynamically)
    
    def _addressing(self, memory, head_params, prev_weights):
        """
        For demonstration: a simplistic addressing mechanism.
        head_params: [batch_size, num_heads, head_size + memory_size + 3]
          includes read_key, write_key, shift weighting, etc. (all simplified)
        """
        batch_size, mem_size, _ = memory.size()
        # We'll do content-based addressing by matching the "key" with memory.
        # Then optionally shift or focus. This is extremely simplified.
        # Let's say:
        #   read_key = head_params[:, :, :head_size]
        #   write_key = head_params[:, :, head_size:2*head_size]
        # We won't perfectly replicate real addressing logic, just a rough approach.
        
        # For demonstration, let's do a naive approach:
        # We'll interpret the first part as a "key" vector, compute dot products with memory,
        # and do a softmax.
        
        # Suppose read_key = head_params[:, :, :head_size]
        read_keys = head_params[:, :, :self.head_size]
        
        # Dot product with memory
        # memory: [batch_size, mem_size, head_size]
        # read_keys: [batch_size, num_heads, head_size]
        # => we broadcast to [batch_size, num_heads, mem_size]
        read_weights = torch.einsum('bnk,bm k->bnm', read_keys, memory)
        read_weights = read_weights / (self.head_size ** 0.5)
        read_weights = F.softmax(read_weights, dim=-1)  # [batch_size, num_heads, mem_size]
        
        # We'll do a single read head for demonstration (or sum read across heads).
        read_content = torch.einsum('bnm,bmh->bnh', read_weights, memory)  # [b, num_heads, head_size]
        
        # Write: we'll skip the full erase/add mechanics for brevity and just do a naive update
        write_keys = head_params[:, :, self.head_size: 2 * self.head_size]
        write_strength = torch.sigmoid(head_params[:, :, 2*self.head_size:2*self.head_size+1])  # [b, num_heads, 1]
        
        # Dot product for write
        write_weights = torch.einsum('bnk,bmk->bnm', write_keys, memory) / (self.head_size ** 0.5)
        write_weights = F.softmax(write_weights, dim=-1)
        # We'll do a simple additive write
        # next_memory = memory + alpha * (outer(write_weights, write_keys))
        # This is extremely naive; a real NTM uses erase and add vectors.
        write_content = write_keys  # re-using the key as content
        write_content = write_strength * write_content  # scale by strength
        # broadcast: write_weights: [b, heads, mem_size], write_content: [b, heads, head_size]
        # we want to shape to [b, mem_size, head_size]
        # We'll sum across heads
        delta = torch.einsum('bnm,bnh->bmh', write_weights, write_content)
        memory = memory + delta
        
        return memory, read_content, read_weights, write_weights
    
    def forward(self, x, hidden=None, memory=None):
        """
        x: [batch_size, seq_len, input_size]
        memory: [batch_size, memory_size, head_size]
        hidden: (h, c) for the LSTM
        """
        batch_size, seq_len, input_size = x.size()
        
        if hidden is None:
            h0 = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
            hidden = (h0, c0)
        
        if memory is None:
            memory = torch.zeros(batch_size, self.memory_size, self.head_size, device=x.device)
        
        outputs = []
        read_contents = torch.zeros(batch_size, self.num_heads, self.head_size, device=x.device)
        
        for t in range(seq_len):
            # Combine input with read vectors
            inp = torch.cat([x[:, t, :], read_contents.view(batch_size, -1)], dim=-1).unsqueeze(1)
            out_ctrl, hidden = self.controller(inp, hidden)
            h = out_ctrl.squeeze(1)  # [batch_size, hidden_size]
            
            # Head parameters
            head_params = self.fc_head(h)  # [batch_size, num_heads*(head_size+memory_size+3)]
            head_params = head_params.view(batch_size, self.num_heads, (self.head_size + self.memory_size + 3))
            
            # Addressing
            memory, read_contents, _, _ = self._addressing(memory, head_params, None)
            
            # Output
            out = torch.cat([h, read_contents.view(batch_size, -1)], dim=-1)
            out = self.fc_out(out)  # [batch_size, output_size]
            outputs.append(out.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)  # [batch_size, seq_len, output_size]
        return outputs, memory, hidden


##############################################################################
# Training Loop
##############################################################################
def train_step(model, x, y, criterion, optimizer):
    """
    A standard training step using backprop (e.g., Adam).
    """
    model.train()
    optimizer.zero_grad()
    outputs, _, _ = model(x)
    # We want to compute loss only up to the length of y
    # x.size(1) and y.size(1) might differ, so let's align them
    seq_len = min(outputs.size(1), y.size(1))
    loss = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])
    loss.backward()
    optimizer.step()
    return loss.item()


# def mezo_step(model, x, y, criterion, epsilon=1e-3, layerwise=False, lr = 1e-3):
#     """
#     Memory-Efficient Zero-Order (MeZO) demonstration:
#     ...
#     Returns a PyTorch tensor (so main loop can do .item()).
#     """
#     model.train()
#     all_params = list(model.parameters())

#     if layerwise:
#         total_loss = 0.0
#         for param in all_params:
#             if param.requires_grad:
#                 original_data = param.data.clone()

#                 # +epsilon
#                 param.data = original_data + epsilon
#                 outputs, _, _ = model(x)
#                 seq_len = min(outputs.size(1), y.size(1))
#                 loss_plus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])

#                 # -2 * epsilon
#                 param.data = original_data - 2 * epsilon
#                 outputs, _, _ = model(x)
#                 loss_minus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])

#                 grad_est = (loss_plus - loss_minus) / (2 * epsilon)

#                 # restore
#                 param.data = original_data

#                 # manual update
                
#                 param.data = param.data - lr * grad_est * epsilon * torch.sign(torch.randn_like(param.data))

#                 total_loss += ((loss_plus + loss_minus) / 2.0).item()
#         avg_loss = total_loss / len(all_params)
#         return torch.tensor(avg_loss, device=x.device)

#     else:
#         original_data = [p.data.clone() for p in all_params]
#         directions = [torch.randn_like(p.data) for p in all_params]

#         # +epsilon
#         for p, d in zip(all_params, directions):
#             p.data = p.data + epsilon * d.sign()

#         outputs, _, _ = model(x)
#         seq_len = min(outputs.size(1), y.size(1))
#         loss_plus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])

#         # -2 epsilon
#         for p, d in zip(all_params, directions):
#             p.data = p.data - 2 * epsilon * d.sign()

#         outputs, _, _ = model(x)
#         loss_minus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])

#         # restore
#         for p, orig_data in zip(all_params, original_data):
#             p.data = orig_data

#         grad_est = (loss_plus - loss_minus) / (2 * epsilon)
        
#         for p, d in zip(all_params, directions):
#             p.data = p.data - lr * grad_est.item() * epsilon * d.sign()

#         avg_loss = 0.5 * (loss_plus.item() + loss_minus.item())
#         return torch.tensor(avg_loss, device=x.device)


def mezo_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: torch.nn.Module,
    epsilon: float = 1e-3,
    layerwise: bool = False,
    weight_decay: float = 1e-5
) -> torch.Tensor:
    """
    Memory-Efficient Zero-Order (MeZO) demonstration with optional weight decay.
    
    Args:
      model: The NTM (or any) model to update.
      x: Input batch tensor.
      y: Target batch tensor.
      criterion: Loss function (e.g., BCEWithLogitsLoss).
      epsilon: Perturbation magnitude for finite difference.
      layerwise: If True, do layer-by-layer (param-by-param) perturbation.
                 If False, do a single big perturbation for the entire model.
      weight_decay: L2 regularization coefficient. 
                    If > 0, adds 0.5 * weight_decay * sum(param^2) to the loss.
    
    Returns:
      A PyTorch scalar tensor representing the average loss from the plus/minus passes.
    """
    model.train()
    all_params = list(model.parameters())

    def compute_weight_decay_loss(model, wd):
        """
        Compute 0.5 * wd * sum of squares of all parameters that require grad.
        Returns a float or a torch scalar. We'll do float + manual add to the final
        so we keep the data consistent with the final computations.
        """
        if wd == 0.0:
            return 0.0
        sum_of_squares = 0.0
        for p in model.parameters():
            if p.requires_grad:
                sum_of_squares += p.data.pow(2).sum().item()
        return 0.5 * wd * sum_of_squares

    if layerwise:
        # Layerwise approach: For each parameter, do +epsilon pass, -2*epsilon pass, etc.
        total_loss = 0.0
        count_params = 0
        for param in all_params:
            if not param.requires_grad:
                continue
            count_params += 1

            original_data = param.data.clone()

            # + epsilon
            param.data = original_data + epsilon
            outputs, _, _ = model(x)
            seq_len = min(outputs.size(1), y.size(1))
            loss_plus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])
            # Add weight decay penalty
            loss_plus = loss_plus + compute_weight_decay_loss(model, weight_decay)

            # -2 * epsilon
            param.data = original_data - 2 * epsilon
            outputs, _, _ = model(x)
            loss_minus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])
            # Add weight decay penalty
            loss_minus = loss_minus + compute_weight_decay_loss(model, weight_decay)

            # Finite difference
            grad_est = (loss_plus - loss_minus) / (2 * epsilon)

            # Restore original
            param.data = original_data

            # Manual update: For demonstration, we pick a small LR
            lr = 1e-3
            # We multiply by 'grad_est' (a scalar), then epsilon, etc.
            # This is a toy approach to do "coordinate-free" direction updates.
            with torch.no_grad():
                # param update
                param.data -= lr * grad_est * epsilon * torch.sign(torch.randn_like(param.data))

            # Accumulate average loss for logging
            # We'll take the mean of (loss_plus + loss_minus) / 2 to approximate the loss
            avg_pass_loss = 0.5 * (loss_plus.item() + loss_minus.item())
            total_loss += avg_pass_loss

        # Average over number of parameters
        if count_params > 0:
            total_loss /= float(count_params)
        # Return as a Torch tensor so we can call .item() outside
        return torch.tensor(total_loss, device=x.device)

    else:
        # Single-perturbation approach
        original_data = [p.data.clone() for p in all_params]
        directions = [torch.randn_like(p.data) if p.requires_grad else None for p in all_params]

        # + epsilon
        for p, d in zip(all_params, directions):
            if p.requires_grad and d is not None:
                p.data = p.data + epsilon * d.sign()

        outputs, _, _ = model(x)
        seq_len = min(outputs.size(1), y.size(1))
        loss_plus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])
        loss_plus = loss_plus + compute_weight_decay_loss(model, weight_decay)

        # -2 epsilon
        for p, d in zip(all_params, directions):
            if p.requires_grad and d is not None:
                p.data = p.data - 2 * epsilon * d.sign()

        outputs, _, _ = model(x)
        loss_minus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])
        loss_minus = loss_minus + compute_weight_decay_loss(model, weight_decay)

        # Restore original data
        for p, orig_data in zip(all_params, original_data):
            p.data = orig_data

        # Finite difference gradient estimate
        grad_est = (loss_plus - loss_minus) / (2 * epsilon)

        # Manual update
        lr = 1e-3
        with torch.no_grad():
            for p, d in zip(all_params, directions):
                if p.requires_grad and d is not None:
                    # grad_est is a scalar
                    p.data = p.data - lr * grad_est.item() * epsilon * d.sign()

        # Return average of plus/minus losses as a scalar
        avg_loss = 0.5 * (loss_plus.item() + loss_minus.item())
        return torch.tensor(avg_loss, device=x.device)



##############################################################################
# Main Training/CLI
##############################################################################
def main():
    parser = argparse.ArgumentParser(description="Train an NTM on classical tasks.")
    
    # Task
    parser.add_argument("--task", type=str, default="copy", 
                        choices=["copy", "repeat_copy", "associative_recall"],
                        help="Which task to train on.")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length (task-dependent).")
    
    # Model
    parser.add_argument("--input_size", type=int, default=9, help="Input size (bits+1 for tasks).")
    parser.add_argument("--output_size", type=int, default=8, help="Output size (bits).")
    parser.add_argument("--hidden_size", type=int, default=128, help="Controller hidden size.")
    parser.add_argument("--memory_size", type=int, default=128, help="Number of memory slots.")
    parser.add_argument("--head_size", type=int, default=64, help="Dimension of each head vector.")
    parser.add_argument("--num_heads", type=int, default=1, help="Number of heads.")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "mezo"],
                        help="Which optimizer to use.")
    
    # Cosine LR & Warmup
    parser.add_argument("--cosine_lr", action="store_true", help="Use cosine learning rate scheduling.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps if using warmup.")
    
    # MeZO
    parser.add_argument("--mezo", action="store_true", help="Use MeZO updates (instead of standard grad).")
    parser.add_argument("--mezo_layerwise", action="store_true", help="Apply MeZO updates layer by layer.")
    
    # Misc
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval.")
    parser.add_argument("--wandb_proj", type=str, default=None, help="Weights & Biases project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name.")
    args = parser.parse_args()
    init_weight_decay = 1e-5
    
    # Device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"[INFO] Using device: {device}")
    
    # If you prefer to allow CPU usage when no GPUs are available, do a check:
    if torch.cuda.is_available():
        gpu_index = pick_gpu_with_most_free_mem()  
        device = torch.device(f"cuda:{gpu_index}")
        print(f"[INFO] Using GPU: {gpu_index}")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")

    model = NTM(
        input_size=args.input_size,
        output_size=args.output_size,
        hidden_size=args.hidden_size,
        memory_size=args.memory_size,
        head_size=args.head_size,
        num_heads=args.num_heads
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    max_iters = 1000000000
    # Create optimizer
    if args.optimizer == "adam":
        # Example: if you want to see weight_decay in logs, set it here
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=init_weight_decay)
    else:
        optimizer = None  # We'll do mezo manually

    # Learning rate scheduler (cosine)
    if args.cosine_lr and args.optimizer == "adam":
        # Important: T_max should match how many times you step the scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_iters,
            eta_min=1e-6
        )
    else:
        scheduler = None

    # Initialize W&B if desired
    if args.wandb_proj is not None:
        wandb.init(project=args.wandb_proj, name=args.wandb_run_name)
        wandb.config.update(args)

    global_step = 0

    for iteration in range(1, max_iters + 1):
        # Generate a training batch
        if args.task == "copy":
            x, y = generate_copy_task(args.batch_size, args.seq_len, bits=args.output_size)
        elif args.task == "repeat_copy":
            x, y = generate_repeat_copy_task(args.batch_size, args.seq_len, bits=args.output_size)
        else:
            # default: associative recall
            x, y = generate_associative_recall_task(args.batch_size, item_len=3, num_items=args.seq_len, bits=args.output_size)

        x, y = x.to(device), y.to(device)

        # Optional warmup
        if args.warmup_steps > 0 and iteration <= args.warmup_steps and args.optimizer == "adam":
            warmup_frac = iteration / float(args.warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = args.learning_rate * warmup_frac

        # ===================
        #     TRAIN STEP
        # ===================
        if args.optimizer == "adam":
            # Typical backprop
            model.train()
            optimizer.zero_grad()
            outputs, _, _ = model(x)
            seq_len = min(outputs.size(1), y.size(1))
            loss = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])
            loss.backward()
            optimizer.step()
        else:
            # MeZO or other custom
            outputs, _, _ = model(x)
            loss = mezo_step(model, 
                             x, 
                             y, 
                             criterion, 
                             layerwise=args.mezo_layerwise, 
                             lr=args.learning_rate,
                             weight_decay=init_weight_decay)

        # Step scheduler if using one
        if scheduler is not None and iteration > args.warmup_steps:
            scheduler.step()

        # =====================
        #    LOGGING BLOCK
        # =====================
        global_step += 1

        # Log every N iterations
        if iteration % args.log_interval == 0:
            # Compute current LR
            if scheduler is not None:
                lr_current = scheduler.get_last_lr()[0]  # for multi-param groups, pick group[0]
            elif optimizer is not None and len(optimizer.param_groups) > 0:
                lr_current = optimizer.param_groups[0]["lr"]
            else:
                lr_current = 0.0

            # Compute training accuracy on this batch
            seq_len = min(outputs.size(1), y.size(1))
            train_acc = compute_batch_accuracy(outputs[:, :seq_len, :], y[:, :seq_len, :])
            # Debug print for TRAIN
            print("TRAIN:")
            debug_print_samples(
                x,                 # the train input
                y,                 # the train target
                outputs,           # the train model outputs
                iteration,
                tag="Train",
                n=3
            )
            # Create a small validation batch
            # (If you want a *real* separate dataset, adjust as needed)
            print("VAL:")
            with torch.no_grad():
                if args.task == "copy":
                    val_x, val_y = generate_copy_task(args.batch_size, args.seq_len, bits=args.output_size)
                elif args.task == "repeat_copy":
                    val_x, val_y = generate_repeat_copy_task(args.batch_size, args.seq_len, bits=args.output_size)
                else:
                    val_x, val_y = generate_associative_recall_task(
                        args.batch_size, item_len=3, num_items=args.seq_len, bits=args.output_size
                    )
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_outputs, _, _ = model(val_x)
                val_seq_len = min(val_outputs.size(1), val_y.size(1))
                val_loss = criterion(val_outputs[:, :val_seq_len, :], val_y[:, :val_seq_len, :])
                val_acc = compute_batch_accuracy(val_outputs[:, :val_seq_len, :], val_y[:, :val_seq_len, :])
                debug_print_samples(
                    val_x,
                    val_y,
                    val_outputs,
                    iteration,
                    tag="Val",
                    n=3
                )

            # Weight decay term (if used)
            if optimizer is not None and "weight_decay" in optimizer.param_groups[0]:
                wd = optimizer.param_groups[0]["weight_decay"]
            else:
                wd = 0.0

            wd_term = compute_weight_decay_term(model, wd)

            # Print to stdout
            msg = (f"Iter: {iteration}, "
                   f"Train Loss: {loss}, "
                   f"Train Acc: {train_acc:.3f}, "
                   f"Val Loss: {val_loss}, "
                   f"Val Acc: {val_acc:.3f}, "
                   f"WD term: {wd_term:.6f}, "
                   f"LR: {lr_current:.8f}")
            print(msg)
            sys.stdout.flush()

            # Log to W&B
            if args.wandb_proj is not None:
                wandb.log({
                    "train_loss": loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "weight_decay_term": wd_term,
                    "lr": lr_current
                }, step=global_step)


if __name__ == "__main__":
    main()
    print("finished training")
