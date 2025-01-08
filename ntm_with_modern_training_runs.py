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
os.environ["WANDB_API_KEY"] = "cce47709d839921f0b13533529f31c8af7f3f4dc"
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
# Data Generation for Tasks
##############################################################################

def generate_copy_task(
    batch_size: int, 
    seq_len: int, 
    bits: int = 8, 
    train: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The copy task:
      - If train=True, we use seq_len as is.
      - If train=False, we use seq_len * 5 to validate on a longer sequence.

    x: [batch_size, 2*seq_len + 1, bits+1]
    y: [batch_size, 2*seq_len + 1, bits]
    """
    if not train:
        seq_len = seq_len * 5  # 5x longer sequences for validation

    # 1) Generate random bit sequences [batch_size, seq_len, bits].
    seq = torch.randint(0, 2, (batch_size, seq_len, bits), dtype=torch.float32)
    
    # 2) Convert to [batch_size, seq_len, bits+1] by appending a zero column for the marker slot.
    zero_col = torch.zeros(batch_size, seq_len, 1, dtype=torch.float32)
    seq_input = torch.cat([seq, zero_col], dim=-1)  # shape: [B, seq_len, bits+1]

    # 3) Create an EOS marker of shape [B, 1, bits+1], last entry is 1.
    eos = torch.zeros(batch_size, 1, bits + 1, dtype=torch.float32)
    eos[..., -1] = 1.0

    # 4) Concatenate sequence and the EOS marker => [B, seq_len+1, bits+1]
    seq_with_eos = torch.cat([seq_input, eos], dim=1)

    # 5) For the output phase, add extra seq_len "time steps" of zeros
    pad_input = torch.zeros(batch_size, seq_len, bits + 1)
    x = torch.cat([seq_with_eos, pad_input], dim=1)  # [B, 2*seq_len + 1, bits+1]

    # 6) Build the target: zero for first seq_len+1 steps, then replicate original seq bits
    pad_out = torch.zeros(batch_size, seq_len + 1, bits)
    y_copy = seq
    y = torch.cat([pad_out, y_copy], dim=1)  # [B, 2*seq_len + 1, bits]

    return x, y


def generate_repeat_copy_task(
    batch_size: int,
    seq_len: int,
    bits: int = 8,
    repeat_min: int = 1,
    repeat_max: int = 3,
    train: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The repeat-copy task, input dimension = bits+1.

    - If train=False, we do seq_len*5 for validation.
    - We present a sequence [seq_len, bits], plus a row storing the repeat count,
      plus a marker row, then pad zeros for the output phase.

    Returns:
      x: [B, (seq_len+2) + (seq_len*repeat_max), bits+1]
      y: [B, (seq_len+2) + (seq_len*repeat_max), bits]
    """
    if not train:
        seq_len = seq_len * 5

    seq = torch.randint(0, 2, (batch_size, seq_len, bits), dtype=torch.float32)
    zero_col = torch.zeros(batch_size, seq_len, 1, dtype=torch.float32)
    seq_input = torch.cat([seq, zero_col], dim=-1)  # [B, seq_len, bits+1]

    repeat_counts = torch.randint(repeat_min, repeat_max + 1, (batch_size,))

    count_row = torch.zeros(batch_size, 1, bits + 1, dtype=torch.float32)
    for i in range(batch_size):
        c = float(repeat_counts[i].item())
        count_row[i, 0, -2] = c

    marker_row = torch.zeros(batch_size, 1, bits + 1, dtype=torch.float32)
    marker_row[..., -1] = 1.0

    seq_with_count_marker = torch.cat([seq_input, count_row, marker_row], dim=1)

    pad_input = torch.zeros(batch_size, seq_len * repeat_max, bits + 1, dtype=torch.float32)
    x = torch.cat([seq_with_count_marker, pad_input], dim=1)

    total_time = (seq_len + 2) + (seq_len * repeat_max)
    zero_target = torch.zeros(batch_size, seq_len + 2, bits, dtype=torch.float32)

    repeated_stacks = []
    for i in range(batch_size):
        c = int(repeat_counts[i].item())
        repeated_seq = seq[i].repeat((c, 1))
        pad_len = seq_len * repeat_max - repeated_seq.size(0)
        out_pad = torch.zeros(pad_len, bits, dtype=torch.float32)
        repeated_out = torch.cat([repeated_seq, out_pad], dim=0)
        repeated_stacks.append(repeated_out)

    repeated_outs = torch.stack(repeated_stacks, dim=0)
    y = torch.cat([zero_target, repeated_outs], dim=1)

    return x, y




def generate_associative_recall_task(
    batch_size: int,
    item_len: int = 3,
    num_items: int = 3,
    bits: int = 8,
    train: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    If train=False, we do num_items *= 5 for validation.

    x: [B, (num_items*item_len + 1 + item_len), bits+1]
    y: [B, same_time, bits]
    """
    if not train:
        num_items = num_items * 5

    items = torch.randint(0, 2, (batch_size, num_items, item_len, bits), dtype=torch.float32)
    query_indices = torch.randint(0, num_items - 1, (batch_size,))

    queries, answers = [], []
    for i in range(batch_size):
        q_idx = query_indices[i].item()
        queries.append(items[i, q_idx])
        answers.append(items[i, q_idx + 1])

    queries = torch.stack(queries, dim=0)
    answers = torch.stack(answers, dim=0)

    flattened = items.view(batch_size, num_items * item_len, bits)
    extra_col = torch.zeros(batch_size, num_items * item_len, 1, dtype=torch.float32)
    flattened_in = torch.cat([flattened, extra_col], dim=-1)

    marker_row = torch.zeros(batch_size, 1, bits + 1, dtype=torch.float32)
    marker_row[..., -1] = 1.0

    zero_col_query = torch.zeros(batch_size, item_len, 1, dtype=torch.float32)
    query_in = torch.cat([queries, zero_col_query], dim=-1)

    x = torch.cat([flattened_in, marker_row, query_in], dim=1)

    T_in = x.size(1)
    out_len = answers.size(1)
    pad_len = T_in - out_len
    pad_zeros = torch.zeros(batch_size, pad_len, bits, dtype=torch.float32)
    y = torch.cat([answers, pad_zeros], dim=1)

    return x, y


def generate_arithmetic_task(
    batch_size: int,
    task_type: str = "add",
    max_num: int = 100,
    train: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a simple arithmetic dataset for tasks = {add, sub, mul, div}.
    Returns x => [B, 1, input_dim], y => [B, 1, output_dim].
    """
    # Example domain shift if !train, etc.
    if train:
        lo, hi = 1, max_num
    else:
        lo, hi = max_num + 1, max_num + 10

    # sample a,b ensuring no negative results for sub,div if desired
    a = torch.randint(lo, hi + 1, (batch_size,))
    b = torch.randint(lo, hi + 1, (batch_size,))

    if task_type in ["sub", "div"]:
        # force a >= b
        for i in range(batch_size):
            a_val = a[i].item()
            if a_val < lo:
                a_val = lo
                a[i] = lo
            b[i] = torch.randint(lo, a_val+1, (1,))

    # compute result
    if task_type == "add":
        res = a + b
        max_res = 2 * hi
    elif task_type == "sub":
        res = a - b
        max_res = hi
    elif task_type == "mul":
        res = a * b
        max_res = hi * hi
    elif task_type == "div":
        res = a // b
        max_res = hi
    else:
        # fallback => add
        res = a + b
        max_res = 2 * hi

    # figure bits_in, bits_out
    bits_in = math.ceil(math.log2(hi + 1)) 
    bits_out = math.ceil(math.log2(max_res + 1))

    x_batch = []
    y_batch = []

    for i in range(batch_size):
        a_val = int(a[i].item())
        b_val = int(b[i].item())
        r_val = int(res[i].item())

        a_bin = f"{a_val:0{bits_in}b}"
        b_bin = f"{b_val:0{bits_in}b}"
        r_bin = f"{r_val:0{bits_out}b}"

        # input => [a_bin + b_bin + marker], length=2*bits_in+1
        x_i_list = [int(c) for c in a_bin] + [int(c) for c in b_bin] + [0]
        x_i = torch.tensor(x_i_list, dtype=torch.float32)  # shape [input_length]

        # output => shape [bits_out]
        y_i = torch.tensor([int(c) for c in r_bin], dtype=torch.float32)

        # Now ensure 3D => [1, 1, input_length] & [1, 1, bits_out]
        # then we can stack to [B, 1, input_length]
        x_i = x_i.unsqueeze(0).unsqueeze(0)  # => [1,1,length]
        y_i = y_i.unsqueeze(0).unsqueeze(0)  # => [1,1,bits_out]

        x_batch.append(x_i)
        y_batch.append(y_i)

    # cat along dim=0 => [B, 1, length], [B, 1, bits_out]
    x = torch.cat(x_batch, dim=0)
    y = torch.cat(y_batch, dim=0)

    return x, y



def generate_fibonacci_task(
    batch_size: int,
    max_n: int = 10,
    train: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    If train=True, n in [1..max_n].
    If train=False, n in [max_n+1..max_n+10].
    x: [B, 1, bits_in+1], y: [B, 1, bits_out]
    """
    if train:
        lo, hi = 1, max_n
    else:
        lo, hi = max_n + 1, max_n + 10

    n_vals = torch.randint(lo, hi + 1, (batch_size,))

    # Precompute fib up to hi+10 in worst case
    max_fib = hi + 10
    fib_cache = [0, 1]
    for i in range(2, max_fib + 1):
        fib_cache.append(fib_cache[-1] + fib_cache[-2])

    # The largest fib we might see is fib_cache[hi+10].
    max_fib_val = fib_cache[hi]

    # bits for input n, bits for fib(n)
    bits_in = math.ceil(math.log2(hi + 1))
    bits_out = math.ceil(math.log2(max_fib_val + 1))

    x_batch, y_batch = [], []

    for i in range(batch_size):
        n_i = n_vals[i].item()
        fib_n = fib_cache[int(n_i)]

        n_bin = f"{int(n_i):0{bits_in}b}"
        fib_bin = f"{int(fib_n):0{bits_out}b}"

        # input => [n_bin + marker], shape => 1 x (bits_in + 1)
        x_i = [int(c) for c in n_bin] + [0]
        x_i = torch.tensor(x_i, dtype=torch.float32).unsqueeze(0)

        # output => shape => 1 x bits_out
        y_i = [int(c) for c in fib_bin]
        y_i = torch.tensor(y_i, dtype=torch.float32).unsqueeze(0)

        x_batch.append(x_i)
        y_batch.append(y_i)

    x = torch.cat(x_batch, dim=0)
    y = torch.cat(y_batch, dim=0)
    return x, y



def generate_factorial_task(
    batch_size: int,
    max_n: int = 6,
    train: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    If train=True, n in [1..max_n].
    If train=False, n in [max_n+1..max_n+10].
    x: [B, 1, bits_in+1], y: [B, 1, bits_out]
    """
    if train:
        lo, hi = 1, max_n
    else:
        lo, hi = max_n + 1, max_n + 10

    n_vals = torch.randint(lo, hi + 1, (batch_size,))

    # Precompute factorial up to hi+10 in worst case
    max_fact = hi + 10
    fact_cache = [1]
    for i in range(1, max_fact + 1):
        fact_cache.append(fact_cache[-1] * i)

    # bits
    bits_in = math.ceil(math.log2(hi + 1))
    # max factorial we might see is fact_cache[hi+10], but let's do fact_cache[hi]
    max_fact_val = fact_cache[hi]
    bits_out = math.ceil(math.log2(max_fact_val + 1))

    x_batch, y_batch = [], []
    for i in range(batch_size):
        n_i = n_vals[i].item()
        fact_n = fact_cache[int(n_i)]

        n_bin = f"{int(n_i):0{bits_in}b}"
        fact_bin = f"{int(fact_n):0{bits_out}b}"

        # input => [n_bin + marker]
        x_i = [int(c) for c in n_bin] + [0]
        x_i = torch.tensor(x_i, dtype=torch.float32).unsqueeze(0)

        y_i = [int(c) for c in fact_bin]
        y_i = torch.tensor(y_i, dtype=torch.float32).unsqueeze(0)

        x_batch.append(x_i)
        y_batch.append(y_i)

    x = torch.cat(x_batch, dim=0)
    y = torch.cat(y_batch, dim=0)
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
# DNC definition
##############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def softplus(x):
    return torch.log1p(torch.exp(x))

class DNC(nn.Module):
    """
    A minimal Differentiable Neural Computer implementation with:
      - Single LSTM Controller
      - Single read head, single write head
      - Usage vector, temporal link, precedence weighting
      - Erase & add operations for writing
      - Weighted read for reading

    Usage Example:
      dnc = DNC(
          input_size=10,
          output_size=8,
          hidden_size=64,
          memory_size=32,   # number of memory slots
          head_size=16,     # dimension of each memory slot
          num_heads=1       # single read head
      )
      outputs, memory_state, hidden = dnc(inputs)
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        memory_size: int,
        head_size: int,
        num_heads: int = 1,
    ):
        super(DNC, self).__init__()
        assert num_heads == 1, "This minimal DNC only supports 1 read head for simplicity."

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size  # N
        self.head_size = head_size      # W
        self.num_reads = num_heads      # =1 for read head

        # 1) LSTM controller: feed (input + read_vector) each timestep
        self.controller = nn.LSTM(
            input_size + (self.num_reads * self.head_size),
            hidden_size,
            batch_first=True
        )

        # 2) Interface vector: erase(W) + write(W) + write_gate(1) + alloc_gate(1)
        #                     + read_key(W) + read_strength(1) + read_mode(3)
        self.interface_size = (
            self.head_size +  # erase
            self.head_size +  # write
            1 +               # write_gate
            1 +               # allocation_gate
            self.head_size +  # read_key
            1 +               # read_strength
            3                 # read mode
        )
        self.fc_interface = nn.Linear(hidden_size, self.interface_size)

        # 3) Output projection
        # final = [controller_output + read_vector]
        self.fc_output = nn.Linear(self.hidden_size + self.num_reads * self.head_size, output_size)

        # 4) Initialize memory buffers
        self.reset_memory()

    def reset_memory(self, batch_size=1, device="cpu"):
        """
        Initialize memory states. Call at the start of a new sequence/batch if needed.
        """
        self.memory = None            # [B, N, W]
        self.usage = None             # [B, N]
        self.precedence = None        # [B, N]
        self.link = None             # [B, N, N]
        self.read_weights = None      # [B, N]
        self.write_weights = None     # [B, N]

    def _init_memory_if_needed(self, batch_size, device):
        """
        If memory is None or batch size changed, create fresh zero memory.
        """
        if self.memory is None or self.memory.size(0) != batch_size:
            self.memory = torch.zeros(batch_size, self.memory_size, self.head_size, device=device)
            self.usage = torch.zeros(batch_size, self.memory_size, device=device)
            self.precedence = torch.zeros(batch_size, self.memory_size, device=device)
            self.link = torch.zeros(batch_size, self.memory_size, self.memory_size, device=device)
            self.read_weights = torch.zeros(batch_size, self.memory_size, device=device)
            self.write_weights = torch.zeros(batch_size, self.memory_size, device=device)

    def _read_from_memory(self, read_weights):
        """
        read_weights: [B, N]
        memory: [B, N, W]
        => read_vector: [B, W]
        """
        return torch.bmm(read_weights.unsqueeze(1), self.memory).squeeze(1)

    def _update_usage(self):
        """
        Simple usage update:
          usage = usage + (1 - usage)*write_w
        ignoring read frees for simplicity.
        """
        w = self.write_weights  # [B, N]
        self.usage = self.usage + (1 - self.usage) * w

    def _get_allocation_weights(self):
        """
        Allocate to least-used slots first. Sort usage ascending, do 'cumulative product'.
        Returns: [B, N] alloc weights
        """
        usage_sorted, idx_sorted = torch.sort(self.usage, dim=-1)  # ascending
        alloc_w = torch.zeros_like(self.usage)
        cprod = torch.cumprod(usage_sorted, dim=-1)
        cprod = F.pad(cprod[:, :-1], (1, 0), value=1.0)  # shift right

        alloc_in_order = (1 - usage_sorted) * cprod
        alloc_w.scatter_(1, idx_sorted, alloc_in_order)
        return alloc_w

    def _update_temporal_link(self, write_w):
        """
        Update link => [B, N, N], precedence => [B, N].
        """
        ww_ij = write_w.unsqueeze(-1) + write_w.unsqueeze(1)
        self.link = (1 - ww_ij) * self.link

        # add precedence->write
        self.link += torch.bmm(self.precedence.unsqueeze(-1), write_w.unsqueeze(1))

        # zero diag
        diag = torch.eye(self.memory_size, device=write_w.device).unsqueeze(0)
        self.link = self.link * (1 - diag)

        # precedence
        self.precedence = (1 - write_w.sum(dim=-1, keepdim=True)) * self.precedence
        self.precedence = self.precedence + write_w

    def _content_addressing(self, key, strength=1.0):
        """
        Content-based addressing: dot(key, memory), scaled by strength, softmax.
        key: [B, W], memory: [B, N, W]
        => weights: [B, N]
        """
        dot = torch.einsum("bkw,bnw->bn", key.unsqueeze(1), self.memory)
        key_norm = torch.norm(key, 2, dim=-1, keepdim=True) + 1e-8
        mem_norm = torch.norm(self.memory, 2, dim=-1) + 1e-8
        dot = dot / (key_norm * mem_norm)
        dot = dot * strength.unsqueeze(-1)
        return F.softmax(dot, dim=-1)

    def _write_to_memory(self, interface):
        """
        parse interface => erase_vec, write_vec, write_gate, alloc_gate,
        and do memory update: erase + add
        """
        offset = 0
        erase_vec = torch.sigmoid(interface[..., offset:offset+self.head_size])  # [B,W]
        offset += self.head_size

        write_vec = interface[..., offset:offset+self.head_size]                # [B,W]
        offset += self.head_size

        write_gate = torch.sigmoid(interface[..., offset:offset+1]).squeeze(-1) # [B]
        offset += 1

        alloc_gate = torch.sigmoid(interface[..., offset:offset+1]).squeeze(-1) # [B]
        offset += 1

        # get allocation
        alloc_w = self._get_allocation_weights()  # [B, N]
        w_gate = write_gate.unsqueeze(-1)         # [B, 1]
        a_gate = alloc_gate.unsqueeze(-1)         # [B, 1]
        write_w = w_gate * a_gate * alloc_w       # [B, N]
        self.write_weights = write_w

        # erase
        erase_mat = erase_vec.unsqueeze(1)  # [B,1,W]
        self.memory = self.memory * (1 - torch.bmm(write_w.unsqueeze(-1), erase_mat))

        # add
        add_mat = write_vec.unsqueeze(1)    # [B,1,W]
        self.memory = self.memory + torch.bmm(write_w.unsqueeze(-1), add_mat)

        # usage + link
        self._update_usage()
        self._update_temporal_link(write_w)

    def _get_read_weights(self, interface, read_strength):
        """
        read_mode = 3 => [back, forward, content]
        read_key => content-based addressing
        """
        read_mode = interface[..., -3:]          # [B,3]
        read_mode = F.softmax(read_mode, dim=-1) # [B,3]

        read_key = interface[..., :self.head_size]
        cw = self._content_addressing(read_key, strength=read_strength)  # [B, N]

        # forward/backward from link
        bw = torch.bmm(self.link.transpose(1, 2), self.read_weights.unsqueeze(-1)).squeeze(-1) # [B,N]
        fw = torch.bmm(self.link, self.read_weights.unsqueeze(-1)).squeeze(-1)                 # [B,N]

        weights = (read_mode[..., 0:1] * bw) + \
                  (read_mode[..., 1:2] * fw) + \
                  (read_mode[..., 2:3] * cw)
        weights += 1e-8
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights

    def forward(self, x, hidden=None):
        """
        x: [B, seq_len, input_size]
        hidden: (h, c), optional
        Return:
          outs: [B, seq_len, output_size]
          (memory, usage, link, precedence)
          hidden
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        # init memory if needed
        self._init_memory_if_needed(batch_size, device)

        if hidden is None:
            h0 = x.new_zeros(1, batch_size, self.hidden_size)
            c0 = x.new_zeros(1, batch_size, self.hidden_size)
            hidden = (h0, c0)

        # single read vector => [B,W]
        read_vec = torch.zeros(batch_size, self.head_size, device=device)

        outputs = []
        for t in range(seq_len):
            inp_t = torch.cat([x[:, t, :], read_vec], dim=-1).unsqueeze(1)  # [B,1, input_size+W]
            out_ctrl, hidden = self.controller(inp_t, hidden)   # LSTM
            h = out_ctrl.squeeze(1)  # [B, hidden_size]

            # parse interface
            interface = self.fc_interface(h)  # [B, interface_size]
            # parse read_key, read_strength
            offset = (self.head_size * 2) + 2
            read_key = interface[..., offset:offset+self.head_size]
            offset += self.head_size
            read_strength = softplus(interface[..., offset:offset+1]).squeeze(-1)
            offset += 1
            # last 3 => read_mode

            # write
            self._write_to_memory(interface)
            # read
            rw = self._get_read_weights(interface, read_strength)
            self.read_weights = rw

            read_vec = self._read_from_memory(rw)
            # final output
            out = torch.cat([h, read_vec], dim=-1)
            out = self.fc_output(out)
            outputs.append(out.unsqueeze(1))

        outs = torch.cat(outputs, dim=1)  # [B, seq_len, output_size]

        # -------------------------------------------
        # Detach memory + hidden so we don't accidentally
        # backprop through old states across iterations.
        # This prevents "Trying to backward ... second time" errors
        # if you reuse the same DNC object next iteration.
        # If you want full BPTT across iterations, remove these lines.
        # -------------------------------------------
        self.memory = self.memory.detach()  
        self.usage = self.usage.detach()
        self.precedence = self.precedence.detach()
        self.link = self.link.detach()
        self.read_weights = self.read_weights.detach()
        self.write_weights = self.write_weights.detach()

        hidden = (hidden[0].detach(), hidden[1].detach())

        return outs, (self.memory, self.usage, self.link, self.precedence), hidden



class TransformerController(nn.Module):
    """
    A simple Transformer-based module that processes each time step.
    For NTM-like usage, you'd still have an external 'read/write' operation,
    but let's keep it simple here.
    """
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerController, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        out = self.encoder(x)  # same shape
        return out


##############################################################################
# Transformer based NTM
##############################################################################
class TransformerNTM(nn.Module):
    """
    A 'transformer-based' variant for demonstration.
    We skip the external memory for brevity, but you could combine
    this with NTM-like read/write heads if desired.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        # memory stuff if you want,
        # or skip for a pure transformer
    ):
        super(TransformerNTM, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer = TransformerController(d_model=hidden_size, nhead=4,
                                                 num_layers=2, dim_feedforward=4*hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None, memory=None):
        # x: [batch_size, seq_len, input_size]
        emb = self.embedding(x)  # [B, seq_len, hidden_size]
        # pass through stacked Transformer
        trans_out = self.transformer(emb)  # [B, seq_len, hidden_size]
        out = self.fc_out(trans_out)       # [B, seq_len, output_size]
        return out, None, None

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



# def mezo_step(
#     model: torch.nn.Module,
#     x: torch.Tensor,
#     y: torch.Tensor,
#     criterion: torch.nn.Module,
#     epsilon: float = 1e-3,
#     lr: float = 1e-3,
#     weight_decay: float = 1e-5
#     layerwise: bool = False,
# ) -> torch.Tensor:
#     """
#     Memory-Efficient Zero-Order (MeZO) demonstration with optional weight decay.
    
#     Args:
#       model: The NTM (or any) model to update.
#       x: Input batch tensor.
#       y: Target batch tensor.
#       criterion: Loss function (e.g., BCEWithLogitsLoss).
#       epsilon: Perturbation magnitude for finite difference.
#       layerwise: If True, do layer-by-layer (param-by-param) perturbation.
#                  If False, do a single big perturbation for the entire model.
#       weight_decay: L2 regularization coefficient. 
#                     If > 0, adds 0.5 * weight_decay * sum(param^2) to the loss.
    
#     Returns:
#       A PyTorch scalar tensor representing the average loss from the plus/minus passes.
#     """
#     model.train()
#     all_params = list(model.parameters())

#     def compute_weight_decay_loss(model, wd):
#         """
#         Compute 0.5 * wd * sum of squares of all parameters that require grad.
#         Returns a float or a torch scalar. We'll do float + manual add to the final
#         so we keep the data consistent with the final computations.
#         """
#         if wd == 0.0:
#             return 0.0
#         sum_of_squares = 0.0
#         for p in model.parameters():
#             if p.requires_grad:
#                 sum_of_squares += p.data.pow(2).sum().item()
#         return 0.5 * wd * sum_of_squares

#     if layerwise:
#         # Layerwise approach: For each parameter, do +epsilon pass, -2*epsilon pass, etc.
#         total_loss = 0.0
#         count_params = 0
#         for param in all_params:
#             if not param.requires_grad:
#                 continue
#             count_params += 1

#             original_data = param.data.clone()

#             # + epsilon
#             param.data = original_data + epsilon
#             outputs, _, _ = model(x)
#             seq_len = min(outputs.size(1), y.size(1))
#             loss_plus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])
#             # Add weight decay penalty
#             loss_plus = loss_plus + compute_weight_decay_loss(model, weight_decay)

#             # -2 * epsilon
#             param.data = original_data - 2 * epsilon
#             outputs, _, _ = model(x)
#             loss_minus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])
#             # Add weight decay penalty
#             loss_minus = loss_minus + compute_weight_decay_loss(model, weight_decay)

#             # Finite difference
#             grad_est = (loss_plus - loss_minus) / (2 * epsilon)

#             # Restore original
#             param.data = original_data

            
#             # We multiply by 'grad_est' (a scalar), then epsilon, etc.
#             # This is a toy approach to do "coordinate-free" direction updates.
#             with torch.no_grad():
#                 # param update
#                 param.data -= lr * grad_est * epsilon * torch.sign(torch.randn_like(param.data))

#             # Accumulate average loss for logging
#             # We'll take the mean of (loss_plus + loss_minus) / 2 to approximate the loss
#             avg_pass_loss = 0.5 * (loss_plus.item() + loss_minus.item())
#             total_loss += avg_pass_loss

#         # Average over number of parameters
#         if count_params > 0:
#             total_loss /= float(count_params)
#         # Return as a Torch tensor so we can call .item() outside
#         return torch.tensor(total_loss, device=x.device)

#     else:
#         # Single-perturbation approach
#         original_data = [p.data.clone() for p in all_params]
#         directions = [torch.randn_like(p.data) if p.requires_grad else None for p in all_params]

#         # + epsilon
#         for p, d in zip(all_params, directions):
#             if p.requires_grad and d is not None:
#                 p.data = p.data + epsilon * d.sign()

#         outputs, _, _ = model(x)
#         seq_len = min(outputs.size(1), y.size(1))
#         loss_plus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])
#         loss_plus = loss_plus + compute_weight_decay_loss(model, weight_decay)

#         # -2 epsilon
#         for p, d in zip(all_params, directions):
#             if p.requires_grad and d is not None:
#                 p.data = p.data - 2 * epsilon * d.sign()

#         outputs, _, _ = model(x)
#         loss_minus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])
#         loss_minus = loss_minus + compute_weight_decay_loss(model, weight_decay)

#         # Restore original data
#         for p, orig_data in zip(all_params, original_data):
#             p.data = orig_data

#         # Finite difference gradient estimate
#         grad_est = (loss_plus - loss_minus) / (2 * epsilon)

#         # Manual update
#         with torch.no_grad():
#             for p, d in zip(all_params, directions):
#                 if p.requires_grad and d is not None:
#                     # grad_est is a scalar
#                     p.data = p.data - lr * grad_est.item() * epsilon * d.sign()

#         # Return average of plus/minus losses as a scalar
#         avg_loss = 0.5 * (loss_plus.item() + loss_minus.item())
#         return torch.tensor(avg_loss, device=x.device)

def mezo_zero_order_grad_single(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    epsilon: float,
    weight_decay: float
) -> float:
    """
    Single-direction MeZO:
    1) Create a single random direction for all params.
    2) Do +epsilon pass -> loss_plus
    3) Do -epsilon pass -> loss_minus
    4) grad_est = (loss_plus - loss_minus)/(2*epsilon) (a scalar)
    5) param.grad = grad_est * sign(direction) for each param
    6) Return average loss for logging
    """
    model.train()
    all_params = list(model.parameters())

    # Zero existing grads
    for p in all_params:
        if p.grad is not None:
            p.grad.zero_()

    orig_data = [p.data.clone() for p in all_params]
    directions = [torch.randn_like(p) if p.requires_grad else None for p in all_params]

    # +epsilon
    for p, d in zip(all_params, directions):
        if p.requires_grad and d is not None:
            p.data.add_(epsilon * d.sign())

    outputs, _, _ = model(x)
    seq_len = min(outputs.size(1), y.size(1))
    loss_plus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])

    # WD for +epsilon pass
    if weight_decay > 0.0:
        wd_plus = 0.5 * weight_decay * sum((p.data**2).sum().item() for p in all_params if p.requires_grad)
        loss_plus = loss_plus + wd_plus

    # -2epsilon
    for p, d in zip(all_params, directions):
        if p.requires_grad and d is not None:
            p.data.sub_(2.0 * epsilon * d.sign())

    outputs, _, _ = model(x)
    loss_minus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])

    # WD for -epsilon pass
    if weight_decay > 0.0:
        wd_minus = 0.5 * weight_decay * sum((p.data**2).sum().item() for p in all_params if p.requires_grad)
        loss_minus = loss_minus + wd_minus

    # grad_est (scalar)
    grad_est = (loss_plus - loss_minus) / (2.0 * epsilon)

    # Restore
    for p, od in zip(all_params, orig_data):
        p.data = od

    # param.grad = grad_est * sign(direction)
    for p, d in zip(all_params, directions):
        if p.requires_grad and d is not None:
            p.grad = grad_est * d.sign()

    # For logging
    avg_loss = 0.5 * (loss_plus.item() + loss_minus.item())
    return avg_loss


def mezo_zero_order_grad_layerwise(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    epsilon: float,
    weight_decay: float
) -> float:
    """
    Layerwise (param-by-param) MeZO with no_grad() to avoid building huge graphs.
    For each param p:
      1) +epsilon pass -> loss_plus
      2) -epsilon pass -> loss_minus
      3) grad_est = (loss_plus - loss_minus)/(2*epsilon)  (scalar)
      4) p.grad += grad_est * sign(direction)
      5) restore p.data
    Return average (loss_plus + loss_minus)/2 over all params for logging.
    """
    model.train()
    all_params = list(model.parameters())

    # We do not do normal autograd, so zero any existing grads:
    for p in all_params:
        if p.grad is not None:
            p.grad.zero_()

    total_loss = 0.0
    param_count = 0

    # We'll compute weight decay penalty if needed
    def compute_wd(params, wd):
        if wd <= 0.0:
            return 0.0
        # Summation of (p^2)
        s = 0.0
        for pp in params:
            if pp.requires_grad:
                s += pp.data.pow(2).sum().item()
        return 0.5 * wd * s

    for param in all_params:
        if not param.requires_grad:
            continue

        param_count += 1
        original_data = param.data.clone()
        direction = torch.randn_like(param)

        # +epsilon pass
        param.data = original_data + epsilon * direction.sign()
        with torch.no_grad():  # <--- no_grad to avoid building a graph
            outputs, _, _ = model(x)
            seq_len = min(outputs.size(1), y.size(1))
            loss_plus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])
            wd_plus = compute_wd(all_params, weight_decay)
            loss_plus = loss_plus + wd_plus

        # -2 epsilon pass
        param.data = original_data - 2.0 * epsilon * direction.sign()
        with torch.no_grad():  # <--- no_grad again
            outputs, _, _ = model(x)
            seq_len = min(outputs.size(1), y.size(1))
            loss_minus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])
            wd_minus = compute_wd(all_params, weight_decay)
            loss_minus = loss_minus + wd_minus

        # finite difference
        grad_est = (loss_plus - loss_minus) / (2.0 * epsilon)

        # restore original parameter
        param.data = original_data

        # Accumulate gradient in param.grad
        # param.grad shape = param.shape
        # grad_est is a scalar, direction.sign() is param-shaped
        if param.grad is None:
            param.grad = grad_est * direction.sign()
        else:
            param.grad.add_(grad_est * direction.sign())

        # logging
        avg_p_loss = 0.5 * (loss_plus.item() + loss_minus.item())
        total_loss += avg_p_loss

    if param_count > 0:
        total_loss /= param_count

    return total_loss


def softplus(x):
    return torch.log1p(torch.exp(x))


##############################################################################
# Main Training/CLI
##############################################################################
def main():
    parser = argparse.ArgumentParser(description="Train an NTM on classical tasks.")
    # Model
    parser.add_argument("--arch", type=str, default="ntm",
                    choices=["ntm", "dnc", "tra"],
                    help="Which architecture to use: NTM, DNC, or a Transformer-based model.")
    
    # Task
    parser.add_argument("--task", type=str, default="copy",
                    choices=["copy", "repeat_copy", "associative_recall",
                             "add", "sub", "mul", "div", "fib", "factorial"],
                    help="Which task to train on.")
    
    parser.add_argument("--seq_len", type=int, default=10)
    
    # Model
    parser.add_argument("--input_size", type=int, default=9)
    parser.add_argument("--output_size", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--memory_size", type=int, default=128)
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=1)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "mezo"])
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    
    parser.add_argument("--max_num", type=int, default=100)

    
    # Cosine LR & Warmup
    parser.add_argument("--cosine_lr", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=0)

    
    # MeZO
    parser.add_argument("--mezo", action="store_true",
                        help="If true, uses mezo-based gradient instead of backprop.")
    parser.add_argument("--epsilon", type=float, default=1e-3,
                        help="Perturbation size for mezo.")
    parser.add_argument("--mezo_layerwise", action="store_true",
                        help="Apply MeZO updates layer by layer.")
    
    # Misc
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--wandb_proj", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()


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

    if args.arch == "ntm":
        model = NTM(
            input_size=args.input_size,
            output_size=args.output_size,
            hidden_size=args.hidden_size,
            memory_size=args.memory_size,
            head_size=args.head_size,
            num_heads=args.num_heads
        )
    elif args.arch == "dnc":
        model = DNC(
            input_size=args.input_size,
            output_size=args.output_size,
            hidden_size=args.hidden_size,
            memory_size=args.memory_size,
            head_size=args.head_size,
            num_heads=args.num_heads
        )
    else:  # "transformer"
        model = TransformerNTM(
            input_size=args.input_size,
            output_size=args.output_size,
            hidden_size=args.hidden_size
            # skip memory if purely Transformer
        )
    
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    max_iters = 1000000000
    
    # Create an optimizer that handles momentum, WD, LR, etc.
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    else:
        # mezo uses e.g. SGD w/ momentum, so LR & WD apply
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                              momentum=0.9, weight_decay=args.weight_decay)
    
    # LR scheduler
    if args.cosine_lr:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=args.max_iters,
                                                         eta_min=1e-6)
    else:
        scheduler = None

    # Initialize W&B if desired
    if args.wandb_proj is not None:
        wandb.init(project=args.wandb_proj, name=args.wandb_run_name)
        wandb.config.update(args)

    global_step = 0

    # ---------- MAIN TRAINING LOOP ------------
    for iteration in range(1, args.max_iters + 1):

        # ~~~~~~~~~~~~ Generate TRAIN data ~~~~~~~~~~~~
        if args.task == "copy":
            x, y = generate_copy_task(
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                bits=args.output_size,
                train=True
            )
        elif args.task == "repeat_copy":
            x, y = generate_repeat_copy_task(
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                bits=args.output_size,
                train=True
            )
        elif args.task == "associative_recall":
            x, y = generate_associative_recall_task(
                batch_size=args.batch_size,
                item_len=3,
                num_items=args.seq_len,
                bits=args.output_size,
                train=True
            )
        elif args.task in ["add", "sub", "mul", "div"]:
            x, y = generate_arithmetic_task(
                batch_size=args.batch_size,
                task_type=args.task,
                max_num=args.max_num,
                train=True
            )
        elif args.task == "fib":
            x, y = generate_fibonacci_task(
                batch_size=args.batch_size,
                max_n=args.max_num,
                train=True
            )
        elif args.task == "factorial":
            x, y = generate_factorial_task(
                batch_size=args.batch_size,
                max_n=args.max_num,
                train=True
            )
        else:
            raise ValueError(f"Unknown task {args.task}")

        x, y = x.to(device), y.to(device)

        # =========== WARMUP LR if needed ============
        if args.optimizer == "adam" and args.warmup_steps > 0 and iteration <= args.warmup_steps:
            frac = iteration / float(args.warmup_steps)
            new_lr = args.learning_rate * frac
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr

        # ================== TRAIN STEP ==================
        if args.mezo:
            # Zero-order approach
            if args.mezo_layerwise:
                avg_loss = mezo_zero_order_grad_layerwise(
                    model, x, y, criterion, epsilon=args.epsilon, weight_decay=args.weight_decay
                )
            else:
                avg_loss = mezo_zero_order_grad_single(
                    model, x, y, criterion, epsilon=args.epsilon, weight_decay=args.weight_decay
                )

            optimizer.step()
            loss_val = avg_loss

        else:
            loss_val = train_step(model, x, y, criterion, optimizer)  # standard backprop

        # Step scheduler
        if scheduler is not None and iteration > args.warmup_steps:
            scheduler.step()

        global_step += 1

        # ~~~~~~~~~ LOG & VALIDATE ~~~~~~~~~
        if iteration % args.log_interval == 0:
            lr_current = optimizer.param_groups[0]["lr"]

            # Evaluate on a small VAL batch
            with torch.no_grad():
                if args.task == "copy":
                    val_x, val_y = generate_copy_task(
                        batch_size=args.batch_size,
                        seq_len=args.seq_len,
                        bits=args.output_size,
                        train=False  # domain shift / 5x seq len
                    )
                elif args.task == "repeat_copy":
                    val_x, val_y = generate_repeat_copy_task(
                        batch_size=args.batch_size,
                        seq_len=args.seq_len,
                        bits=args.output_size,
                        train=False
                    )
                elif args.task == "associative_recall":
                    val_x, val_y = generate_associative_recall_task(
                        batch_size=args.batch_size,
                        item_len=3,
                        num_items=args.seq_len,
                        bits=args.output_size,
                        train=False
                    )
                elif args.task in ["add", "sub", "mul", "div"]:
                    val_x, val_y = generate_arithmetic_task(
                        batch_size=args.batch_size,
                        task_type=args.task,
                        max_num=args.max_num,
                        train=False
                    )
                elif args.task == "fib":
                    val_x, val_y = generate_fibonacci_task(
                        batch_size=args.batch_size,
                        max_n=args.max_num,
                        train=False
                    )
                elif args.task == "factorial":
                    val_x, val_y = generate_factorial_task(
                        batch_size=args.batch_size,
                        max_n=args.max_num,
                        train=False
                    )
                else:
                    raise ValueError(f"Unknown task {args.task}")

                val_x, val_y = val_x.to(device), val_y.to(device)
                val_outputs, _, _ = model(val_x)
                
                # Align seq_len if needed
                seq_len_val = min(val_outputs.size(1), val_y.size(1))
                val_loss_t = criterion(val_outputs[:, :seq_len_val, :], val_y[:, :seq_len_val, :])
                # Suppose we compute bitwise accuracy
                val_acc = compute_batch_accuracy(val_outputs[:, :seq_len_val, :], val_y[:, :seq_len_val, :])

            # Print to stdout
            msg = (f"Iter {iteration}, "
                   f"TrainLoss={loss_val:.4f}, "
                   f"ValLoss={val_loss_t.item():.4f}, "
                   f"ValAcc={val_acc:.3f}, "
                   f"LR={lr_current:.6f}")
            print(msg)
            sys.stdout.flush()

            # W&B logging if desired
            if args.wandb_proj is not None:
                wandb.log({
                    "train_loss": loss_val,
                    "val_loss": val_loss_t.item(),
                    "val_acc": val_acc,
                    "lr": lr_current,
                }, step=global_step)

    print("Finished training!")


if __name__ == "__main__":
    main()
