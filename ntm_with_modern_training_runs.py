#!/usr/bin/env python3
"""
Training script to train a Neural Turing Machine (NTM) on classic tasks w/ multiple "modern" optimization techniques (Adam, Cosine LR, Warmup, MeZO, etc.) to see if anything has changed.

Usage Example:
    python train_ntm.py \
        --task copy \
        --batch_size 16 \
        --max_iters 50000 \
        --hidden_size 128 \
        --memory_size 128 \
        --head_size 64 \
        --num_heads 1 \
        --optimizer adam \
        --learning_rate 1e-3 \
        --cosine_lr True \
        --warmup_steps 1000 \
        --mezo False \
        --wandb_proj "NTM-Experiments" \
        --wandb_run_name "test-run"

Author: Francois Chaubard
"""

import os
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
      x: [batch_size, 2*seq_len + 1, bits+1]
      y: [batch_size, 2*seq_len + 1, bits]
    """
    # Random bit sequences
    seq = torch.randint(0, 2, (batch_size, seq_len, bits), dtype=torch.float32)
    # Add an end-of-sequence marker (one-hot at the last bit index)
    eos = torch.zeros(batch_size, 1, bits + 1)
    eos[..., -1] = 1.0  # marker in the "bits+1"-th dimension
    seq_with_eos = torch.cat([seq, eos], dim=1)
    
    # The input to the network is the sequence + marker + placeholders
    # We'll pad with zeros for the output phase
    pad = torch.zeros(batch_size, seq_len, bits + 1)
    x = torch.cat([seq_with_eos, pad], dim=1)
    
    # The target is zero during the input phase (except ignoring it in loss),
    # then the original bits during the second phase (after the marker).
    pad_out = torch.zeros(batch_size, seq_len + 1, bits)  # input phase + marker
    y_copy = seq  # we want the bits to be reproduced
    y = torch.cat([pad_out, y_copy], dim=1)
    return x, y


def generate_repeat_copy_task(
    batch_size: int, seq_len: int, repeat_min: int = 1, repeat_max: int = 3, bits: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The repeat copy task:
      - Input is a random sequence of bits, then a separate random 'repeat count' 
        (encoded in unary or some form), plus a marker.
      - Model outputs the entire sequence repeated 'count' times.
    """
    # Random bit sequences
    seq = torch.randint(0, 2, (batch_size, seq_len, bits), dtype=torch.float32)
    
    # Random repeat counts
    repeat_counts = torch.randint(repeat_min, repeat_max + 1, (batch_size, 1))
    
    # Marker
    eos = torch.zeros(batch_size, 1, bits + 2)
    eos[..., -1] = 1.0  # marker in last dimension
    
    # We'll encode the repeat count in one additional dimension as one-hot
    # or we can do something simpler, like single dimension = count
    count_vec = torch.zeros(batch_size, 1, bits + 2)
    for i in range(batch_size):
        c = repeat_counts[i].item()
        if c < bits + 2:
            count_vec[i, 0, c] = 1.0  # one-hot in dimension c

    # Combine the sequence with the count and marker
    seq_padded = torch.cat([seq, torch.zeros(batch_size, seq_len, 2)], dim=-1)
    x_input = torch.cat([seq_padded, count_vec, eos], dim=1)
    
    # Construct the target
    max_repeat = repeat_max
    total_out_len = seq_len * max_repeat  # over-generate, then we won't compute loss beyond actual
    target = []
    for i in range(batch_size):
        c = repeat_counts[i].item()
        repeated = seq[i].repeat((c, 1))
        # Pad to max_repeat * seq_len
        pad_len = total_out_len - repeated.shape[0]
        pad_zone = torch.zeros(pad_len, bits)
        out_seq = torch.cat([repeated, pad_zone], dim=0)
        target.append(out_seq)
    y_out = torch.stack(target, dim=0)  # [batch_size, total_out_len, bits]
    
    # For alignment, we make sure x_input and y_out have the same time dimension
    # We'll just match to the bigger dimension (seq_len + 2 vs total_out_len).
    T_in = x_input.size(1)
    T_out = y_out.size(1)
    T = max(T_in, T_out)
    if T_in < T:
        pad_in = torch.zeros(batch_size, T - T_in, bits + 2)
        x_input = torch.cat([x_input, pad_in], dim=1)
    if T_out < T:
        pad_out = torch.zeros(batch_size, T - T_out, bits)
        y_out = torch.cat([y_out, pad_out], dim=1)
    return x_input, y_out


def generate_associative_recall_task(
    batch_size: int, item_len: int = 3, num_items: int = 3, bits: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified version of the Associative Recall task:
      - We have num_items items, each of length item_len (bits).
      - We ask the model to recall the item that follows a given 'query' item in the list.
      - The input includes all items plus the query item + marker, the output is the single item that follows the query.
    """
    # Generate random items
    items = torch.randint(0, 2, (batch_size, num_items, item_len, bits), dtype=torch.float32)
    
    # Randomly pick which item is the query, ensuring it isn't the last
    query_indices = torch.randint(0, num_items - 1, (batch_size,))
    queries = []
    answers = []
    for i in range(batch_size):
        q_idx = query_indices[i].item()
        queries.append(items[i, q_idx])
        answers.append(items[i, q_idx + 1])  # The "associated" item is the next one
    
    queries = torch.stack(queries, dim=0)   # [batch_size, item_len, bits]
    answers = torch.stack(answers, dim=0)   # [batch_size, item_len, bits]
    
    # Flatten the items into a single sequence to feed
    # Then place a marker at the end and then the query
    flattened = []
    for i in range(batch_size):
        batch_seq = items[i].reshape(num_items * item_len, bits)
        flattened.append(batch_seq)
    flattened = torch.stack(flattened, dim=0)  # [batch_size, num_items*item_len, bits]
    
    # Append a marker dimension
    eos = torch.zeros(batch_size, 1, bits + 1)
    eos[..., -1] = 1.0
    # Expand flattened to bits+1
    flattened_pad = torch.cat([flattened, torch.zeros(batch_size, flattened.size(1), 1)], dim=-1)
    
    # Expand queries to bits+1
    query_pad = torch.cat([queries, torch.zeros(batch_size, item_len, 1)], dim=-1)
    
    x_input = torch.cat([flattened_pad, eos, query_pad], dim=1)
    
    # Our target is the item that follows the query
    y_out = answers  # [batch_size, item_len, bits]
    # We might want to pad up to match length
    T_in = x_input.size(1)
    T_out = y_out.size(1)
    if T_out < T_in:
        pad_out = torch.zeros(batch_size, T_in - T_out, bits)
        y_out = torch.cat([y_out, pad_out], dim=1)
    return x_input, y_out


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


def mezo_step(model, x, y, criterion, epsilon=1e-3, layerwise=False):
    """
    Memory-Efficient Zero-Order (MeZO) demonstration:
    - For each layer (or for the entire model if layerwise=False), we do:
      1) Save original weights
      2) Perturb weights by +epsilon, compute forward pass & loss
      3) Perturb by -2*epsilon, compute forward pass & loss
      4) Finite diff => grad ~ (loss_plus - loss_minus) / 2*epsilon
      5) Restore original, apply update
    This is a toy demonstration (not a fully optimized or stable approach).
    """
    model.train()
    all_params = list(model.parameters())
    
    if layerwise:
        # For each layer, do the perturbation
        total_loss = 0.0
        for param in all_params:
            if param.requires_grad:
                original_data = param.data.clone()
                
                # +epsilon
                param.data = original_data + epsilon
                outputs, _, _ = model(x)
                seq_len = min(outputs.size(1), y.size(1))
                loss_plus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])
                
                # -2 epsilon
                param.data = original_data - 2 * epsilon
                outputs, _, _ = model(x)
                loss_minus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])
                
                # gradient estimate
                grad_est = (loss_plus - loss_minus) / (2 * epsilon)
                
                # restore
                param.data = original_data
                
                # manual update
                # Let's pick some "learning rate" for demonstration
                lr = 1e-3
                param.data = param.data - lr * grad_est * epsilon * torch.sign(torch.randn_like(param.data))
                
                total_loss += (loss_plus.item() + loss_minus.item()) / 2.0
        return total_loss / len(all_params)
    else:
        # Single big perturbation approach
        original_data = [p.data.clone() for p in all_params]
        
        # Random direction for each param
        directions = [torch.randn_like(p.data) for p in all_params]
        
        # +epsilon
        for p, d in zip(all_params, directions):
            p.data = p.data + epsilon * d.sign()
        outputs, _, _ = model(x)
        seq_len = min(outputs.size(1), y.size(1))
        loss_plus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])
        
        # -2 epsilon
        for p, d in zip(all_params, directions):
            p.data = p.data - 2 * epsilon * d.sign()
        outputs, _, _ = model(x)
        loss_minus = criterion(outputs[:, :seq_len, :], y[:, :seq_len, :])
        
        # gradient estimate
        grad_est = (loss_plus - loss_minus) / (2 * epsilon)
        
        # restore
        for p, orig_data in zip(all_params, original_data):
            p.data = orig_data
        
        # manual update
        lr = 1e-3
        for p, d in zip(all_params, directions):
            p.data = p.data - lr * grad_est.item() * epsilon * d.sign()
        
        return (loss_plus.item() + loss_minus.item()) / 2.0


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
    
    # Create model
    model = NTM(
        input_size=args.input_size, 
        output_size=args.output_size,
        hidden_size=args.hidden_size,
        memory_size=args.memory_size,
        head_size=args.head_size,
        num_heads=args.num_heads
    ).to(device)
    
    # Criterion
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = None  # We'll do mezo manually
    
    # Learning rate scheduler (cosine)
    if args.cosine_lr and args.optimizer == "adam":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iters, eta_min=1e-6)
    else:
        scheduler = None
    
    # W&B init
    if args.wandb_proj is not None:
        wandb.init(project=args.wandb_proj, name=args.wandb_run_name)
        wandb.config.update(args)
    
    global_step = 0
    
    # Training loop
    for iteration in range(1, args.max_iters + 1):
        # Generate data
        if args.task == "copy":
            x, y = generate_copy_task(args.batch_size, args.seq_len, bits=args.output_size)
        elif args.task == "repeat_copy":
            x, y = generate_repeat_copy_task(args.batch_size, args.seq_len, bits=args.output_size)
        else:
            # default: associative recall
            x, y = generate_associative_recall_task(args.batch_size, item_len=3, num_items=args.seq_len, bits=args.output_size)
        
        x, y = x.to(device), y.to(device)
        
        # Possibly do warmup
        if args.warmup_steps > 0 and iteration <= args.warmup_steps and args.optimizer == "adam":
            warmup_frac = iteration / float(args.warmup_steps)
            for g in optimizer.param_groups:
                g['lr'] = args.learning_rate * warmup_frac
        
        # One step
        if args.mezo:
            loss_val = mezo_step(model, x, y, criterion, layerwise=args.mezo_layerwise)
        else:
            loss_val = train_step(model, x, y, criterion, optimizer)
        
        # Step LR scheduler
        if scheduler is not None:
            scheduler.step()
        
        global_step += 1
        
        # Logging
        if iteration % args.log_interval == 0:
            lr_current = (
                optimizer.param_groups[0]["lr"] if args.optimizer == "adam" else 0.0
            )
            msg = f"Iter: {iteration}, Loss: {loss_val:.4f}, LR: {lr_current:.6f}"
            print(msg)
            sys.stdout.flush()
            
            if args.wandb_proj is not None:
                wandb.log({"loss": loss_val, "lr": lr_current}, step=global_step)
                


if __name__ == "__main__":
    main()
