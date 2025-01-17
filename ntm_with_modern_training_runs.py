#!/usr/bin/env python3
"""
NTM/DNC/Transformer code with ASCII tasks, plus:
 - micro & macro batch logic
 - shift-by-one (<bos>, <eos>) ignoring <PAD>
 - gradient clipping
 - mezo or standard backprop
 - warmup & cosine LR
 - max_seq_len for consistent shapes
 - input_sample_length for generating tasks
 - Logging every iteration (train loss/acc/time) + validation at log_interval
 - "Layerwise" MeZO grouped by layer
 - MeZO now uses momentum from the chosen optimizer (e.g. Adam) by setting param.grad, not param.data
 - A simple curriculum for 'copy' and 'add' tasks
"""

import os
os.environ["WANDB_API_KEY"] = ""
import sys
import math
import random
import argparse
import numpy as np
import pynvml
import string
import wandb
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

##############################################################################
# GPU selection
##############################################################################
def pick_gpu_with_most_free_mem() -> int:
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
# ASCII Vocab + Tokenizer (with <bos> and <eos>)
##############################################################################
def get_char_vocab():
    """
    We'll define a small ASCII set:
      index 0 => <PAD>
      index 1 => <bos>
      index 2 => <eos>
      then digits 0..9, uppercase A..Z, operators + - * / =, space, '|'
    """
    special = ['<PAD>', '<bos>', '<eos>', '+', '-', '*', '/', '=', ' ', '|']
    digits = list(string.digits)
    letters = list(string.ascii_uppercase)

    vocab_list = special + digits + letters  # Ensures special tokens come first
    char_to_id = {ch: i for i, ch in enumerate(vocab_list)}
    id_to_char = {i: ch for i, ch in enumerate(vocab_list)}

    return vocab_list, char_to_id, id_to_char

##############################################################################
# Convert strings -> fixed [B, max_seq_len], pad or truncate
##############################################################################

def str_to_tensor(batch_strs, char_to_id, max_seq_len):
    """
    Convert a batch of tokenized strings to a tensor of fixed size [B, max_seq_len],
    padded or truncated as needed.
    """
    B = len(batch_strs)
    out = torch.zeros(B, max_seq_len, dtype=torch.long)
    for i, s in enumerate(batch_strs):
        # Tokenize the string while preserving special tokens like <bos> and <eos>
        tokens = tokenize_special(s, char_to_id)
        for j, token in enumerate(tokens):
            if j >= max_seq_len:
                break
            out[i, j] = char_to_id.get(token, 0)
    return out


def tensor_to_string(tensor, id_to_char):
    """
    Convert a tensor of token IDs (1D or 2D) back into a string or list of strings.

    Args:
        tensor (torch.Tensor): Tensor of token IDs (e.g., [1, 10, 11, 12, 13, 2]).
        id_to_char (dict): A mapping from token IDs to characters.

    Returns:
        str or List[str]: The corresponding string(s) representation of the tensor.
    """
    if tensor.ndim == 1:
        # Process a single tensor
        tokens = [id_to_char.get(id.item(), '<UNK>') for id in tensor if id.item() != 0]  # Exclude <PAD>
        return ''.join(tokens)
    elif tensor.ndim == 2:
        # Process a batch tensor
        return [''.join([id_to_char.get(id.item(), '<UNK>') for id in row if id.item() != 0]) for row in tensor]
    else:
        raise ValueError("Input tensor must be 1D or 2D.")


def tokenize_special(string, char_to_id):
    """
    Tokenize a string while preserving special tokens.
    For example:
      "<bos>0123<eos>" -> ["<bos>", "0", "1", "2", "3", "<eos>"]
    """
    special_tokens = {'<bos>', '<eos>', '<PAD>'}  # Define your special tokens
    tokens = []
    i = 0
    while i < len(string):
        # Check for special tokens
        if string[i:i+5] == '<bos>':
            tokens.append('<bos>')
            i += 5
        elif string[i:i+5] == '<eos>':
            tokens.append('<eos>')
            i += 5
        else:
            tokens.append(string[i])
            i += 1
    return tokens


##############################################################################
# Shift-by-one logic for tasks
##############################################################################
def shift_by_one_pairs(x_str, y_str):
    return f"<bos>{x_str}", f"{y_str}<eos>"

##############################################################################
# Task Generators
##############################################################################
def generate_copy_task_str(num_samples, input_sample_length, train=True):
    # if not train => input_sample_length *=5 (but we may override with curriculum too)
    import random
    letters = string.ascii_uppercase
    in_list, out_list = [], []
    for _ in range(num_samples):
        data_str = "".join(random.choice(letters) for _ in range(input_sample_length))
        xinp, xtgt = shift_by_one_pairs(data_str, data_str)
        in_list.append(xinp)
        out_list.append(xtgt)
    return in_list, out_list

def generate_repeat_copy_task_str(num_samples, input_sample_length, repeat_min=1, repeat_max=3, train=True):
    import random
    letters = string.ascii_uppercase
    in_list, out_list = [], []
    for _ in range(num_samples):
        data_str = "".join(random.choice(letters) for __ in range(input_sample_length))
        c_val = random.randint(repeat_min, repeat_max)
        repeated = data_str*c_val
        xinp, xtgt = shift_by_one_pairs(data_str+str(c_val)+"|", repeated)
        in_list.append(xinp)
        out_list.append(xtgt)
    return in_list, out_list

def generate_associative_recall_task_str(num_samples, input_sample_length=3, num_items=3, train=True):
    import random
    letters = string.ascii_uppercase
    in_list, out_list = [], []
    for _ in range(num_samples):
        items = ["".join(random.choice(letters) for __ in range(input_sample_length)) for __ in range(num_items)]
        q_idx = random.randint(0, num_items-2)
        query_item = items[q_idx]
        ans_item = items[q_idx+1]
        flat_items= "".join(items)
        xinp, xtgt = shift_by_one_pairs(flat_items+"|"+query_item, ans_item)
        in_list.append(xinp)
        out_list.append(xtgt)
    return in_list, out_list

def generate_arithmetic_task_str(num_samples,
                                 input_sample_length,
                                 task_type="add",
                                 max_num=10,
                                 train=True):
    """
    Generate arithmetic expressions with `input_sample_length` operands.
      e.g. 
        - input_sample_length=2 => "a+b="
        - input_sample_length=3 => "a+b+c="
      Then produce the corresponding result (e.g. sum, difference, product, etc.).
    
    For each expression, we call shift_by_one_pairs(expr_in, out_str) 
    to create the final input/target sequences.
    
    Args:
        num_samples (int): Number of samples to generate.
        input_sample_length (int): Number of operands (e.g. 2 => 'a + b =').
        task_type (str): One of {"add", "sub", "mul", "div"}.
        max_num (int): Upper bound for operand sampling.
        train (bool): If True, we sample from [0..max_num]. Else max_num to max_num*5.
                     
    Returns:
        in_list, out_list: Two lists of strings, typically fed into a function 
                           like shift_by_one_pairs.
    """
    
    in_list, out_list = [], []
    
    # If you want a different range for validation, adapt here:
    if train:
        lo = 0
        hi = max_num  # or something else if not train
    else:
        
        lo = max_num
        hi = max_num*5  # or something else if not train
    
    # Basic operator symbol for string
    # (the sign logic will be handled in how we compute the result, etc.)
    op_symbol = {
        'add': '+',
        'sub': '-',
        'mul': '*',
        'div': '/'
    }.get(task_type, '+')
    
    for _ in range(num_samples):
        # 1) Sample the operands
        nums = [random.randint(lo, hi) for __ in range(input_sample_length)]
        
        # 2) Possibly reorder or avoid zeros, based on task_type
        if task_type == 'sub':
            # Sort descending so you don't get negative results 
            # (this mimics your older "if b>a: swap them" but generalized)
            nums.sort(reverse=True)
        elif task_type == 'div':
            # Also sort descending
            nums.sort(reverse=True)
            # Avoid zero in subsequent operands
            for i in range(1, len(nums)):
                if nums[i] == 0:
                    nums[i] = 1
        
        # 3) Compute the result of applying the operation sequentially 
        #    across the list of operands
        if task_type == 'add':
            result = sum(nums)
        elif task_type == 'sub':
            # a - b - c - ...
            tmp = nums[0]
            for x in nums[1:]:
                tmp -= x
            result = tmp
        elif task_type == 'mul':
            tmp = 1
            for x in nums:
                tmp *= x
            result = tmp
        elif task_type == 'div':
            tmp = nums[0]
            for x in nums[1:]:
                # ensure we handle zero 
                if x == 0: 
                    x = 1
                tmp //= x  # integer division
            result = tmp
        else:
            # Fallback to add
            result = sum(nums)
        
        # 4) Build the expression string. 
        #    For example, if input_sample_length=3 and task_type='add':
        #    "a+b+c="
        expr_str_parts = []
        for i, val in enumerate(nums):
            expr_str_parts.append(str(val))
            if i < len(nums) - 1:
                expr_str_parts.append(op_symbol)
        
        expr_in = "".join(expr_str_parts) + "="
        out_str = str(result)
        
        # 5) Shift-by-one pairs -> <bos> + expr_in => out, out_str => <eos>
        #    This depends on your own shift_by_one_pairs function
        xinp, xtgt = shift_by_one_pairs(expr_in, out_str)
        in_list.append(xinp)
        out_list.append(xtgt)
    
    return in_list, out_list


def generate_fibonacci_task_str(num_samples, input_sample_length, max_n=10, train=True):
    import random
    fib_cache= [0,1]
    for i in range(2, max_n+20):
        fib_cache.append(fib_cache[-1]+ fib_cache[-2])
    lo=0
    hi= max_n
    in_list, out_list= [], []
    for _ in range(num_samples):
        n_val= random.randint(lo,hi)
        fib_n= fib_cache[n_val]
        xinp, xtgt= shift_by_one_pairs(f"{n_val}=", f"{fib_n}")
        in_list.append(xinp)
        out_list.append(xtgt)
    return in_list, out_list

def generate_factorial_task_str(num_samples, input_sample_length, max_n=6, train=True):
    import random
    fact_cache= [1]
    for i in range(1, max_n+20):
        fact_cache.append(fact_cache[-1]*i)
    lo=0
    hi= max_n
    in_list, out_list= [], []
    for _ in range(num_samples):
        n_val= random.randint(lo,hi)
        f_val= fact_cache[n_val]
        xinp, xtgt= shift_by_one_pairs(f"{n_val}=", f"{f_val}")
        in_list.append(xinp)
        out_list.append(xtgt)
    return in_list, out_list

##############################################################################
# Model Classes
##############################################################################
class NTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, memory_size, head_size, num_heads, embed):
        super(NTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.memory_size = memory_size
        self.head_size = head_size
        self.embed = embed

        # Input normalization
        controller_input_size = input_size + num_heads * head_size
        self.input_norm = nn.LayerNorm(controller_input_size)

        # Controller with input normalization
        self.controller = nn.LSTM(controller_input_size, hidden_size, batch_first=True)
        self.controller_norm = nn.LayerNorm(hidden_size)

        # Memory operation layers with normalization
        self.fc_read_keys = nn.Linear(hidden_size, num_heads * head_size)
        self.fc_write_keys = nn.Linear(hidden_size, num_heads * head_size)
        self.fc_write_strength = nn.Linear(hidden_size, num_heads)
        self.fc_erase_vector = nn.Linear(hidden_size, num_heads * head_size)
        
        self.read_keys_norm = nn.LayerNorm(head_size)
        self.write_keys_norm = nn.LayerNorm(head_size)
        self.memory_norm = nn.LayerNorm(head_size)

        # Output projection with normalization - project directly to vocab size
        total_output_size = hidden_size + num_heads * head_size
        self.pre_output_norm = nn.LayerNorm(total_output_size)
        self.fc_proj = nn.Linear(total_output_size, output_size)  # Direct to vocab size

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with appropriate distributions"""
        # Initialize LSTM params
        for name, p in self.controller.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0)

        # Initialize memory operation layers
        for name, p in self.named_parameters():
            if 'fc_' in name and 'weight' in name:
                nn.init.uniform_(p, -0.1, 0.1)
            elif 'fc_' in name and 'bias' in name:
                nn.init.constant_(p, 0)

    def _addressing(self, memory, read_keys, write_keys, write_strengths, erase_vectors):
        """
        Perform memory addressing with normalized inputs and memory.
        
        Args:
            memory: [B, memory_size, head_size]
            read_keys: [B, num_heads, head_size]
            write_keys: [B, num_heads, head_size]
            write_strengths: [B, num_heads]
            erase_vectors: [B, num_heads, head_size]
        """
        B, N, W = memory.size()

        # Normalize memory and keys
        memory_normalized = self.memory_norm(memory)
        read_keys = self.read_keys_norm(read_keys.view(-1, W)).view(B, self.num_heads, W)
        write_keys = self.write_keys_norm(write_keys.view(-1, W)).view(B, self.num_heads, W)
    
        # Read operation with normalized attention
        read_weights = torch.einsum('bnk,bmk->bnm', read_keys, memory_normalized)  # No scaling, let LayerNorm handle it
        read_weights = F.softmax(read_weights, dim=-1)
        read_content = torch.einsum('bnm,bmh->bnh', read_weights, memory)
    
        # Write operation with normalized attention
        write_weights = torch.einsum('bnk,bmk->bnm', write_keys, memory_normalized)
        write_weights = F.softmax(write_weights, dim=-1)
    
        # Erase and write operations
        erase_vectors_expanded = torch.einsum('bnm,bnh->bmh', write_weights, erase_vectors)
        memory = memory * (1 - erase_vectors_expanded)
    
        add_content = write_strengths.unsqueeze(-1) * write_keys
        add_content_expanded = torch.einsum('bnm,bnh->bmh', write_weights, add_content)
        memory = memory + add_content_expanded
    
        return memory, read_content, read_weights, write_weights

    def forward(self, x_emb, hidden=None, memory=None):
        B, L, E = x_emb.size()
        device = x_emb.device

        # Initialize states if needed
        if hidden is None:
            h0 = x_emb.new_zeros(1, B, self.hidden_size)
            c0 = x_emb.new_zeros(1, B, self.hidden_size)
            hidden = (h0, c0)
        
        if memory is None:
            memory = x_emb.new_zeros(B, self.memory_size, self.head_size)

        outputs = []
        read_contents = x_emb.new_zeros(B, self.num_heads, self.head_size)

        for t in range(L):
            # Normalize and combine input with read vector
            controller_input = torch.cat([x_emb[:, t, :], read_contents.view(B, -1)], dim=-1)
            controller_input = self.input_norm(controller_input)
            
            # Controller
            out_ctrl, hidden = self.controller(controller_input.unsqueeze(1), hidden)
            h = self.controller_norm(out_ctrl.squeeze(1))

            # Generate memory parameters
            read_keys = self.fc_read_keys(h).view(B, self.num_heads, self.head_size)
            write_keys = self.fc_write_keys(h).view(B, self.num_heads, self.head_size)
            write_strengths = torch.sigmoid(self.fc_write_strength(h)).view(B, self.num_heads)
            erase_vectors = torch.sigmoid(self.fc_erase_vector(h)).view(B, self.num_heads, self.head_size)

            # Memory operations
            memory, read_contents, _, _ = self._addressing(
                memory, read_keys, write_keys, write_strengths, erase_vectors
            )

            # Output projection with normalization - project directly to logits
            output = torch.cat([h, read_contents.view(B, -1)], dim=-1)
            output = self.pre_output_norm(output)
            logits = self.fc_proj(output)  # Direct projection to vocab size
            outputs.append(logits.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs, memory, hidden



class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TransformerMemoryNTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, memory_size, head_size, num_heads, embed):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.memory_size = memory_size
        self.head_size = head_size
        self.embed = embed

        # Input normalization
        self.input_norm = nn.LayerNorm(input_size + num_heads * head_size)

        # Keep LSTM controller
        self.controller = nn.LSTM(input_size + num_heads * head_size, hidden_size, batch_first=True)
        self.controller_norm = nn.LayerNorm(hidden_size)

        # Memory operation layers
        self.fc_read_keys = nn.Linear(hidden_size, num_heads * head_size)
        self.fc_write_keys = nn.Linear(hidden_size, num_heads * head_size)
        self.fc_write_strength = nn.Linear(hidden_size, num_heads)
        self.fc_erase_vector = nn.Linear(hidden_size, num_heads * head_size)

        # Add transformer for memory processing
        self.memory_transformer = TransformerBlock(head_size, nhead=4, dim_feedforward=2*head_size)
        
        # Normalization layers
        self.read_keys_norm = nn.LayerNorm(head_size)
        self.write_keys_norm = nn.LayerNorm(head_size)
        self.memory_norm = nn.LayerNorm(head_size)
        
        # Output layers
        total_output_size = hidden_size + num_heads * head_size
        self.pre_output_norm = nn.LayerNorm(total_output_size)
        self.fc_proj = nn.Linear(total_output_size, output_size)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        # Initialize LSTM params
        for name, p in self.controller.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0)

        # Initialize memory operation layers
        for name, p in self.named_parameters():
            if 'fc_' in name and 'weight' in name:
                nn.init.uniform_(p, -0.1, 0.1)
            elif 'fc_' in name and 'bias' in name:
                nn.init.constant_(p, 0)

    def _addressing(self, memory, read_keys, write_keys, write_strengths, erase_vectors):
        """Memory addressing with transformer-enhanced memory"""
        B, N, W = memory.size()

        # Transform memory using transformer
        memory = self.memory_transformer(memory)
        
        # Normalize memory and keys
        memory_normalized = self.memory_norm(memory)
        read_keys = self.read_keys_norm(read_keys.view(-1, W)).view(B, self.num_heads, W)
        write_keys = self.write_keys_norm(write_keys.view(-1, W)).view(B, self.num_heads, W)
    
        # Read operation
        read_weights = torch.einsum('bnk,bmk->bnm', read_keys, memory_normalized)
        read_weights = F.softmax(read_weights, dim=-1)
        read_content = torch.einsum('bnm,bmh->bnh', read_weights, memory)
    
        # Write operation
        write_weights = torch.einsum('bnk,bmk->bnm', write_keys, memory_normalized)
        write_weights = F.softmax(write_weights, dim=-1)
        
        # Memory update
        erase_vectors_expanded = torch.einsum('bnm,bnh->bmh', write_weights, erase_vectors)
        memory = memory * (1 - erase_vectors_expanded)
    
        add_content = write_strengths.unsqueeze(-1) * write_keys
        add_content_expanded = torch.einsum('bnm,bnh->bmh', write_weights, add_content)
        memory = memory + add_content_expanded
    
        return memory, read_content, read_weights, write_weights

    def forward(self, x_emb, hidden=None, memory=None):
        B, L, E = x_emb.size()

        # Initialize states if needed
        if hidden is None:
            h0 = x_emb.new_zeros(1, B, self.hidden_size)
            c0 = x_emb.new_zeros(1, B, self.hidden_size)
            hidden = (h0, c0)

        if memory is None:
            memory = x_emb.new_zeros(B, self.memory_size, self.head_size)

        outputs = []
        read_contents = x_emb.new_zeros(B, self.num_heads, self.head_size)

        for t in range(L):
            # LSTM controller
            inp_t = torch.cat([x_emb[:, t, :], read_contents.view(B, -1)], dim=-1)
            inp_t = self.input_norm(inp_t)
            out_ctrl, hidden = self.controller(inp_t.unsqueeze(1), hidden)
            h = self.controller_norm(out_ctrl.squeeze(1))

            # Memory operations
            read_keys = self.fc_read_keys(h).view(B, self.num_heads, self.head_size)
            write_keys = self.fc_write_keys(h).view(B, self.num_heads, self.head_size)
            write_strengths = torch.sigmoid(self.fc_write_strength(h)).view(B, self.num_heads)
            erase_vectors = torch.sigmoid(self.fc_erase_vector(h)).view(B, self.num_heads, self.head_size)

            memory, read_contents, _, _ = self._addressing(
                memory, read_keys, write_keys, write_strengths, erase_vectors
            )

            # Output projection
            output = torch.cat([h, read_contents.view(B, -1)], dim=-1)
            output = self.pre_output_norm(output)
            logits = self.fc_proj(output)
            outputs.append(logits.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs, memory, hidden


class TransformerMemoryDNC(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, memory_size, head_size, num_heads, embed):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.head_size = head_size
        self.num_reads = num_heads
        self.embed = embed

        # Input normalization
        controller_input_size = input_size + self.num_reads * head_size
        self.input_norm = nn.LayerNorm(controller_input_size)

        # Keep LSTM controller
        self.controller = nn.LSTM(controller_input_size, hidden_size, batch_first=True)
        self.controller_norm = nn.LayerNorm(hidden_size)

        # Memory operation layers
        self.fc_read_keys = nn.Linear(hidden_size, self.num_reads * head_size)
        self.fc_write_keys = nn.Linear(hidden_size, head_size)
        self.fc_write_strength = nn.Linear(hidden_size, 1)
        self.fc_erase_vector = nn.Linear(hidden_size, head_size)
        self.fc_add_vector = nn.Linear(hidden_size, head_size)

        # Add transformers for memory processing
        self.memory_transformer = TransformerBlock(head_size, nhead=4, dim_feedforward=2*head_size)
        self.read_transformer = TransformerBlock(head_size, nhead=4, dim_feedforward=2*head_size)
        
        # Normalization layers
        self.read_keys_norm = nn.LayerNorm(head_size)
        self.write_keys_norm = nn.LayerNorm(head_size)
        self.memory_norm = nn.LayerNorm(head_size)

        # Output layers
        total_output_size = hidden_size + self.num_reads * head_size
        self.pre_output_norm = nn.LayerNorm(total_output_size)
        self.fc_proj = nn.Linear(total_output_size, output_size)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        # Initialize LSTM params
        for name, p in self.controller.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0)

        # Initialize memory operation layers
        for name, p in self.named_parameters():
            if 'fc_' in name and 'weight' in name:
                nn.init.uniform_(p, -0.1, 0.1)
            elif 'fc_' in name and 'bias' in name:
                nn.init.constant_(p, 0)

    def _read_memory(self, memory, read_keys):
        """Enhanced memory reading with transformer processing"""
        # Transform memory
        memory = self.memory_transformer(memory)
        
        # Process read keys with transformer
        read_keys = self.read_transformer(read_keys.view(-1, self.head_size).unsqueeze(1)).squeeze(1)
        
        # Normalize memory and keys
        memory_normalized = self.memory_norm(memory)
        read_keys = self.read_keys_norm(read_keys).view(-1, self.num_reads, self.head_size)

        # Compute attention weights
        read_weights = torch.softmax(
            torch.einsum('bnh,bmh->bnm', read_keys, memory_normalized),
            dim=2
        )
        read_vectors = torch.einsum('bnm,bmh->bnh', read_weights, memory)
        return read_vectors

    def _write_memory(self, memory, write_keys, write_str, erase_vec, write_vec):
        """Enhanced memory writing with transformer processing"""
        # Transform memory
        memory = self.memory_transformer(memory)
        
        # Normalize memory and keys
        memory_normalized = self.memory_norm(memory)
        write_keys = self.write_keys_norm(write_keys)

        # Compute write weights
        write_weights = torch.softmax(
            torch.einsum('bh,bmh->bm', write_keys, memory_normalized),
            dim=1
        ).unsqueeze(1)

        # Scale by write strength
        write_weights = write_weights * write_str.unsqueeze(1)

        # Update memory
        erase = torch.einsum('bnm,bh->bmh', write_weights, erase_vec)
        write = torch.einsum('bnm,bh->bmh', write_weights, write_vec)
        memory = memory * (1 - erase) + write
        
        return memory

    def forward(self, x_emb, hidden=None, memory=None):
        B, L, E = x_emb.size()
        device = x_emb.device

        # Initialize states if needed
        if hidden is None:
            h0 = x_emb.new_zeros(1, B, self.hidden_size)
            c0 = x_emb.new_zeros(1, B, self.hidden_size)
            hidden = (h0, c0)

        if memory is None:
            memory = x_emb.new_zeros(B, self.memory_size, self.head_size)

        read_vec = x_emb.new_zeros(B, self.num_reads * self.head_size)
        outputs = []

        for t in range(L):
            # LSTM controller
            inp_t = torch.cat([x_emb[:, t, :], read_vec], dim=-1)
            inp_t = self.input_norm(inp_t)
            out_ctrl, hidden = self.controller(inp_t.unsqueeze(1), hidden)
            h = self.controller_norm(out_ctrl.squeeze(1))

            # Memory parameters
            read_keys = self.fc_read_keys(h).view(B, self.num_reads, self.head_size)
            write_keys = self.fc_write_keys(h)
            write_str = torch.sigmoid(self.fc_write_strength(h))
            erase_vec = torch.sigmoid(self.fc_erase_vector(h))
            write_vec = torch.tanh(self.fc_add_vector(h))

            # Memory operations with transformer enhancement
            memory = self._write_memory(memory, write_keys, write_str, erase_vec, write_vec)
            read_vectors = self._read_memory(memory, read_keys)
            read_vec = read_vectors.reshape(B, -1)

            # Output projection
            output = torch.cat([h, read_vec], dim=-1)
            output = self.pre_output_norm(output)
            logits = self.fc_proj(output)
            outputs.append(logits.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs, memory, hidden



class DNC(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, memory_size, head_size, num_heads, embed):
        super(DNC, self).__init__()
        self.input_size = input_size
        self.output_size = output_size  # This should be vocab_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.head_size = head_size
        self.num_reads = num_heads
        self.embed = embed

        # Input normalization
        controller_input_size = input_size + self.num_reads * self.head_size
        self.input_norm = nn.LayerNorm(controller_input_size)
        
        # Controller with normalization
        self.controller = nn.LSTM(controller_input_size, hidden_size, batch_first=True)
        self.controller_norm = nn.LayerNorm(hidden_size)

        # Memory operation layers with normalization
        self.fc_read_keys = nn.Linear(hidden_size, self.num_reads * self.head_size)
        self.fc_write_keys = nn.Linear(hidden_size, self.head_size)
        self.fc_write_strength = nn.Linear(hidden_size, 1)
        self.fc_erase_vector = nn.Linear(hidden_size, self.head_size)
        self.fc_add_vector = nn.Linear(hidden_size, self.head_size)

        self.read_keys_norm = nn.LayerNorm(head_size)
        self.write_keys_norm = nn.LayerNorm(head_size)
        self.memory_norm = nn.LayerNorm(head_size)

        # Output projection with normalization - project directly to vocab size
        total_output_size = hidden_size + self.num_reads * self.head_size
        self.pre_output_norm = nn.LayerNorm(total_output_size)
        self.fc_proj = nn.Linear(total_output_size, output_size)  # Project directly to vocab size

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with appropriate distributions"""
        # Initialize LSTM params
        for name, p in self.controller.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0)

        # Initialize memory operation layers
        for name, p in self.named_parameters():
            if 'fc_' in name and 'weight' in name:
                nn.init.uniform_(p, -0.1, 0.1)
            elif 'fc_' in name and 'bias' in name:
                nn.init.constant_(p, 0)

    def _read_memory(self, memory, read_keys):
        """Read from memory using normalized attention."""
        # Normalize memory and keys
        memory_normalized = self.memory_norm(memory)
        read_keys = self.read_keys_norm(read_keys.view(-1, self.head_size)).view(-1, self.num_reads, self.head_size)

        # Compute attention weights (no scaling, let LayerNorm handle it)
        read_weights = torch.softmax(
            torch.einsum('bnh,bmh->bnm', read_keys, memory_normalized),
            dim=2
        )
        read_vectors = torch.einsum('bnm,bmh->bnh', read_weights, memory)
        return read_vectors

    def _write_memory(self, memory, write_keys, write_str, erase_vec, write_vec):
        """Write to memory using normalized attention."""
        # Normalize memory and keys
        memory_normalized = self.memory_norm(memory)
        write_keys = self.write_keys_norm(write_keys)

        # Compute write weights
        write_weights = torch.softmax(
            torch.einsum('bh,bmh->bm', write_keys, memory_normalized),
            dim=1
        ).unsqueeze(1)  # [B, 1, memory_size]

        # Scale by write strength
        write_weights = write_weights * write_str.unsqueeze(1)

        # Erase and write operations
        erase = torch.einsum('bnm,bh->bmh', write_weights, erase_vec)
        write = torch.einsum('bnm,bh->bmh', write_weights, write_vec)
        
        # Update memory
        memory = memory * (1 - erase) + write
        return memory

    def forward(self, x_emb, hidden=None, memory=None):
        B, L, E = x_emb.size()
        device = x_emb.device

        # Initialize states if needed
        if hidden is None:
            h0 = x_emb.new_zeros(1, B, self.hidden_size)
            c0 = x_emb.new_zeros(1, B, self.hidden_size)
            hidden = (h0, c0)

        if memory is None:
            memory = x_emb.new_zeros(B, self.memory_size, self.head_size)

        read_vec = x_emb.new_zeros(B, self.num_reads * self.head_size)
        outputs = []

        for t in range(L):
            # Normalize and combine input with read vector
            controller_input = torch.cat([x_emb[:, t, :], read_vec], dim=-1)
            controller_input = self.input_norm(controller_input)
            
            # Controller
            out_ctrl, hidden = self.controller(controller_input.unsqueeze(1), hidden)
            h = self.controller_norm(out_ctrl.squeeze(1))

            # Memory parameters
            read_keys = self.fc_read_keys(h).view(B, self.num_reads, self.head_size)
            write_keys = self.fc_write_keys(h)
            write_str = torch.sigmoid(self.fc_write_strength(h))
            erase_vec = torch.sigmoid(self.fc_erase_vector(h))
            write_vec = torch.tanh(self.fc_add_vector(h))

            # Memory operations
            memory = self._write_memory(memory, write_keys, write_str, erase_vec, write_vec)
            read_vectors = self._read_memory(memory, read_keys)
            read_vec = read_vectors.reshape(B, -1)

            # Output projection with normalization - project directly to logits
            output = torch.cat([h, read_vec], dim=-1)
            output = self.pre_output_norm(output)
            logits = self.fc_proj(output)  # Direct projection to vocab size
            outputs.append(logits.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs, memory, hidden



class TransformerController(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerController, self).__init__()
        encoder_layer= nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout, batch_first=True)
        self.encoder= nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x):
        return self.encoder(x)


class TransformerNTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, embed):
        super(TransformerNTM, self).__init__()
        self.embed = embed  # Assign the embedding layer
        self.transformer = TransformerController(
            d_model=input_size, nhead=4, num_layers=2, dim_feedforward=4 * input_size
        )
        self.fc_out = nn.Linear(input_size, output_size)

    def forward(self, x_emb, hidden=None, memory=None):
        trans_out = self.transformer(x_emb)
        out = self.fc_out(trans_out)
        return out, None, None


##############################################################################
# Grouping parameters by top-level "layer" for faster "layerwise" mezo
##############################################################################
def group_params_by_layer(named_params):
    layer_dict = {}
    for name, p in named_params:
        if not p.requires_grad:
            continue
        parts = name.split('.')
        if len(parts) >=2:
            layer_name = '.'.join(parts[:2])
        else:
            layer_name = name
        if layer_name not in layer_dict:
            layer_dict[layer_name] = []
        layer_dict[layer_name].append((name, p))
    return layer_dict

from torch.nn.utils.rnn import pad_sequence
import pdb
def teacher_forcing_loss_emb(model, x_emb, y_ids, criterion, teacher_force=True):
    """
    Teacher-forcing loss for step-by-step sequence generation.
    
    Args:
        model (nn.Module): The sequence-to-sequence model.
        x_emb (torch.Tensor): [B, Lx, E], embedded input sequence.
        y_ids (torch.LongTensor): [B, Ly], token IDs for the target sequence.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).

    Returns:
        torch.Tensor: Average loss per token.
    """
    B, Lx, E = x_emb.size()
    y_ids_trimmed = [y[y != 0] for y in y_ids]
    y_ids_unpadded = pad_sequence(y_ids_trimmed, batch_first=True, padding_value=0)

    Ly = y_ids_unpadded.size(1)
    device = x_emb.device


    # Initialize hidden states
    hidden = None
    memory = None

    # Forward input sequence step-by-step
    for t in range(Lx):
        inp_t = x_emb[:, t, :].unsqueeze(1)  # [B, 1, E]
        logits, memory, hidden = model(inp_t, hidden=hidden, memory=memory)

    # Now process target sequence step-by-step
    total_loss = 0.0
    num_tokens = 0

    for t in range(Ly):  
        logits = logits.squeeze(1)  # [B, vocab_size]
        # Determine whether to continue teacher forcing
        pred_t = logits.argmax(dim=-1)  # Predicted token IDs
        counter=0
        while pred_t.eq(0).all() and counter>10: # its allowed to think for 10 iters per token.. after that incur loss
            counter+=1
            inp_t = model.embed(pred_t).unsqueeze(1)
            logits, memory, hidden = model(inp_t, hidden=hidden, memory=memory)
            logits = logits.squeeze(1)  # [B, vocab_size]
            # Determine whether to continue teacher forcing
            pred_t = logits.argmax(dim=-1)  # Predicted token IDs
        
        # pdb.set_trace()
        # we made our prediction for this t so lets get some loss
        target_t = y_ids_unpadded[:, t ]  # Ground truth at step t+1 (excluding <bos>)
        step_loss = criterion(logits, target_t)
        total_loss += step_loss #.item()
        # correct += (target_t == pred_t).sum().item()
        num_tokens += (target_t != 0).sum().item()

        if not teacher_force: # if we are not teacher forcing, we should feed its own output back in
            target_t = pred_t
        
  
        # now we take the next truth and give it to the model and iter again
        target_t_emb = model.embed(target_t).unsqueeze(1)  # [B, 1, E]
  
        # Generate prediction for the current step
        logits, memory, hidden = model(target_t_emb, hidden=hidden, memory=memory) # TODO, hidden may need a transpose?

    # Average loss across all valid tokens
    return total_loss/Ly #  / num_tokens if num_tokens > 0 else 0.0

##############################################################################
# MeZO single
##############################################################################
def mezo_char_single(model, x_emb, y, criterion, epsilon=1e-3):
    """
    Single-step MeZO + teacher forcing when 'x_emb' is [B, Lx, E] 
    and 'y' is [B, Ly] (token IDs).

    Steps:
      0) local_seed = torch.randint(0, 2**32, (1,))
      1) "Plus pass": 
         - Set torch.manual_seed(local_seed)
         - For each param p: p += ε * randn_like(p)
         - forward => loss_plus = teacher_forcing_loss_emb(...)
      2) "Minus pass":
         - Set torch.manual_seed(local_seed)
         - For each param p: p -= 2ε * randn_like(p)   (so net = θ - ε·d)
         - forward => loss_minus
      3) "Revert":
         - Set torch.manual_seed(local_seed)
         - For each param p: p += ε * randn_like(p)    (so net = θ again)
      4) "Gradient":
         - grad_est = (loss_plus - loss_minus) / (2ε)
         - For each param p: p.grad += grad_est * randn_like(p)
         (so your momentum/optimizer can apply the update)

    Args:
        model (nn.Module):
            - Must define `model.embed` if you call it in teacher_forcing_loss_emb
            - forward(...) returns shape [B, T, vocab] plus memory, hidden
        x_emb (torch.Tensor): [B, Lx, E], the embedded input 
        y (torch.LongTensor): [B, Ly] target token IDs
        criterion (nn.Module): e.g. nn.CrossEntropyLoss()
        epsilon (float): FD scale

    Returns:
        float: average (loss_plus + loss_minus)/2
    """

    # 0) local_seed => ensures the same directions in plus/minus
    local_seed = torch.randint(0, 2**32, (1,)).item()
    all_params = list(model.parameters())

    # Zero out old gradients
    for p in all_params:
        if p.grad is not None:
            p.grad.zero_()

    # -----------------------------------------------------
    # 1) PLUS pass: (θ -> θ + ε·d)
    # -----------------------------------------------------
    torch.manual_seed(local_seed)
    with torch.no_grad():
        for p in all_params:
            if p.requires_grad:
                d = torch.randn_like(p)
                p.data.add_(epsilon * d)

    loss_plus = teacher_forcing_loss_emb(model, x_emb, y, criterion)

    # -----------------------------------------------------
    # 2) MINUS pass: (from θ + ε·d => θ - ε·d)
    # -----------------------------------------------------
    torch.manual_seed(local_seed)
    with torch.no_grad():
        for p in all_params:
            if p.requires_grad:
                d = torch.randn_like(p)
                p.data.sub_(2.0 * epsilon * d)

    loss_minus = teacher_forcing_loss_emb(model, x_emb, y, criterion)

    # -----------------------------------------------------
    # 3) Revert => (θ - ε·d + ε·d => θ)
    # -----------------------------------------------------
    torch.manual_seed(local_seed)
    with torch.no_grad():
        for p in all_params:
            if p.requires_grad:
                d = torch.randn_like(p)
                p.data.add_(epsilon * d)

    # -----------------------------------------------------
    # 4) Finite-Difference Gradient => p.grad += grad_est * d
    # -----------------------------------------------------
    grad_est = (loss_plus - loss_minus) / (2.0 * epsilon)

    torch.manual_seed(local_seed)
    for p in all_params:
        if p.requires_grad:
            d = torch.randn_like(p)
            if p.grad is None:
                p.grad = grad_est * d
            else:
                p.grad.add_(grad_est * d)

    # Return the average of the two losses
    return 0.5 * (loss_plus.item() + loss_minus.item())



# UNTESTED AND NOT WORKING.. JUST A ROUGH IDEA! 
# ##############################################################################
# # Initialize Rolling MeZO
# ##############################################################################
# def init_rolling_mezo(base_model, epsilon, directions=None, seed=0):
#     """
#     1) Creates two perturbed copies (model_plus, model_minus) = (theta^+, theta^-).
#     2) Stores the random direction z_i used for each parameter, plus the scalar epsilon.
#     3) Optionally seeds the random generator for reproducibility.

#     Args:
#         base_model (nn.Module): The original (unperturbed) model M_theta.
#         epsilon (float): Perturbation scale, ε.
#         directions (list[torch.Tensor] or None): If provided, we use these as z_i.
#                                                 Otherwise, sample random.
#         seed (int): Random seed for reproducibility.

#     Returns:
#         mezo_state (dict) containing:
#             {
#                "model_plus":  (nn.Module)  # copy of base_model with +ε z_i
#                "model_minus": (nn.Module)  # copy of base_model with -ε z_i
#                "directions":  list[Tensor] # the random directions z_i
#                "epsilon":     float
#                "seed":        int          # updated seed
#             }
#     """
#     torch.manual_seed(seed)
#     all_params = list(base_model.parameters())

#     # If no directions given, generate random directions z_i
#     if directions is None:
#         directions = []
#         for p in all_params:
#             if p.requires_grad:
#                 directions.append(torch.randn_like(p))
#             else:
#                 directions.append(None)

#     # Create deep copies of the base model
#     model_plus  = type(base_model)().to(next(base_model.parameters()).device)
#     model_minus = type(base_model)().to(next(base_model.parameters()).device)

#     # Copy state_dict from base_model into them
#     model_plus.load_state_dict(base_model.state_dict())
#     model_minus.load_state_dict(base_model.state_dict())

#     # Now apply +ε z_i and -ε z_i in place
#     plus_params  = list(model_plus.parameters())
#     minus_params = list(model_minus.parameters())

#     for p_plus, p_minus, d in zip(plus_params, minus_params, directions):
#         if d is not None:
#             # p_plus = p_plus + εz_i
#             p_plus.data.add_(epsilon * d)
#             # p_minus = p_minus - εz_i
#             p_minus.data.sub_(epsilon * d)

#     mezo_state = {
#         "model_plus":  model_plus,
#         "model_minus": model_minus,
#         "directions":  directions,
#         "epsilon":     epsilon,
#         "seed":        seed
#     }
#     return mezo_state


# ##############################################################################
# # Rolling MeZO Single Step (loop over time steps)
# ##############################################################################
# def mezo_char_single_rolling(base_model,
#                              x_seq,
#                              y_seq,
#                              criterion,
#                              mezo_state):
#     """
#     Rolling MeZO single step, looping over time steps t in [1..B].
    
#     This follows the pseudocode structure:
#       d ← 0
#       for t in [1.. B]:
#          L_t^+ = Loss(model_plus(x_t), y_t)
#          L_t^- = Loss(model_minus(x_t), y_t)
#          d += (L_t^+ - L_t^-)/(2ε)

#     Then we set p.grad = d * z_i (for each parameter p).

#     Args:
#         base_model (nn.Module): The original unperturbed model (used only to get .parameters()).
#         x_seq (torch.Tensor): Input sequence for B time steps, shape [B, ...] or [B, L, ...].
#         y_seq (torch.Tensor): Target sequence, same leading shape as x_seq.
#         criterion (callable): Loss function (e.g., CrossEntropyLoss).
#         mezo_state (dict): 
#             {
#                "model_plus":  nn.Module,  # the +ε model
#                "model_minus": nn.Module,  # the -ε model
#                "directions":  list of Tensors (z_i for each param)
#                "epsilon":     float
#                ...
#             }
#     Returns:
#         float: Average loss => (sum(L_t^+ + L_t^-)/ (2*B)).
#     """
#     model_plus  = mezo_state["model_plus"]
#     model_minus = mezo_state["model_minus"]
#     directions  = mezo_state["directions"]
#     epsilon     = mezo_state["epsilon"]

#     # We assume x_seq.shape[0] = B time steps (or B examples).
#     # If your model processes entire sequences at once, you can just do
#     # one big pass. But the pseudocode suggests step-by-step. So:
#     B = x_seq.shape[0]

#     # Zero out old grads from base_model
#     all_params = list(base_model.parameters())
#     for p in all_params:
#         if p.grad is not None:
#             p.grad.zero_()

#     # Make sure model_plus, model_minus are in eval mode (no dropout, etc.)
#     model_plus.eval()
#     model_minus.eval()

#     # We'll track the sums of plus/minus losses across time steps
#     loss_plus_sum  = 0.0
#     loss_minus_sum = 0.0

#     # Option 1: Step-by-step approach (like the pseudocode).
#     # If your RNN requires hidden states carried over, you must keep them
#     # in e.g. hidden_plus, hidden_minus, and update them each step.
#     hidden_plus = None
#     hidden_minus = None

#     for t in range(B):
#         x_t = x_seq[t].unsqueeze(0)  # shape [1, ...]
#         y_t = y_seq[t].unsqueeze(0)  # shape [1, ...]

#         # Forward pass on model_plus
#         out_plus, hidden_plus_mem, hidden_plus = model_plus(x_t, hidden_plus)
#         # We might only get out_plus if it's a single-step RNN. Adjust as needed.
#         # Compute the step-loss
#         loss_plus_t = criterion(out_plus.view(-1, out_plus.size(-1)),
#                                 y_t.view(-1))

#         # Forward pass on model_minus
#         out_minus, hidden_minus_mem, hidden_minus = model_minus(x_t, hidden_minus)
#         loss_minus_t = criterion(out_minus.view(-1, out_minus.size(-1)),
#                                  y_t.view(-1))

#         loss_plus_sum  += loss_plus_t.item()
#         loss_minus_sum += loss_minus_t.item()

#     # -- Now compute the total FD gradient estimate over all time steps:
#     # d = sum_t [ (L_t^+ - L_t^-)/(2ε) ]
#     # We'll store just the sum of (L_t^+ - L_t^-), then scale once.
#     delta_loss = (loss_plus_sum - loss_minus_sum)
#     d_est = delta_loss / (2.0 * epsilon)

#     # -- Write param.grad = d_est * directions[i]
#     for p, z_i in zip(all_params, directions):
#         if (p.requires_grad) and (z_i is not None):
#             if p.grad is None:
#                 p.grad = d_est * z_i.clone()  # clone for safety
#             else:
#                 p.grad.add_(d_est * z_i)

#     # Return average loss over all time steps (for logging)
#     avg_loss = 0.5 * (loss_plus_sum + loss_minus_sum) / float(B)
#     return avg_loss



##############################################################################
# "Layerwise" mezo but grouped => fewer fwd passes
##############################################################################
def mezo_char_layerwise(model, x, y, criterion, epsilon=1e-3):
    named_params= list(model.named_parameters())
    layer_dict= group_params_by_layer(named_params)
    all_params= list(model.parameters())
    for p in all_params:
        if p.grad is not None:
            p.grad.zero_()

    total_loss= 0.0
    layer_count=0
    with torch.no_grad():
        for layer_name, param_list in layer_dict.items():
            layer_count+=1
            orig_data= []
            directions= []
            for _, p in param_list:
                orig_data.append(p.data.clone())
                directions.append(torch.randn_like(p))

            # + eps
            for i, (_, p) in enumerate(param_list):
                p.data.add_(epsilon * directions[i])

                # p.data.add_(epsilon * directions[i].sign())

            out_p,_,_= model(x)
            Bp,Lp,Vp= out_p.size()
            loss_plus= criterion(out_p.view(Bp*Lp, Vp), y.view(Bp*Lp))

            # -2 eps
            for i, (_, p) in enumerate(param_list):
                p.data.sub_(2.0*epsilon * directions[i])

                # p.data.sub_(2.0* epsilon* directions[i].sign())

            out_m,_,_= model(x)
            Bm,Lm,Vm= out_m.size()
            loss_minus= criterion(out_m.view(Bm*Lm,Vm), y.view(Bm*Lm))

            # restore
            for i, (_, p) in enumerate(param_list):
                p.data.copy_(orig_data[i])

            grad_est= (loss_plus- loss_minus)/(2*epsilon)

            # accumulate param.grad
            for i, (_, p) in enumerate(param_list):
                if p.grad is None:
                    p.grad= grad_est* directions[i]#.sign()
                else:
                    p.grad.add_( grad_est* directions[i])#.sign())

            avg_loss= 0.5*(loss_plus.item()+ loss_minus.item())
            total_loss+= avg_loss

    if layer_count>0:
        total_loss/= float(layer_count)
    return total_loss





def mezo_char_single(model, x_emb, y, criterion, epsilon=1e-3, verbose=False):
    """Instrumented version of mezo_char_single"""
    if verbose:
        print("\n=== MEZO SINGLE DEBUG ===")
        print(f"Input shape: {x_emb.shape}")
        print(f"Target shape: {y.shape}")

    local_seed = torch.randint(0, 2**32, (1,)).item()
    all_params = list(model.parameters())

    if verbose:
        print(f"\nRandom seed: {local_seed}")
        print("Parameter shapes:")
        for i, p in enumerate(all_params):
            if p.requires_grad:
                print(f"Param {i}: {p.shape}")

    # Zero gradients
    for p in all_params:
        if p.grad is not None:
            p.grad.zero_()
    
    if verbose:
        print("\n=== Plus Pass ===")

    # Plus pass
    torch.manual_seed(local_seed)
    with torch.no_grad():
        for i, p in enumerate(all_params):
            if p.requires_grad:
                d = torch.randn_like(p)
                p.data.add_(epsilon * d)
                if verbose and i < 3:  # Show first few params
                    print(f"\nParam {i} perturbation statistics:")
                    print(f"Mean perturbation: {d.mean().item()}")
                    print(f"Std perturbation: {d.std().item()}")

    loss_plus = teacher_forcing_loss_emb(model, x_emb, y, criterion)
    if verbose:
        print(f"Plus pass loss: {loss_plus.item()}")

    if verbose:
        print("\n=== Minus Pass ===")

    # Minus pass
    torch.manual_seed(local_seed)
    with torch.no_grad():
        for p in all_params:
            if p.requires_grad:
                d = torch.randn_like(p)
                p.data.sub_(2.0 * epsilon * d)

    loss_minus = teacher_forcing_loss_emb(model, x_emb, y, criterion)
    if verbose:
        print(f"Minus pass loss: {loss_minus.item()}")

    # Revert parameters
    torch.manual_seed(local_seed)
    with torch.no_grad():
        for p in all_params:
            if p.requires_grad:
                d = torch.randn_like(p)
                p.data.add_(epsilon * d)

    # Compute and apply gradients
    grad_est = (loss_plus - loss_minus) / (2.0 * epsilon)
    if verbose:
        print(f"\nEstimated gradient: {grad_est}")

    torch.manual_seed(local_seed)
    for i, p in enumerate(all_params):
        if p.requires_grad:
            d = torch.randn_like(p)
            if p.grad is None:
                p.grad = grad_est * d
            else:
                p.grad.add_(grad_est * d)
            
            if verbose and i < 3:  # Show first few params
                print(f"\nParam {i} gradient statistics:")
                print(f"Mean gradient: {p.grad.mean().item()}")
                print(f"Std gradient: {p.grad.std().item()}")

    avg_loss = 0.5 * (loss_plus.item() + loss_minus.item())
    if verbose:
        print(f"\nFinal average loss: {avg_loss}")
        print("=== MEZO Single Complete ===\n")
        # import pdb
        # pdb.set_trace()
    return avg_loss



def generate_sequence_batched(model, x_ids, embed, char_to_id, id_to_char, max_seq_len, device, 
                               criterion=None, y_ids=None, bos_token='<bos>', eos_token='<eos>', 
                               pad_token='<PAD>', teacher_force=False, verbose=False):
    """
    Generate sequences step-by-step using a sequence-to-sequence model.

    Args:
        model (nn.Module): Sequence-to-sequence model.
        x_ids (torch.Tensor): [B, Lx], token IDs for input sequences.
        embed (nn.Embedding): Embedding layer.
        char_to_id (dict): Token-to-ID mapping.
        id_to_char (dict): ID-to-token mapping.
        max_seq_len (int): Maximum sequence length for generation.
        device (torch.device): Device for computation.
        criterion (nn.Module, optional): Loss function for token-level loss computation.
        y_ids (torch.Tensor, optional): [B, Ly], token IDs for target sequences.
        bos_token (str): Beginning-of-sequence token.
        eos_token (str): End-of-sequence token.
        pad_token (str): Padding token.
        teacher_force (bool): Whether to apply teacher forcing during generation.
        verbose (bool): If True, prints debug information.

    Returns:
        Tuple containing:
        - generated_strs: List[str], generated strings without special tokens.
        - generated_strs_with_all_special_tokens: List[str], generated strings with special tokens.
        - generated_ids: List[List[int]], token IDs of generated sequences.
        - probs_batch: List[List[float]], probabilities for each generated token.
        - avg_loss: float, average token-level loss.
        - avg_token_level_accuracy: float, average token-level accuracy.
        - avg_sample_level_accuracy: float, average sample-level accuracy.
    """
    B, Lx = x_ids.size()
    device = x_ids.device

    # Special token IDs
    bos_id = char_to_id[bos_token]
    eos_id = char_to_id[eos_token]
    pad_id = char_to_id[pad_token]

    # Embed the input sequence
    x_emb = embed(x_ids)  # [B, Lx, E]

    # Initialize memory, hidden states, and outputs
    memory = None
    hidden = None
    generated_ids = [[] for _ in range(B)]
    probs_batch = [[] for _ in range(B)]
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_samples = 0

    # Process input sequence step-by-step
    for t in range(Lx):
        inp_t = x_emb[:, t, :].unsqueeze(1)  # [B, 1, E]
        logits, memory, hidden = model(inp_t, hidden=hidden, memory=memory)
        
    
    for step in range(max_seq_len):
        logits = logits.squeeze(1)
        probs = torch.softmax(logits, dim=-1)
        next_tokens = probs.argmax(dim=-1)  # Greedy decoding

        # Handle "thinking steps" for PAD predictions
        counter = 0
        while next_tokens.eq(pad_id).all() and counter < 10:
            counter += 1
            inp_t_emb = model.embed(next_tokens).unsqueeze(1) 
            logits, memory, hidden = model(inp_t_emb, hidden=hidden, memory=memory)
            logits = logits.squeeze(1)  # [B, vocab_size]
            probs = torch.softmax(logits, dim=-1)
            next_tokens = probs.argmax(dim=-1)

        # Record generated tokens and probabilities
        for b in range(B):
            if len(generated_ids[b]) < max_seq_len and next_tokens[b] != eos_id:
                generated_ids[b].append(next_tokens[b].item())
                probs_batch[b].append(float(probs[b, next_tokens[b].item()]))
                
        # Compute loss if criterion and targets are provided
        if criterion is not None and y_ids is not None and step < y_ids.size(1):
            target_t = y_ids[:, step]  # Ground truth for this step
            assert not torch.isnan(logits).any(), "Logits contain NaN!"
            assert not torch.isinf(logits).any(), "Logits contain Inf!"
            assert y_ids.max() < logits.size(-1), "Target token ID out of bounds!"

            
            step_loss = criterion(logits, target_t)
            total_loss += step_loss.item()
            total_correct += (next_tokens == target_t).sum().item()
            total_tokens += (target_t != pad_id).sum().item()

        
        if all(eos_id in gen for gen in generated_ids):  # Stop if all sequences contain EOS
            break

        # Teacher forcing: use ground truth if provided
        if teacher_force and y_ids is not None and step < y_ids.size(1):
            next_tokens = y_ids[:, step]
        inp_t_emb = model.embed(next_tokens).unsqueeze(1) 
        logits, memory, hidden = model(inp_t_emb, hidden=hidden, memory=memory)

        
    # Convert token IDs to strings
    generated_strs = [tensor_to_string(torch.tensor(gen), id_to_char) for gen in generated_ids]
    generated_strs_with_all_special_tokens = [
        tensor_to_string(torch.tensor(gen), id_to_char) for gen in generated_ids
    ]

    # Calculate accuracies
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    avg_token_level_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    avg_sample_level_accuracy = sum(
        [1 for b in range(B) if generated_ids[b] == y_ids[b].tolist()]
    ) / B if y_ids is not None else 0.0

    return (
        generated_strs,
        generated_strs_with_all_special_tokens,
        generated_ids,
        probs_batch,
        avg_loss,
        avg_token_level_accuracy,
        avg_sample_level_accuracy,
    )

# def generate_sequence_batched(model, x_ids, embed, char_to_id, id_to_char, max_seq_len, device, 
#                              criterion=None, y_ids=None, bos_token='<bos>', eos_token='<eos>', 
#                              pad_token='<PAD>', verbose=False):
#     """
#     Debug-instrumented version of sequence generation.
#     """
#     if verbose:
#         print("\n=== SEQUENCE GENERATION DEBUG ===")
#         print(f"Input x_ids shape: {x_ids.shape}")
#         print(f"First sequence in batch:")
#         print(f"Raw x_ids: {x_ids[0].tolist()}")
#         print(f"Decoded x_ids: {''.join([id_to_char[id.item()] for id in x_ids[0] if id.item() != 0])}")
#         if y_ids is not None:
#             print(f"\nTarget y_ids shape: {y_ids.shape}")
#             print(f"Raw y_ids: {y_ids[0].tolist()}")
#             print(f"Decoded y_ids: {''.join([id_to_char[id.item()] for id in y_ids[0] if id.item() != 0])}")
    
#     model.eval()
    
#     # Get batch size and sequence length
#     B, Lx = x_ids.shape
    
#     # Get special token IDs
#     bos_id = char_to_id.get(bos_token, 1)
#     eos_id = char_to_id.get(eos_token, 2)
#     pad_id = char_to_id.get(pad_token, 0)

#     # Debug info for dimensions and special tokens
#     if verbose:
#         print(f"\nBatch size: {B}, Input length: {Lx}")
#         print(f"Special tokens - BOS: {bos_id}, EOS: {eos_id}, PAD: {pad_id}")

#     # Pad sequences
#     x_ids = torch.stack([
#         F.pad(x_ids[i][x_ids[i] != 0], (0, x_ids.size(1) - (x_ids[i] != 0).sum()), value=0)
#         for i in range(x_ids.size(0))
#     ])
    
#     if verbose:
#         print("\nAfter padding:")
#         print(f"Padded x_ids: {x_ids[0].tolist()}")

#     # Initial forward pass
#     with torch.no_grad():
#         prompt_emb = embed(x_ids)
#         out_prompt, memory, hidden = model(prompt_emb, hidden=None, memory=None)
#         if verbose:
#             print(f"\nPrompt embedding shape: {prompt_emb.shape}")
#             print(f"Initial output shape: {out_prompt.shape}")

#     # Initialize generation lists
#     generated_ids = []
#     for b in range(B):
#         row = x_ids[b].tolist()
#         generated_ids.append(row[:])
#         if verbose and b == 0:
#             print(f"\nInitial generated_ids for batch 0: {row}")

#     # Initialize tracking
#     ended = [False]*B
#     probs_batch = [[] for _ in range(B)]
#     pred_seqs = [[] for _ in range(B)]
#     all_logits = [[] for _ in range(B)]

#     # Process targets
#     target_seqs = []
#     if y_ids is not None:
#         if verbose:
#             print("\nProcessing targets:")
#         for b in range(B):
#             tgt_seq = []
#             for t in y_ids[b]:
#                 tok = t.item()
#                 if tok != pad_id:
#                     tgt_seq.append(tok)
#             target_seqs.append(tgt_seq)
#             if verbose and b == 0:
#                 print(f"Target sequence for batch 0: {tgt_seq}")
#                 print(f"Decoded: {''.join([id_to_char[id] for id in tgt_seq])}")

#     # Generation loop
#     if verbose:
#         print("\n=== Starting Generation ===")
    
#     for step_idx in range(max_seq_len):
#         if all(ended):
#             if verbose:
#                 print("All sequences ended")
#             break

#         # Prepare input tokens
#         tokens_to_feed = []
#         for b in range(B):
#             if ended[b]:
#                 tokens_to_feed.append(pad_id)
#             else:
#                 last_token = generated_ids[b][-1]
#                 tokens_to_feed.append(last_token)
#                 if verbose and b == 0:
#                     print(f"\nStep {step_idx}:")
#                     print(f"Feeding token: {last_token} ({id_to_char.get(last_token, '?')})")

#         # Forward pass
#         tokens_tensor = torch.tensor(tokens_to_feed, dtype=torch.long, device=device)
#         x_emb_step = embed(tokens_tensor.unsqueeze(1))

#         with torch.no_grad():
#             out_step, memory, hidden = model(x_emb_step, hidden=hidden, memory=memory)
#             logits = out_step[:, -1, :]
#             probs = F.softmax(logits, dim=-1)
#             next_tokens = probs.argmax(dim=-1)
            
#             if verbose:
#                 print(f"Generated token: {next_tokens[0].item()} ({id_to_char.get(next_tokens[0].item(), '?')})")
#                 # Show top 5 probable tokens
#                 top_k = 5
#                 top_probs, top_indices = probs[0].topk(top_k)
#                 print(f"Top {top_k} probable next tokens:")
#                 for prob, idx in zip(top_probs, top_indices):
#                     print(f"  {id_to_char.get(idx.item(), '?')}: {prob.item():.4f}")

#         # Record predictions
#         for b in range(B):
#             if not ended[b]:
#                 chosen_id = next_tokens[b].item()
#                 pval = probs[b, chosen_id].item()
#                 probs_batch[b].append(pval)
#                 pred_seqs[b].append(chosen_id)
#                 all_logits[b].append(logits[b])

#         # Update sequences
#         for b in range(B):
#             if not ended[b]:
#                 ntok = next_tokens[b].item()
#                 generated_ids[b].append(ntok)
#                 if ntok == eos_id:
#                     ended[b] = True
#                     if verbose and b == 0:
#                         print("Sequence ended with EOS")

#     # Generation results
#     if verbose:
#         print("\n=== Generation Complete ===")
#         print("Batch 0 results:")
#         print(f"Predictions: {pred_seqs[0]}")
#         print(f"Decoded: {''.join([id_to_char.get(id, '?') for id in pred_seqs[0]])}")
#         if y_ids is not None:
#             print(f"Target: {target_seqs[0]}")
#             print(f"Decoded: {''.join([id_to_char.get(id, '?') for id in target_seqs[0]])}")

#     # Loss computation
#     total_loss = 0.0
#     total_correct = 0
#     total_gentoks = 0

#     if y_ids is not None and criterion is not None:
#         if verbose:
#             print("\n=== Computing Loss ===")
#         for b in range(B):
#             pred_len = len(pred_seqs[b])
#             tgt_len = len(target_seqs[b])
            
#             if pred_len > 0 and tgt_len > 0:
#                 min_len = min(pred_len, tgt_len)
#                 batch_preds = pred_seqs[b][:min_len]
#                 batch_targets = target_seqs[b][:min_len]
#                 batch_logits = all_logits[b][:min_len]
                
#                 if len(batch_logits) > 0:
#                     stacked_logits = torch.stack(batch_logits)
#                     target_tensor = torch.tensor(batch_targets, device=device)
                    
#                     if verbose and b == 0:
#                         print("\nToken-by-token comparison:")
#                         for i, (p, t) in enumerate(zip(batch_preds, batch_targets)):
#                             print(f"Position {i}: {id_to_char.get(p, '?')} vs {id_to_char.get(t, '?')}")
                    
#                     seq_loss = criterion(stacked_logits, target_tensor)
#                     step_loss = seq_loss.item() * min_len
#                     total_loss += step_loss
                    
#                     pred_tensor = torch.tensor(batch_preds, device=device)
#                     step_correct = (pred_tensor == target_tensor).sum().item()
#                     total_correct += step_correct
#                     total_gentoks += min_len
                    
#                     if verbose and b == 0:
#                         print(f"Sequence loss: {seq_loss.item()}")
#                         print(f"Correct tokens: {step_correct}/{min_len}")

#     # Calculate final metrics
#     avg_loss = 0.0
#     avg_accuracy = 0.0
#     if total_gentoks > 0:
#         avg_loss = total_loss / total_gentoks
#         avg_accuracy = total_correct / total_gentoks

#     # Build output strings
#     generated_strs = []
#     generated_strs_with_all_special_tokens = []

#     for b in range(B):
#         new_seq = generated_ids[b][Lx:]  
#         skip_pad_chars = []
#         all_chars = []

#         for tid in new_seq:
#             ch = id_to_char.get(tid, "?")
#             all_chars.append(ch)
#             if tid != pad_id:
#                 skip_pad_chars.append(ch)

#         generated_str = "".join(skip_pad_chars)
#         full_str = "".join(all_chars)
#         generated_strs.append(generated_str)
#         generated_strs_with_all_special_tokens.append(full_str)

#         if verbose and b == 0:
#             print(f"\nGenerated string: '{generated_str}'")
#             print(f"With special tokens: '{full_str}'")

#     if verbose:
#         print("\n=== Final Statistics ===")
#         print(f"Total tokens evaluated: {total_gentoks}")
#         print(f"Total correct: {total_correct}")
#         print(f"Average loss: {avg_loss}")
#         print(f"Average accuracy: {avg_accuracy}")
#         print("=== Debug Complete ===\n")
        
#         # import pdb
#         # pdb.set_trace()

#     return (
#         generated_strs,
#         generated_strs_with_all_special_tokens,
#         generated_ids,
#         probs_batch,
#         avg_loss,
#         avg_accuracy
#     )




##############################################################################
# Main
##############################################################################
def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="ntm", choices=["ntm","dnc","tra", "tdnc", "tntm"])
    parser.add_argument("--task", type=str, default="copy",
                        choices=["copy","repeat_copy","associative_recall","add","sub","mul","div","fib","factorial"])
    parser.add_argument("--input_sample_length", type=int, default=2,
                        help="Base length for generating tasks. We'll do a simple curriculum on some tasks.")
    parser.add_argument("--max_seq_len", type=int, default=50,
                        help="We pad/truncate all inputs/targets to this length.")

    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--macro_batch_size", type=int, default=1)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--max_num", type=int, default=100,
                        help="This is the max number in the domain to use in training for arithmetic tasks. Min in the domain is 0. We'll do a simple curriculum for arithmetic if task in all. i.e. [add,sub,mul,div].")

    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--memory_size", type=int, default=128)
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=1)

    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd","mezo"])
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epsilon", type=float, default=1e-2, help="MeZO eps.")
    parser.add_argument("--mezo", action="store_true")

    parser.add_argument("--mezo_flavor", type=str, default="None", choices=["mezo_single","mezo_layerwise", "mezo_rolling","None"])
    
    parser.add_argument("--cosine_lr", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=0)

    parser.add_argument("--grad_norm", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="If >0, grad norm clipped.")
    
    parser.add_argument("--pad_bias", type=float, default=0.0, help="Initial logit bias for <PAD> in final layer. NOT IMPLEMENTED YET")
    parser.add_argument("--log_interval", type=int, default=300)
    parser.add_argument("--wandb_proj", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args= parser.parse_args()

    verbose = False 
    
    total_samples_per_iter = args.micro_batch_size* args.macro_batch_size

    # wandb
    if args.wandb_proj is not None:
        wandb.init(project=args.wandb_proj, name=args.wandb_run_name)
        wandb.config.update(args)

    
    # pick device
    if torch.cuda.is_available():
        gpu_index= pick_gpu_with_most_free_mem()
        device= torch.device(f"cuda:{gpu_index}")
        print(f"[INFO] Using GPU: {gpu_index}")
    else:
        device= torch.device("cpu")
        print("[INFO] Using CPU")

    # build vocab
    vocab_list, char_to_id, id_to_char= get_char_vocab()
    vocab_size= len(vocab_list)

    # embed
    embed= nn.Embedding(vocab_size, args.input_size, padding_idx=0).to(device)

    # model
    if args.arch == "ntm":
        model = NTM(args.input_size, vocab_size, args.hidden_size, args.memory_size, args.head_size, args.num_heads, embed).to(device)
    elif args.arch == "dnc":
        model = DNC(args.input_size, vocab_size, args.hidden_size, args.memory_size, args.head_size, args.num_heads, embed).to(device)
    elif args.arch == "tdnc":
        model = TransformerMemoryDNC(args.input_size, vocab_size, args.hidden_size, embed).to(device)
    elif args.arch == "tntm":
        model = TransformerMemoryNTM(args.input_size, vocab_size, args.hidden_size, embed).to(device)    
    else:
        model = TransformerNTM(args.input_size, vocab_size, args.hidden_size, embed).to(device)


    
    # build optimizer
    params= list(model.parameters())+ list(embed.parameters())
    if args.optimizer=="sgd":
        optimizer= optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        # mezo uses the same optimizer for momentum, we'll do param.grad => momentum => param.data
        # optimizer= optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        optimizer= optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler= None
    if args.cosine_lr:
        scheduler= optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iters, eta_min=1e-6)

    
    criterion= nn.CrossEntropyLoss(ignore_index=0)
    global_step=0

    # track time
    train_start_time= time.time()

    # ++++++++ CURRICULUM LOGIC ++++++++
    # We'll do a naive approach:
    # For "copy" => start with input_sample_length=2, each time we see train_acc>0.95 for a consecutive # of times => +1
    # For "add" => start with max_num=5, each time train_acc>0.95 => max_num+=5
    # We'll track consecutive_succ
    consecutive_succ=0
    def maybe_update_curriculum(train_acc, current_context, current_maxnum, task):
        nonlocal consecutive_succ
        threshold= 0.95
        if train_acc> threshold:
            consecutive_succ+=1
        else:
            consecutive_succ=0
        # if we pass threshold 5 times in a row => increment difficulty
        if consecutive_succ>=5:
            consecutive_succ=0
            # update
            if task=="copy":
                new_ct= current_context+1
                print(f"[CURRICULUM] copy: increasing input_sample_length => {new_ct}")
                return new_ct, current_maxnum
            elif task in ["add","sub","mul","div"]:
                new_mn= current_maxnum+5
                print(f"[CURRICULUM] {task}: increasing max_num => {new_mn}")
                return current_context, new_mn
        return current_context, current_maxnum

    # to start the curriculum
    # if copy => start with input_sample_length=2 (we ignore user param if they want, or we do min(2, user param) for train)
    # if add => start with max_num=5 if user param is bigger
    # if args.task=="copy":
    #     current_context_len= min(args.input_sample_length,2)
    # else:
    current_context_len= args.input_sample_length

    # if args.task in ["add","sub","mul","div"]:
    #     current_max_num= min(args.max_num, 5)
    # else:
    current_max_num= args.max_num

    def generate_task_data(num_samples, task, context_len, maxn, train=True):
        # we do a simplified approach => if not train => we multiply context_len *5
        # but if we want to keep it dynamic, let's do:
        if not train:
            # for copy => use 5x
            # for add => use bigger range
            pass
        # now do the actual generation
        if task=="copy":
            return generate_copy_task_str(num_samples, context_len, train=train)
        elif task=="repeat_copy":
            return generate_repeat_copy_task_str(num_samples, context_len, train=train)
        elif task=="associative_recall":
            return generate_associative_recall_task_str(num_samples, item_len=3, num_items=context_len, train=train)
        elif task in ["add","sub","mul","div"]:
            return generate_arithmetic_task_str(num_samples, context_len, task_type=task,
                                                max_num=maxn, train=train)
        elif task=="fib":
            return generate_fibonacci_task_str(num_samples, context_len, max_n=maxn, train=train)
        elif task=="factorial":
            return generate_factorial_task_str(num_samples, context_len, max_n=maxn, train=train)
        else:
            raise ValueError(f"Unknown task {task}")

    def train_micro_batch(x_emb, y_ids, mezo_state=None):
        """
        x_emb => [micro_batch_size, max_seq_len, embed_dim]
        y_ids => [micro_batch_size, max_seq_len]
        returns => float loss
        """
        if args.optimizer == "mezo":
            if args.mezo_flavor == "mezo_layerwise":
                loss_val= mezo_char_layerwise(model, x_emb, y_ids, criterion, epsilon=args.epsilon)
            elif args.mezo_flavor == "mezo_rolling": 
                loss_val = mezo_char_single_rolling(model, x_emb, y_ids, criterion, mezo_state)

            elif args.mezo_flavor == "mezo_single":
                loss_val= mezo_char_single(model, x_emb, y_ids, criterion, epsilon=args.epsilon)
            else:
                raise Exception("No flavor")
            

        else:
            model.train()
            optimizer.zero_grad()
            out, _, _= model(x_emb)
            B,L,V= out.size()
            loss= criterion(out.view(B*L, V), y_ids.view(B*L))
            loss.backward()
            loss_val= loss.item()
        return loss_val

    mezo_state = None
    if args.mezo_flavor == "mezo_rolling": 
        mezo_state = init_rolling_mezo(model, args.epsilon)

    ####################################
    # TRAIN LOOP
    ####################################
    #for iteration in range(1, args.max_iters+1):
    iteration = -1
    while True:
        iteration+=1
        iter_start_time= time.time()

        # generate data
        # we do a curriculum for train
        x_strs, y_strs= generate_task_data(total_samples_per_iter, args.task,
                                           current_context_len, current_max_num,
                                           train=True)

        model.zero_grad()
        embed.zero_grad()

        micro_loss_sum= 0.0

        # micro/macro approach
        for micro_i in range(args.macro_batch_size):
            start_idx= micro_i* args.micro_batch_size
            end_idx= start_idx+ args.micro_batch_size
            cur_x= x_strs[start_idx:end_idx]
            cur_y= y_strs[start_idx:end_idx]
            
            # take all the excess padding out just in case
            x_ids= str_to_tensor(cur_x, char_to_id, args.input_sample_length+1).to(device)
            x_ids_trimmed = [x[x != 0] for x in x_ids]
            x_ids = pad_sequence(x_ids_trimmed, batch_first=True, padding_value=0)

            y_ids= str_to_tensor(cur_y, char_to_id, args.input_sample_length+1).to(device)
            y_ids_trimmed = [y[y != 0] for y in y_ids]
            y_ids = pad_sequence(y_ids_trimmed, batch_first=True, padding_value=0)

            x_emb= embed(x_ids)
            
                
            loss_val= train_micro_batch(x_emb, y_ids, mezo_state)
            micro_loss_sum+= loss_val

            if verbose:
                print(f"start_idx: {start_idx}")
                print(f"end_idx: {end_idx}")
                print(f"cur_x: {cur_x}")
                print(f"cur_y: {cur_y}")
                print(f"x_ids: {x_ids}")
                print(f"y_ids: {y_ids}")
                print(f"x_emb: {x_emb}")
                print(f"loss_val: {loss_val}")
                print(f"micro_loss_sum: {micro_loss_sum}")
                
                pdb.set_trace()

        # do momentum-based step
        if args.optimizer == "mezo" and args.macro_batch_size>1:
           
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(float(args.macro_batch_size))
            for p in embed.parameters():
                if p.grad is not None:
                    p.grad.div_(float(args.macro_batch_size))
    
        if args.grad_norm:
            # Add gradient normalization
            total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in model.parameters() if p.grad is not None]))
            scale = 1.0 / (total_norm + 1e-6)  # Add epsilon to avoid division by zero
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
        
        if args.grad_clip>0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # finally we step! 
        optimizer.step()

        train_loss_mezo = micro_loss_sum / args.macro_batch_size

        # warmup: TODO, this is NOT implemented correclty but damn does it work lol
        if scheduler is not None and iteration> args.warmup_steps:
            scheduler.step()

        # Something like this:
        # class WarmupScheduler:
        #     def __init__(self, optimizer, warmup_steps, base_lr, final_lr):
        #         """
        #         Args:
        #             optimizer (torch.optim.Optimizer): The optimizer whose LR needs to be warmed up.
        #             warmup_steps (int): Number of warmup steps.
        #             base_lr (float): Initial learning rate (e.g., 0.0).
        #             final_lr (float): Target learning rate after warmup.
        #         """
        #         self.optimizer = optimizer
        #         self.warmup_steps = warmup_steps
        #         self.base_lr = base_lr
        #         self.final_lr = final_lr
        #         self.current_step = 0
        
        #     def step(self):
        #         """Perform one step of warmup."""
        #         self.current_step += 1
        #         if self.current_step <= self.warmup_steps:
        #             # Linearly interpolate between base_lr and final_lr
        #             warmup_lr = self.base_lr + (self.final_lr - self.base_lr) * (self.current_step / self.warmup_steps)
        #             for param_group in self.optimizer.param_groups:
        #                 param_group['lr'] = warmup_lr
        
        #     def finished(self):
        #         """Check if warmup is complete."""
        #         return self.current_step >= self.warmup_steps
        # # Warmup scheduler
        # warmup_scheduler = WarmupScheduler(
        #     optimizer=optimizer,
        #     warmup_steps=args.warmup_steps,
        #     base_lr=0.0,
        #     final_lr=args.learning_rate
        # )
        
        # Warmup step
        # if iteration <= args.warmup_steps:
        #     warmup_scheduler.step()
        # elif scheduler is not None:
        #     scheduler.step()
    
        if iteration % args.log_interval == 0:
            # compute train accuracy on last micro-batch
            with torch.no_grad():
                x_ids = str_to_tensor(cur_x, char_to_id, args.max_seq_len).to(device)  # [B, Lx]
                y_ids = str_to_tensor(cur_y, char_to_id, args.max_seq_len).to(device)  # [B, Ly]
            
                generated_strs, generated_strs_with_padding, gen_ids_batch, probs_batch, train_loss, train_acc, train_acc_sample  = generate_sequence_batched(
                    model=model,
                    x_ids=x_ids,
                    embed=embed,
                    char_to_id=char_to_id,
                    id_to_char=id_to_char,
                    max_seq_len=args.max_seq_len,
                    device=device,
                    criterion=criterion,
                    y_ids=y_ids
                )
    
                counter=0
                for b in range(len(generated_strs)):
                    print("="*30)
                    print(f" [Sample {b}] Input: {cur_x[b]}")
                    print(f" [Sample {b}] Target: {cur_y[b]}")
                    print(f" [Sample {b}] Generated (w/ spec toks): {generated_strs_with_padding[b]}")
                    print(f" [Sample {b}] Generated: {generated_strs[b]}")
                    print(f" [Sample {b}] Token IDs: {gen_ids_batch[b]}")
                    print(f" [Sample {b}] Probabilities: {probs_batch[b]}")
                    print("="*30)
                    counter+=1
                    if counter>4:
                        break
                    
                print(f"Generation loss: {train_loss}, Generation accuracy: {train_acc}, Generation sample accuracy: {train_acc_sample}")
                print("="*30)
                print("="*30)

        # mask = (y_ids != 0)
        # correct = 0
        # total = 0
        # for b in range(1):
        #     # Compare gen_ids_batch[b] (list of length max_seq_len) with y_ids[b]
        #     gen_tensor = torch.tensor(gen_ids_batch[b], device=device)
        #     seq_len = y_ids.size(1)
        #     min_len = min(gen_tensor.size(0), seq_len)
        #     # apply mask
        #     valid_mask = (mask[b, :min_len])
        #     c = ((gen_tensor[:min_len] == y_ids[b, :min_len]) & valid_mask).sum().item()
        #     t = valid_mask.sum().item()
        #     correct += c
        #     total += t
        # train_acc = correct / total if total>0 else 1.0

        if verbose:
            pdb.set_trace()


        iter_end_time= time.time()
        iteration_time= iter_end_time- iter_start_time
        total_elapsed= iter_end_time- train_start_time

        # log every iteration
        lr_current= optimizer.param_groups[0]["lr"]
        
        # msg= (f"Iter={iteration}, train_loss={train_loss_mezo:.3f}, train_acc={train_acc:.3f}, "
        #       f"LR={lr_current:.6f}, iter_time={iteration_time:.2f}s, total_time={total_elapsed/60:.2f}m, "
        #       f"context_len={current_context_len}, max_num={current_max_num}")
        
        msg= (f"Iter={iteration}, train_loss={train_loss_mezo:.3f}, "
              f"LR={lr_current:.6f}, iter_time={iteration_time:.2f}s, total_time={total_elapsed/60:.2f}m, "
              f"context_len={current_context_len}, max_num={current_max_num}")

        print(msg)
        sys.stdout.flush()

        ####################################
        # VAL LOOP
        ####################################
        # validation every log_interval
        if iteration % args.log_interval == 0:
            with torch.no_grad():
                val_samples = total_samples_per_iter
                # Generate a validation batch (vx, vy)
                if args.task=="copy":
                    vx, vy= generate_copy_task_str(val_samples, current_context_len*5, train=False)
                elif args.task=="repeat_copy":
                    vx, vy= generate_repeat_copy_task_str(val_samples, current_context_len*5, train=False)
                elif args.task=="associative_recall":
                    vx, vy= generate_associative_recall_task_str(val_samples, item_len=3,
                                                                 num_items=current_context_len*5, train=False)
                elif args.task in ["add","sub","mul","div"]:
                    vx, vy= generate_arithmetic_task_str(val_samples, input_sample_length=current_context_len,
                                                         task_type=args.task,
                                                         max_num=current_max_num, train=False)
                elif args.task=="fib":
                    vx, vy= generate_fibonacci_task_str(val_samples, current_context_len,
                                                        max_n=current_max_num+5, train=False)
                elif args.task=="factorial":
                    vx, vy= generate_factorial_task_str(val_samples, current_context_len,
                                                        max_n=current_max_num+5, train=False)
                else:
                    raise ValueError(f"Unknown task: {args.task} for val")
        
                # Convert to tensors
                vx_ids= str_to_tensor(vx, char_to_id, args.max_seq_len).to(device)
                vy_ids= str_to_tensor(vy, char_to_id, args.max_seq_len).to(device)
        
                # --------------------------------------------------------------------
                # 1) Optionally, do a teacher-forced pass to measure "val_loss"
                # --------------------------------------------------------------------
                vx_emb= embed(vx_ids)
                model.eval()
                val_out, _, _= model(vx_emb)
                B2,L2,V2= val_out.size()
                val_loss= criterion(val_out.view(B2*L2, V2), vy_ids.view(B2*L2))
        
                # --------------------------------------------------------------------
                # 2) Now do an auto-regressive generation pass
                # --------------------------------------------------------------------
                generated_strs,generated_strs_with_padding, gen_ids_batch, probs_batch, val_gen_loss, val_gen_acc, val_gen_acc_sample = generate_sequence_batched(
                    model=model,
                    x_ids=vx_ids,
                    embed=embed,
                    char_to_id=char_to_id,
                    id_to_char=id_to_char,
                    max_seq_len=args.max_seq_len,
                    device=device,
                    criterion=criterion,  # we pass it to measure cross-entropy while generating
                    y_ids=vy_ids
                )
        
                # For debugging, let's print a few random samples
                sample_indices= random.sample(range(B2), min(3,B2))
                print("\n[DEBUG] Random Val Samples:")
                for idx in sample_indices:
                    print(f"  [Val idx={idx}]")
                    print(f"    Input:  '{vx[idx]}'")
                    print(f"    Target: '{vy[idx]}'")
                    print(f"    Generated(w/ spec toks): '{generated_strs_with_padding[idx]}'")
                    generated_strs_with_padding
                    print(f"    Generated: '{generated_strs[idx]}'")
                    print(f"    Token IDs: {gen_ids_batch[idx]}")
                    print(f"    Probabilities: {probs_batch[idx]}")
        
                    # If you want to see raw token IDs:
                    # print(f"    gen_ids_batch[idx] = {gen_ids_batch[idx]}")
        
                    # If you'd like to see probabilities for newly generated tokens:
                    # print(f"    probs_batch[idx] = {probs_batch[idx]}")


                print(f"Generation loss: {val_gen_loss}, Generation accuracy: {val_gen_acc}, Generation sample accuracy: {val_gen_acc_sample}")
                print("="*30)
                print("="*30)

                print("[END DEBUG]\n")
                # possibly update curriculum
                if args.task in ["copy","add","sub","mul","div"]:
                    new_ctx, new_mn= maybe_update_curriculum(train_acc, current_context_len, current_max_num, args.task)
                    current_context_len= new_ctx
                    current_max_num= new_mn

        
            # Now you have two sets of metrics:
            # val_loss      => teacher-forced cross-entropy
            # val_gen_loss  => cross-entropy measured from auto-regressive generation
            # val_gen_acc   => accuracy from auto-regressive generation
        
            # msg_val= (
            #    f"[VAL] Iter={iteration}, "
            #    f"val_loss={val_loss.item():.3f}, "
            #    f"val_gen_loss={val_gen_loss:.3f}, "
            #    f"val_gen_acc={val_gen_acc:.3f}"
            # )
            # print(msg_val)
            sys.stdout.flush()
        
            if args.wandb_proj is not None:
                # Calculate weight decay loss term
                weight_decay_loss = 0.0
                for param in model.parameters():
                    if param.requires_grad:
                        weight_decay_loss += (args.weight_decay / 2) * torch.sum(param ** 2)

                msg = {
                    "train_loss": train_loss_mezo,
                    "train_gen_loss": train_loss,
                    "train_acc": train_acc,
                    "lr": lr_current,
                    "iter_time_s": iteration_time,
                    "total_time_min": total_elapsed / 60.0,
                    "curr_context_len": current_context_len,
                    "curr_max_num": current_max_num,
                    "val_loss": val_loss.item(),        # teacher-forced
                    "val_gen_loss": val_gen_loss,       # from generation
                    "val_gen_acc": val_gen_acc,
                    "weight_decay_loss": weight_decay_loss.item(),
                }
                print("="*30)
                print("VAL STATS")
                print(msg)
                wandb.log(msg, step=iteration)


    print("Finished.")


if __name__=="__main__":
    main()
