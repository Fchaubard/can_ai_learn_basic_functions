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
# from mongo_wandb import *
import os
os.environ["WANDB_API_KEY"] = ""

# neptune_api_token = ""
# db_url = ""
# db_name="wandb"
# db_project="MeZORNN"

import pdb
import sys
import math
import random
import argparse
import numpy as np
import pynvml
import string
import wandb
import time
import neptune
import copy
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
# VRAM calc
##############################################################################
import torch
from torch.optim import Adam
# Key issues and fixes for VRAM calculation
# def calculate_vram_usage(model, train_micro_batch, x_emb, y_ids, optimizer, criterion, optimizer_type = "sgd",
#                                                                mezo_flavor = None):
#     """
#     Calculate the max VRAM usage of a model during training, with detailed breakdown for different optimizers.

#     Args:
#         model (torch.nn.Module): The model to analyze.
#         train_micro_batch (function): The training function (e.g., train_micro_batch).
#         x_emb (torch.Tensor): Input embeddings for the micro-batch.
#         y_ids (torch.Tensor): Target IDs for the micro-batch.
#         optimizer being used
#         optimizer_type SGD or Mezo
#         mezo_flavor single or layerwise

#     Returns:
#         dict: A dictionary with detailed VRAM usage statistics.
#     """
#     assert torch.cuda.is_available(), "CUDA is required for this calculation."

#     # Detect the current device of the model
#     device = next(model.parameters()).device
    
#     # move the model to cpu for a second
#     model = model.to('cpu')
#     x_emb = x_emb.to('cpu')
#     # Clean up
#     #torch.cuda.empty_cache()
    
#     # Initialize CUDA memory tracking
#     torch.cuda.reset_peak_memory_stats(device)
#     baseline_vram = torch.cuda.memory_allocated(device)  # VRAM used by other processes

#     # move the model back to this gpu
#     model = model.to(device)
#     x_emb = x_emb.to(device)

#     # Measure VRAM for forward pass
#     torch.cuda.reset_peak_memory_stats(device)
#     loss_val = None
    
#     if optimizer_type == "mezo":
        
#         if mezo_flavor == "mezo_layerwise":
#             loss_val = mezo_char_layerwise(model, x_emb, y_ids, criterion, epsilon=0.001)
#         elif mezo_flavor == "mezo_single":
#             loss_val = mezo_char_single(model, x_emb, y_ids, criterion, epsilon=0.001)
#         else:
#             raise Exception("Invalid MeZO flavor")
#         forward_vram = torch.cuda.max_memory_allocated(device) - baseline_vram
#         backward_vram = 0 # there is no backward pass
#     else:
#         # Traditional optimizer training
#         optimizer.zero_grad()
#         loss = teacher_forcing_loss_emb(model, x_emb, y_ids, criterion, backward=True)
#         forward_vram = torch.cuda.max_memory_allocated(device) - baseline_vram

#         # Measure VRAM for backward pass
#         torch.cuda.reset_peak_memory_stats(device)
#         loss.backward()
#         backward_vram = torch.cuda.max_memory_allocated(device) - baseline_vram

#     torch.cuda.reset_peak_memory_stats(device)
#     optimizer.step()  # Perform the step with zero learning rate (no actual update)
#     optimizer.zero_grad()
#     optimizer_vram = torch.cuda.max_memory_allocated(device) - baseline_vram

    
#     # Record peak memory usage across all phases
#     peak_vram = torch.cuda.max_memory_allocated(device) - baseline_vram
    
#     # Clean up
#     #torch.cuda.empty_cache()

#     return {
#         "forward_vram_GB": forward_vram / (1024 ** 3),
#         "backward_vram_GB": backward_vram / (1024 ** 3),
#         "optimizer_vram_GB": optimizer_vram / (1024 ** 3),
#         "total_peak_vram_GB": peak_vram / (1024 ** 3),
#         "max_vram_GB": max(forward_vram,backward_vram,optimizer_vram) / (1024 ** 3),
#     }






##############################################################################
# ASCII Vocab + Tokenizer (with <bos> and <eos>)
##############################################################################

# def get_char_vocab():
#     """
#     Defines a character vocabulary with:
#       - Special tokens: <PAD>, <bos>, <eos>, <UNK>
#       - Digits: 0..9
#       - Uppercase and lowercase letters: A..Z, a..z
#       - Operators: + - * / =
#       - Other symbols: space and '|'
#     """
#     special = ['<PAD>', '<bos>', '<eos>', '<UNK>']  # Add <UNK>
#     digits = list(string.digits)
#     letters_upper = list(string.ascii_uppercase)
#     letters_lower = list(string.ascii_lowercase)
#     operators = ['+', '-', '*', '/', '=', ' ', '|']

#     # Combine all into the vocabulary list
#     vocab_list = special + digits + letters_upper + letters_lower + operators

#     # Create mapping dictionaries
#     char_to_id = {ch: i for i, ch in enumerate(vocab_list)}
#     id_to_char = {i: ch for i, ch in enumerate(vocab_list)}

#     return vocab_list, char_to_id, id_to_char

def get_char_vocab():
    # Special tokens
    special = ['<PAD>', '<bos>', '<eos>', '<UNK>']
    
    # Numbers row (non-shifted and shifted)
    numbers = list('1234567890')
    numbers_shifted = list('!@#$%^&*()')
    
    # Letter rows
    letters = list(string.ascii_lowercase)  # a-z
    letters_upper = list(string.ascii_uppercase)  # A-Z
    
    # Keyboard rows with symbols (non-shifted)
    symbols = list('`-=[]\\;\',./')
    # Shifted versions of those symbols
    symbols_shifted = list('~_+{}|:"<>?')
    
    # Space and tab
    whitespace = [' ', '\t']
    
    # Combine all into vocabulary list
    vocab_list = (numbers + special + numbers_shifted + letters + 
                 letters_upper + symbols + symbols_shifted + whitespace)
    
    # Create mapping dictionaries
    char_to_id = {ch: i for i, ch in enumerate(vocab_list)}
    id_to_char = {i: ch for i, ch in enumerate(vocab_list)}
    
    return vocab_list, char_to_id, id_to_char

def calculate_vram_usage_direct(
    arch: str,
    hidden_size: int,
    memory_size: int,
    head_size: int,
    num_heads: int,
    input_dim: int,
    vocab_size: int,
    micro_batch_size: int,
    input_sample_length: int,
    method: str ,
    use_adam: bool
):
    """
    Estimate VRAM usage (in GB) for a given architecture & training method.
    
    Fixes:
      - MeZO does NOT scale with input_sample_length (since no BPTT).
      - SGD DOES scale with input_sample_length for storing BPTT states.
    
    Args:
      arch (str): One of ['dnc','tra','simplelstm','ntm'].
      hidden_size (int): LSTM/controller hidden size (or transformer's d_model).
      memory_size (int): For NTM/DNC. 
      head_size (int): For NTM/DNC (heads).
      num_heads (int): # of heads or LSTM layers, depending on arch.
      input_dim (int): Dimension of input embeddings.
      vocab_size (int): Output dimension (# classes).
      micro_batch_size (int): Batch size for training.
      input_sample_length (int): Sequence length for training (BPTT).
      method (str): 
        "sgd" => standard backprop 
        "mezo_single" => single-perturbation MeZO
        "mezo_layerwise" => layerwise MeZO
      use_adam (bool): If True, we use Adam's extra states; else simpler momentum.
    
    Returns:
      A dict with:
        'params_gb', 'activations_gb', 'gradients_gb', 'ephemeral_gb', 'optimizer_gb', 'total_estimated_gb'
      summarizing approximate memory usage in GB.
    """

    ################################################################
    # 1) ESTIMATE PARAMETER COUNT
    ################################################################
    if arch == "simplelstm":
        # For each LSTM layer, param_count ~ 4 * hidden_size * (hidden_size + input_dim)
        # times num_layers. Then add some overhead for input->hidden
        param_count_arch = 4 * hidden_size * (hidden_size + input_dim) * num_heads
        activation_scale = 1

    elif arch == "ntm":
        # Roughly hidden_size^2 plus memory ops
        param_count_arch = (hidden_size * hidden_size) + (hidden_size * memory_size * head_size)* num_heads
        activation_scale = param_count_arch
    elif arch == "dnc":
        # Some bigger overhead for memory
        param_count_arch = (hidden_size**2) + 2 * (hidden_size * memory_size * head_size)* num_heads
        activation_scale = param_count_arch
    elif arch == "tra":
        # A rough guess for transformer param: hidden_size^2 * 4 * num_heads
        param_count_arch = 4 * (hidden_size) * num_heads
        activation_scale = num_heads*hidden_size**2
    elif arch == "mamba":
        # Param count for Mamba architecture
        param_count_arch = 4 * (hidden_size * num_heads) + (hidden_size * input_dim)
        activation_scale = hidden_size * num_heads
    else:
        raise Exception(f"No known arch! {arch}") #param_count_arch = hidden_size**2  # fallback if unknown

    # Also consider embedding & final projection
    embed_count = vocab_size * input_dim
    proj_count = hidden_size * vocab_size

    total_params = param_count_arch + embed_count + proj_count

    # param memory in bytes
    params_bytes = total_params * 4
    params_gb = params_bytes / (1024**3)

    ################################################################
    # 2) ACTIVATIONS (Forward) for SGD's BPTT
    ################################################################
    # For standard sgd-based backprop, we store up to 
    # micro_batch_size * input_sample_length * hidden_size
    # (plus some overhead). For mezo => no big BPTT store
    if method == "sgd":
        # scale with B x L x hidden
        # let's say factor=1 
        activation_count = micro_batch_size * input_sample_length * hidden_size * activation_scale
        params_gb*=2 # for adam momentum
    else:
        # MeZO => we do a forward pass or multiple forward passes
        # but do NOT store activations for backprop
        # so we keep it minimal, ignoring L
        activation_count = micro_batch_size * hidden_size  # ignoring input_sample_length
        params_gb*=2 # for adam momentum
        
    # If transformer, add ~some factor
    if arch == "tra":
        activation_count *= 2

    activations_bytes = activation_count * 4
    activations_gb = activations_bytes / (1024**3)

    ################################################################
    # 3) GRADIENT VRAM
    ################################################################
    # If we do BPTT (sgd method), we store gradient for each param => ~ param_count
    # MeZO => no param grad for entire L
    if method == "sgd":
        gradients_bytes = total_params * 4
    else:
        gradients_bytes = 0
    gradients_gb = gradients_bytes / (1024**3)

    ################################################################
    # 4) EPHEMERAL VRAM for multiple forward passes
    ################################################################
    #  - mezo_single => do 2 forward passes => ephemeral not scaling with L
    #  - mezo_layerwise => multiple passes but each "layer" => also not scaling with L
    #  - sgd => ephemeral = 0 here, because we've accounted for BPTT in "activations_gb"
    ephemeral_bytes = 0
    if method == "mezo_single":
        # e.g. 2 forward passes => add 1 more chunk of (B * hidden_size) 
#         ephemeral_count = micro_batch_size * hidden_size  # ignoring L
#         ephemeral_count *= 1  # (we already included 1 pass in "activations_gb")
#         ephemeral_bytes = ephemeral_count * 4
        activations_gb/=1

    elif method == "mezo_layerwise":
        # we do multiple forward passes, but only 1 layer at a time => 
        # let's do half or so 
        activations_gb/=4
#         ephemeral_count = micro_batch_size * hidden_size / 40
#         ephemeral_bytes = ephemeral_count * 4

    ephemeral_gb = ephemeral_bytes / (1024**3)

    ################################################################
    # 5) OPTIMIZER STATES
    ################################################################
    # If use_adam => ~ 2 param_count for moment1, moment2
    # If not => maybe a small factor for momentum
    if use_adam:
        opt_bytes = total_params * 4 * 2
    else:
        # smaller overhead for simpler momentum
        opt_bytes = total_params * 4 * 2  # arbitrary factor
#         if method.startswith("mezo"):
#             # mezo + sgd => maybe even less overhead
#             opt_bytes *= 0.5

    optimizer_gb = opt_bytes / (1024**3)

    ################################################################
    # 6) SUM
    ################################################################
    total_estimated_gb = (
        params_gb + 
        activations_gb + 
        gradients_gb + 
        ephemeral_gb + 
        optimizer_gb
    )

    return {
        "params_gb": params_gb,
        "activations_gb": activations_gb,
        "gradients_gb": gradients_gb,
        "ephemeral_gb": ephemeral_gb,
        "optimizer_gb": optimizer_gb,
        "total_estimated_gb": total_estimated_gb
    }


##############################################################################
# Convert strings -> fixed [B, max_seq_len], pad or truncate
##############################################################################
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



def str_to_tensor(batch_strs, char_to_id):
    """
    Convert a batch of strings (which may contain special tokens like <bos>, <eos>, etc.)
    into a padded LongTensor of shape [B, max_len].

    1) For each string, call `tokenize_special(...)` to produce a list of tokens.
    2) Find the maximum sequence length across the batch.
    3) Create a tensor [B, max_len] of zeros (padding index=0).
    4) Fill in the token IDs for each sequence up to its length.
    """
    # 1) Tokenize each string
    token_lists = []
    for s in batch_strs:
        tokens = tokenize_special(s)
        token_lists.append(tokens)

    # 2) Find the maximum length across all tokenized strings
    max_len = max(len(toks) for toks in token_lists) if token_lists else 1

    # 3) Create the output tensor [B, max_len], initialized to 0 (PAD)
    B = len(batch_strs)
    out = torch.zeros(B, max_len, dtype=torch.long)

    # 4) Fill in the token IDs
    for i, tokens in enumerate(token_lists):
        for j, tok in enumerate(tokens):
            out[i, j] = char_to_id.get(tok, char_to_id["<UNK>"])

    return out

def tokenize_special(s):
    """
    Example tokenizer that treats any substring like <bos>, <eos>, <PAD>, etc.
    as *single* tokens if present. Otherwise, each character is its own token.

    This is just a toy example. Adjust it to fit your needs (e.g. handling
    <bos>... in the input more robustly).
    """
    tokens = []
    i = 0
    while i < len(s):
        if s[i] == '<':
            # Try to find the matching '>'
            j = s.find('>', i + 1)
            if j == -1:
                # No '>' found => treat '<' as a normal character (unlikely in well-formed data).
                tokens.append(s[i])
                i += 1
            else:
                # Collect the full '<...>'
                tokens.append(s[i : j+1])
                i = j + 1
        else:
            # Normal character
            tokens.append(s[i])
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
import random
from datasets import load_dataset


# OWT open web text 
def generate_openwebtext_task_str(
    num_samples: int,
    context_length: int,
    ds,
    max_n=None,   # unused in OWT, but we keep the signature to match the others
    train: bool = True,
    min_total_seq_len=100,
    vebose=False
):
    split_name = "train" if train else "validation"

    size_of_ds = len(ds)
    in_list = []
    out_list = []
    
    while len(in_list) < num_samples:
        # Keep trying docs until we find one that meets our length requirement
        max_tries = 100  # Avoid infinite loop
        tries = 0
        valid_doc = False
        
        while not valid_doc and tries < max_tries:
            doc_idx = random.randint(0, size_of_ds - 1)
            doc = ds[doc_idx]
            text = doc["text"] if doc["text"] else ""
            
            # Calculate total sequence length including special tokens
            # Length will be: len(<bos> + prefix) + len(remainder + <eos>)
            # = 5 + len(text) [5 comes from <bos> and <eos>]
            total_seq_len = len(text) + 5  # +5 for <bos> and <eos>
            
            if total_seq_len >= min_total_seq_len and total_seq_len<20000: # crazy large number
                valid_doc = True
            else:
                tries += 1
        
        # If we couldn't find a long enough doc after max_tries,
        # pad the last one we found
        if not valid_doc:
            needed = min_total_seq_len - total_seq_len
            text = text + (" " * needed)
            
        # Now ensure text is at least context_length
        if len(text) < context_length:
            needed = context_length - len(text)
            text = text + (" " * needed)
            
        # Split into prefix and remainder
        prefix = text[:context_length]
        remainder = text[context_length:min_total_seq_len] # HACKY! TODO TO CLEAN UP BUT ONLY THING I CAN DO TO MAKE NTM AND DNC FASTER
        
        # Create input and target strings
        input_str = f"<bos>{prefix}"
        target_str = f"{remainder}<eos>"
        
        in_list.append(input_str)
        out_list.append(target_str)
        
    # Debug prints
    max_out_len = max(len(i) for i in out_list)
    print(f"   Max lengths: max={max_out_len}")

    if vebose:
        min_in_len = min(len(i) for i in in_list)
        min_out_len = min(len(i) for i in out_list)
        max_in_len = max(len(i) for i in in_list)
        max_out_len = max(len(i) for i in out_list)
    
        print(f"Sequence length stats:")
        print(f"Input lengths: min={min_in_len}, max={max_in_len}")
        print(f"Output lengths: min={min_out_len}, max={max_out_len}")
        print(f"Min total lengths: {[len(i) + len(j) for i,j in zip(in_list, out_list)]}")

    return in_list, out_list



def generate_reverse_task_str(num_samples, input_sample_length, train=True):
    """
    Generates data for the reverse task.
    
    Args:
        num_samples (int): Number of samples to generate.
        input_sample_length (int): Length of each input sample.
        train (bool): Whether the samples are for training or testing.
                      If not train, input_sample_length is scaled up by a factor of 5.
    
    Returns:
        tuple: (input strings, reversed output strings)
    """
    if not train:
        input_sample_length *= 5

    letters = string.ascii_uppercase
    in_list, out_list = [], []

    for _ in range(num_samples):
        data_str = "".join(random.choice(letters) for _ in range(input_sample_length))
        xinp = data_str
        xtgt = data_str[::-1]  # Reverse the string
        in_list.append(xinp)
        out_list.append(xtgt)

    return in_list, out_list

def generate_sort_task_str(num_samples, input_sample_length, train=True):
    """
    Generates data for the sort task.
    
    Args:
        num_samples (int): Number of samples to generate.
        input_sample_length (int): Length of each input sample.
        train (bool): Whether the samples are for training or testing.
                      If not train, input_sample_length is scaled up by a factor of 5.
    
    Returns:
        tuple: (input strings, sorted output strings)
    """
    if not train:
        input_sample_length *= 5

    letters = string.ascii_uppercase
    in_list, out_list = [], []

    for _ in range(num_samples):
        data_str = "".join(random.choice(letters) for _ in range(input_sample_length))
        xinp = data_str
        xtgt = "".join(sorted(data_str))  # Sort the string lexicographically
        in_list.append(xinp)
        out_list.append(xtgt)

    return in_list, out_list

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
# class SimpleLSTM(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size, embed):
#         super(SimpleLSTM, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.hidden_size = hidden_size
#         self.embed = embed

#         # Input normalization
#         controller_input_size = input_size 
#         self.input_norm = nn.LayerNorm(controller_input_size)
#         self.controller_norm = nn.LayerNorm(output_size)

#         # Controller with input normalization
#         self.controller = nn.LSTM(controller_input_size, hidden_size, proj_size=output_size, num_layers=1, batch_first=True)

#     def forward(self, x_emb, hidden=None, memory=None, skip_layer_norm=True):
#         B, L, E = x_emb.size()

#         # Correct hidden and cell state initialization
#         if hidden is None:
#             h0 = x_emb.new_zeros(1, B, self.output_size)  # Hidden state matches proj_size
#             c0 = x_emb.new_zeros(1, B, self.hidden_size)  # Cell state matches hidden_size
#             hidden = (h0, c0)

#         outputs = []
#         for t in range(L):
#             controller_input = x_emb[:, t, :]
#             if not skip_layer_norm:
#                 controller_input = self.input_norm(controller_input)

#             # Controller
#             out_ctrl, hidden = self.controller(controller_input.unsqueeze(1), hidden)
#             # out_ctrl = out_ctrl.squeeze(1)
#             # Normalize output
#             if not skip_layer_norm:
#                 out_ctrl = self.controller_norm(out_ctrl)
#             outputs.append(out_ctrl)

#         outputs = torch.cat(outputs, dim=1)
#         return outputs, memory, hidden
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, embed, num_layers=1):
        """
        Simple LSTM model with a separate projection layer.

        Args:
            input_size (int): Input feature size.
            output_size (int): Desired output feature size.
            hidden_size (int): Hidden size of the LSTM.
            embed (nn.Module): Embedding layer.
            num_layers (int): Number of LSTM layers (default=1).
        """
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = embed

        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)

        # Multi-layer LSTM
        self.controller = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )

        # Projection layer to map hidden_size -> output_size
        self.projection = nn.Linear(hidden_size, output_size)

    def forward(self, x_emb, hidden=None,  memory=None, skip_layer_norm=True):
        """
        Forward pass of the LSTM.

        Args:
            x_emb (torch.Tensor): Input embeddings [batch_size, seq_len, input_size].
            hidden (tuple): Tuple of (h_0, c_0) for LSTM layers.
            skip_layer_norm (bool): Whether to skip layer normalization.

        Returns:
            torch.Tensor: Projected output [batch_size, seq_len, output_size].
            None: Placeholder for memory (not used in this implementation).
            tuple: Updated (hidden, cell) states.
        """
        B, L, E = x_emb.size()

        # Apply input layer normalization if not skipped
        if not skip_layer_norm:
            x_emb = self.input_norm(x_emb)

        # Initialize hidden states if not provided
        if hidden is None:
            h0 = x_emb.new_zeros(self.num_layers, B, self.hidden_size)
            c0 = x_emb.new_zeros(self.num_layers, B, self.hidden_size)
            hidden = (h0, c0)

        # Forward pass through LSTM
        lstm_out, hidden = self.controller(x_emb, hidden)  # lstm_out: [B, L, hidden_size]

        # Project LSTM outputs to desired output_size
        projected_out = self.projection(lstm_out)  # [B, L, output_size]

        return projected_out, None, hidden




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
        self.temperature = 4.0

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
        read_weights = F.softmax(read_weights/self.temperature, dim=-1)
        read_content = torch.einsum('bnm,bmh->bnh', read_weights, memory)
    
        # Write operation with normalized attention
        write_weights = torch.einsum('bnk,bmk->bnm', write_keys, memory_normalized)
        write_weights = F.softmax(write_weights/self.temperature, dim=-1)
    
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



# class TransformerBlock(nn.Module):
#     def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = nn.GELU()

#     def forward(self, x, key_padding_mask=None):
#         attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
#         x = self.norm1(x + self.dropout(attn_out))
#         ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         x = self.norm2(x + self.dropout(ff_out))
#         return x

# class TransformerMemoryNTM(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size, memory_size, head_size, num_heads, embed):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.memory_size = memory_size
#         self.head_size = head_size
#         self.embed = embed

#         # Input normalization
#         self.input_norm = nn.LayerNorm(input_size + num_heads * head_size)

#         # Keep LSTM controller
#         self.controller = nn.LSTM(input_size + num_heads * head_size, hidden_size, batch_first=True)
#         self.controller_norm = nn.LayerNorm(hidden_size)

#         # Memory operation layers
#         self.fc_read_keys = nn.Linear(hidden_size, num_heads * head_size)
#         self.fc_write_keys = nn.Linear(hidden_size, num_heads * head_size)
#         self.fc_write_strength = nn.Linear(hidden_size, num_heads)
#         self.fc_erase_vector = nn.Linear(hidden_size, num_heads * head_size)

#         # Add transformer for memory processing
#         self.memory_transformer = TransformerBlock(head_size, nhead=4, dim_feedforward=2*head_size)
        
#         # Normalization layers
#         self.read_keys_norm = nn.LayerNorm(head_size)
#         self.write_keys_norm = nn.LayerNorm(head_size)
#         self.memory_norm = nn.LayerNorm(head_size)
        
#         # Output layers
#         total_output_size = hidden_size + num_heads * head_size
#         self.pre_output_norm = nn.LayerNorm(total_output_size)
#         self.fc_proj = nn.Linear(total_output_size, output_size)

#         self.reset_parameters()

#     def reset_parameters(self):
#         """Initialize parameters"""
#         # Initialize LSTM params
#         for name, p in self.controller.named_parameters():
#             if 'weight' in name:
#                 nn.init.orthogonal_(p)
#             elif 'bias' in name:
#                 nn.init.constant_(p, 0)

#         # Initialize memory operation layers
#         for name, p in self.named_parameters():
#             if 'fc_' in name and 'weight' in name:
#                 nn.init.uniform_(p, -0.1, 0.1)
#             elif 'fc_' in name and 'bias' in name:
#                 nn.init.constant_(p, 0)

#     def _addressing(self, memory, read_keys, write_keys, write_strengths, erase_vectors):
#         """Memory addressing with transformer-enhanced memory"""
#         B, N, W = memory.size()

#         # Transform memory using transformer
#         memory = self.memory_transformer(memory)
        
#         # Normalize memory and keys
#         memory_normalized = self.memory_norm(memory)
#         read_keys = self.read_keys_norm(read_keys.view(-1, W)).view(B, self.num_heads, W)
#         write_keys = self.write_keys_norm(write_keys.view(-1, W)).view(B, self.num_heads, W)
    
#         # Read operation
#         read_weights = torch.einsum('bnk,bmk->bnm', read_keys, memory_normalized)
#         read_weights = F.softmax(read_weights, dim=-1)
#         read_content = torch.einsum('bnm,bmh->bnh', read_weights, memory)
    
#         # Write operation
#         write_weights = torch.einsum('bnk,bmk->bnm', write_keys, memory_normalized)
#         write_weights = F.softmax(write_weights, dim=-1)
        
#         # Memory update
#         erase_vectors_expanded = torch.einsum('bnm,bnh->bmh', write_weights, erase_vectors)
#         memory = memory * (1 - erase_vectors_expanded)
    
#         add_content = write_strengths.unsqueeze(-1) * write_keys
#         add_content_expanded = torch.einsum('bnm,bnh->bmh', write_weights, add_content)
#         memory = memory + add_content_expanded
    
#         return memory, read_content, read_weights, write_weights

#     def forward(self, x_emb, hidden=None, memory=None):
#         B, L, E = x_emb.size()

#         # Initialize states if needed
#         if hidden is None:
#             h0 = x_emb.new_zeros(1, B, self.hidden_size)
#             c0 = x_emb.new_zeros(1, B, self.hidden_size)
#             hidden = (h0, c0)

#         if memory is None:
#             memory = x_emb.new_zeros(B, self.memory_size, self.head_size)

#         outputs = []
#         read_contents = x_emb.new_zeros(B, self.num_heads, self.head_size)

#         for t in range(L):
#             # LSTM controller
#             inp_t = torch.cat([x_emb[:, t, :], read_contents.view(B, -1)], dim=-1)
#             inp_t = self.input_norm(inp_t)
#             out_ctrl, hidden = self.controller(inp_t.unsqueeze(1), hidden)
#             h = self.controller_norm(out_ctrl.squeeze(1))

#             # Memory operations
#             read_keys = self.fc_read_keys(h).view(B, self.num_heads, self.head_size)
#             write_keys = self.fc_write_keys(h).view(B, self.num_heads, self.head_size)
#             write_strengths = torch.sigmoid(self.fc_write_strength(h)).view(B, self.num_heads)
#             erase_vectors = torch.sigmoid(self.fc_erase_vector(h)).view(B, self.num_heads, self.head_size)

#             memory, read_contents, _, _ = self._addressing(
#                 memory, read_keys, write_keys, write_strengths, erase_vectors
#             )

#             # Output projection
#             output = torch.cat([h, read_contents.view(B, -1)], dim=-1)
#             output = self.pre_output_norm(output)
#             logits = self.fc_proj(output)
#             outputs.append(logits.unsqueeze(1))

#         outputs = torch.cat(outputs, dim=1)
#         return outputs, memory, hidden


# class TransformerMemoryDNC(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size, memory_size, head_size, num_heads, embed):
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.hidden_size = hidden_size
#         self.memory_size = memory_size
#         self.head_size = head_size
#         self.num_reads = num_heads
#         self.embed = embed

#         # Input normalization
#         controller_input_size = input_size + self.num_reads * head_size
#         self.input_norm = nn.LayerNorm(controller_input_size)

#         # Keep LSTM controller
#         self.controller = nn.LSTM(controller_input_size, hidden_size, batch_first=True)
#         self.controller_norm = nn.LayerNorm(hidden_size)

#         # Memory operation layers
#         self.fc_read_keys = nn.Linear(hidden_size, self.num_reads * head_size)
#         self.fc_write_keys = nn.Linear(hidden_size, head_size)
#         self.fc_write_strength = nn.Linear(hidden_size, 1)
#         self.fc_erase_vector = nn.Linear(hidden_size, head_size)
#         self.fc_add_vector = nn.Linear(hidden_size, head_size)

#         # Add transformers for memory processing
#         self.memory_transformer = TransformerBlock(head_size, nhead=4, dim_feedforward=2*head_size)
#         self.read_transformer = TransformerBlock(head_size, nhead=4, dim_feedforward=2*head_size)
        
#         # Normalization layers
#         self.read_keys_norm = nn.LayerNorm(head_size)
#         self.write_keys_norm = nn.LayerNorm(head_size)
#         self.memory_norm = nn.LayerNorm(head_size)

#         # Output layers
#         total_output_size = hidden_size + self.num_reads * head_size
#         self.pre_output_norm = nn.LayerNorm(total_output_size)
#         self.fc_proj = nn.Linear(total_output_size, output_size)

#         self.reset_parameters()
#         self.temperature = 4.0

#     def reset_parameters(self):
#         """Initialize parameters"""
#         # Initialize LSTM params
#         for name, p in self.controller.named_parameters():
#             if 'weight' in name:
#                 nn.init.orthogonal_(p)
#             elif 'bias' in name:
#                 nn.init.constant_(p, 0)

#         # Initialize memory operation layers
#         for name, p in self.named_parameters():
#             if 'fc_' in name and 'weight' in name:
#                 nn.init.uniform_(p, -0.1, 0.1)
#             elif 'fc_' in name and 'bias' in name:
#                 nn.init.constant_(p, 0)

#     def _read_memory(self, memory, read_keys):
#         """Enhanced memory reading with transformer processing"""
#         # Transform memory
#         memory = self.memory_transformer(memory)
        
#         # Process read keys with transformer
#         read_keys = self.read_transformer(read_keys.view(-1, self.head_size).unsqueeze(1)).squeeze(1)
        
#         # Normalize memory and keys
#         memory_normalized = self.memory_norm(memory)
#         read_keys = self.read_keys_norm(read_keys).view(-1, self.num_reads, self.head_size)

#         # Compute attention weights
#         read_weights = torch.softmax(
#             torch.einsum('bnh,bmh->bnm', read_keys, memory_normalized)/self.temperature,
#             dim=2
#         )
#         read_vectors = torch.einsum('bnm,bmh->bnh', read_weights, memory)
#         return read_vectors

#     def _write_memory(self, memory, write_keys, write_str, erase_vec, write_vec):
#         """Enhanced memory writing with transformer processing"""
#         # Transform memory
#         memory = self.memory_transformer(memory)
        
#         # Normalize memory and keys
#         memory_normalized = self.memory_norm(memory)
#         write_keys = self.write_keys_norm(write_keys)

#         # Compute write weights
#         write_weights = torch.softmax(
#             torch.einsum('bh,bmh->bm', write_keys, memory_normalized)/self.temperature,
#             dim=1
#         ).unsqueeze(1)

#         # Scale by write strength
#         write_weights = write_weights * write_str.unsqueeze(1)

#         # Update memory
#         erase = torch.einsum('bnm,bh->bmh', write_weights, erase_vec)
#         write = torch.einsum('bnm,bh->bmh', write_weights, write_vec)
#         memory = memory * (1 - erase) + write
        
#         return memory

#     def forward(self, x_emb, hidden=None, memory=None):
#         B, L, E = x_emb.size()
#         device = x_emb.device

#         # Initialize states if needed
#         if hidden is None:
#             h0 = x_emb.new_zeros(1, B, self.hidden_size)
#             c0 = x_emb.new_zeros(1, B, self.hidden_size)
#             hidden = (h0, c0)

#         if memory is None:
#             memory = x_emb.new_zeros(B, self.memory_size, self.head_size)

#         read_vec = x_emb.new_zeros(B, self.num_reads * self.head_size)
#         outputs = []

#         for t in range(L):
#             # LSTM controller
#             inp_t = torch.cat([x_emb[:, t, :], read_vec], dim=-1)
#             inp_t = self.input_norm(inp_t)
#             out_ctrl, hidden = self.controller(inp_t.unsqueeze(1), hidden)
#             h = self.controller_norm(out_ctrl.squeeze(1))

#             # Memory parameters
#             read_keys = self.fc_read_keys(h).view(B, self.num_reads, self.head_size)
#             write_keys = self.fc_write_keys(h)
#             write_str = torch.sigmoid(self.fc_write_strength(h))
#             erase_vec = torch.sigmoid(self.fc_erase_vector(h))
#             write_vec = torch.tanh(self.fc_add_vector(h))

#             # Memory operations with transformer enhancement
#             memory = self._write_memory(memory, write_keys, write_str, erase_vec, write_vec)
#             read_vectors = self._read_memory(memory, read_keys)
#             read_vec = read_vectors.reshape(B, -1)

#             # Output projection
#             output = torch.cat([h, read_vec], dim=-1)
#             output = self.pre_output_norm(output)
#             logits = self.fc_proj(output)
#             outputs.append(logits.unsqueeze(1))

#         outputs = torch.cat(outputs, dim=1)
#         return outputs, memory, hidden





# ############
# # MAMBA
# #############
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class MambaMultiheadSelfAttention(nn.Module):
    """
    Basic multi-head self-attention without relying on PyTorch's nn.MultiheadAttention.
    """
    def __init__(self, d_model, nhead=4, dropout=0.1):
        """
        Args:
            d_model (int): Dimensionality of tokens/features (total).
            nhead (int): Number of attention heads.
            dropout (float): Dropout rate for attention.
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead  # must divide evenly or handle leftover
        assert d_model % nhead == 0, f"d_model={d_model} must be divisible by nhead={nhead}."

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Final out projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: [B, L, d_model]
        mask: optional
        Returns: attn_out [B, L, d_model]
        """
        B, L, _ = x.shape

        # 1) Q,K,V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2) Reshape => [B, nhead, L, head_dim]
        Q = Q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        # 3) Scaled Dot-Product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            # e.g. scores = scores.masked_fill(mask == 0, float('-inf'))
            pass

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        attn_out = torch.matmul(probs, V)   # => [B, nhead, L, head_dim]

        # 4) Merge heads => [B, L, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)

        # 5) Final linear
        attn_out = self.out_proj(attn_out)
        return attn_out


class MambaEncoderBlock(nn.Module):
    """
    One Transformer encoder block: Self-Attn -> FFN, with residual + layernorm.
    """
    def __init__(self, d_model=128, nhead=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = MambaMultiheadSelfAttention(d_model, nhead, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 1) Self-Attn
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # 2) FFN
        ffn_out = self.ffn(x)
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)

        return x


class MambaEncoder(nn.Module):
    """
    A stack of MambaEncoderBlocks
    """
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


# class Mamba(nn.Module):
#     """
#     A minimal Transformer encoder that uses Mamba-based multihead self-attn,
#     but automatically adjusts head_size if there's a mismatch.
#     """
#     def __init__(self, input_size, output_size, hidden_size, memory_size, head_size, num_heads, embed):
#         """
#         Args:
#             input_size  (int): model dimension (d_model).
#             output_size (int): final projection dimension (e.g. vocab size).
#             hidden_size (int): feedforward dimension in each encoder block.
#             memory_size (int): not used in the forward pass, included for interface only.
#             head_size   (int): user-desired dimension per head.
#             num_heads   (int): number of attention heads.
#             embed       (nn.Embedding): embedding layer.
#         """
#         super().__init__()
#         self.embed = embed
#         self.input_size = input_size
#         self.output_size = output_size
#         self.hidden_size = hidden_size
#         self.memory_size = memory_size
#         self.num_heads = num_heads

#         # If user gave a mismatch, override head_size so that head_size * num_heads == input_size
#         if head_size * num_heads != input_size:
#             # automatically fix if possible
#             new_head_size = input_size // num_heads
#             if input_size % num_heads != 0:
#                 # fallback approach: pick num_heads=1 if we can't do an even split
#                 # or raise an error, your choice:
#                 raise ValueError(f"Cannot evenly distribute input_size={input_size} among num_heads={num_heads}.\n"
#                                  f"Either adjust input_size or num_heads so they match, or set head_size accordingly.")
#             print(f"[Mamba] Overriding head_size from {head_size} -> {new_head_size} so that {new_head_size} * {num_heads} = {input_size}")
#             head_size = new_head_size

#         # Build the encoder
#         self.encoder = MambaEncoder(
#             d_model=input_size,
#             nhead=num_heads,
#             num_layers=2,                 # or parameterize further
#             dim_feedforward=hidden_size,
#             dropout=0.1
#         )

#         self.fc_out = nn.Linear(input_size, output_size)

#     def forward(self, x_emb, hidden=None, memory=None):
#         """
#         x_emb: [B, L, input_size]
#         hidden, memory: placeholders for interface parity
#         Returns: out: [B, L, output_size], None, (None, None)
#         """
#         enc_out = self.encoder(x_emb)
#         out = self.fc_out(enc_out)
#         return out, None, (None, None)




############
# DNC
##############
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







###### NEW MAMBA AND TRANFORMER CLASSES 
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    """
    A single-layer (or minimal) Transformer for causal LM.
    Feeds entire sequence, uses create_causal_mask, 
    slices out the 'context_length' portion from the loss if asked.
    """
    def __init__(self, input_size, output_size, hidden_size, memory_size, head_size, num_heads, embed):
        super().__init__()
        self.embed = embed

        # We'll define d_model as num_heads * head_size
        # so it is divisible by num_heads
        self.d_model = num_heads * head_size
        
        # A small projection from input_size -> d_model
        self.input_proj = nn.Linear(input_size, self.d_model, bias=False)
        
        # Single MultiheadAttention w/ batch_first = True
        self.attention = nn.MultiheadAttention(self.d_model, num_heads, 
                                               dropout=0.1, 
                                               batch_first=True)
        self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.layer_norm2 = nn.LayerNorm(self.d_model)

        # Simple feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.GELU(),
            nn.Linear(4 * self.d_model, self.d_model),
        )

        # Final projection to vocab
        self.fc_out = nn.Linear(self.d_model, output_size, bias=True)

    def create_causal_mask(self, seq_len, device):
        """
        Build a float mask [seq_len, seq_len] with -inf above the diagonal
        so that position i cannot attend to positions j > i.
        0.0 on the diagonal and below => allowed, 
        -inf above diagonal => blocked.
        """
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)  # fill upper triangle w/ -inf
        return mask

    def forward(
        self,
        x_emb,                 # [B, L] long, or [B, L, E] if you embed outside
        hidden=None,            # to conform to other model APIs but ignored
        memory=None,            # to conform to other model APIs but ignored
        context_length=None     # optional int to indicate how many prompt tokens to ignore in the loss
    ):
        """
        If you are embedding outside the model, pass shape [B,L,E]. 
        Otherwise, pass tokens [B,L], we do self.embed inside.

        We'll do a single pass with a causal mask.
        `context_length` is not used inside forward to slice out tokens 
        (some folks prefer to do that logic outside). 
        It's here only if you want to do the slicing inside the model. 
        """
        
        
        B, L, E = x_emb.shape
        device = x_emb.device

        # Project input to Transformer dimension
        x_proj = self.input_proj(x_emb)  # [B, L, d_model]

        # Create causal mask
        attn_mask = self.create_causal_mask(L, device=device)  # [L, L]

        # Self-attention
        attn_out, _ = self.attention(x_proj, x_proj, x_proj, 
                                     attn_mask=attn_mask, 
                                     need_weights=False)
        # Residual + norm
        x = self.layer_norm1(x_proj + attn_out)

        # Feed-forward
        ff_out = self.ffn(x)
        x = self.layer_norm2(x + ff_out)

        # Final projection to vocab
        logits = self.fc_out(x)  # [B, L, vocab_size]

        return logits, None, None

    @torch.no_grad()
    def generate(
        self,
        prompt_ids,        # [B, prompt_len], e.g. token IDs of the prompt
        max_new_tokens=50,
        temperature=1.0,
        stop_id=None
    ):
        """
        Autoregressive generation. Each iteration:
          - embed the entire sequence so far
          - forward pass with causal mask
          - pick the last token's distribution
          - sample or argmax next token
          - append to sequence
        """
        B = prompt_ids.size(0)
        generated = prompt_ids.clone()  # copy
        for _ in range(max_new_tokens):
            # 1) forward on the entire sequence so far
            logits, _, _ = self(generated)  # shape [B, seq_so_far, vocab]
            next_logits = logits[:, -1, :] / temperature
            # 2) greedy or sample
            # let's do greedy for simplicity
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # [B,1]
            # 3) append
            generated = torch.cat([generated, next_token], dim=1)
            # 4) optional stop if the next_token == stop_id
            if stop_id is not None:
                if (next_token == stop_id).all():
                    break
        return generated  # shape [B, prompt_len + new_tokens]




class Mamba(nn.Module):
    """
    A 2-layer "Mamba" encoder using a custom multi-head self-attention,
    again with a full lower-triangular mask for causality.
    """
    def __init__(self, input_size, output_size, hidden_size, memory_size, head_size, num_heads, embed):
        super().__init__()
        self.embed = embed
        self.d_model = num_heads * head_size
        self.input_proj = nn.Linear(input_size, self.d_model, bias=False)

        # We'll just do 2 MambaEncoderBlocks for demonstration
        self.layers = nn.ModuleList([
            MambaEncoderBlock(d_model=self.d_model,
                              nhead=num_heads,
                              dim_feedforward=hidden_size,
                              dropout=0.1)
            for _ in range(2)
        ])

        self.norm_final = nn.LayerNorm(self.d_model)
        self.fc_out = nn.Linear(self.d_model, output_size)

    def create_causal_mask(self, seq_len, device):
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, 
                x_emb, 
                hidden=None,            # to conform to other model APIs but ignored
                memory=None,            # to conform to other model APIs but ignored
                context_length=None):
        """
        tokens: [B,L] or [B,L,E].
        """

        B, L, E = x_emb.shape
        device = x_emb.device

        x = self.input_proj(x_emb)

        # Build causal mask
        attn_mask = self.create_causal_mask(L, device=device)

        # Pass through each MambaEncoderBlock
        for layer in self.layers:
            x = layer(x, mask=attn_mask)

        x = self.norm_final(x)
        logits = self.fc_out(x)  # [B, L, vocab_size]

        return logits, None, None

    @torch.no_grad()
    def generate(
        self,
        prompt_ids,
        max_new_tokens=50,
        temperature=1.0,
        stop_id=None
    ):
        B = prompt_ids.size(0)
        generated = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits, _, _ = self(generated)
            next_logits = logits[:, -1, :] / temperature
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if stop_id is not None:
                if (next_token == stop_id).all():
                    break
        return generated





# class TransformerController(nn.Module):
#     def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0):
#         super(TransformerController, self).__init__()
#         encoder_layer= nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
#                                                   dim_feedforward=dim_feedforward,
#                                                   dropout=dropout, batch_first=True)
#         self.encoder= nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#     def forward(self, x):
#         return self.encoder(x)


# class Transformer(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size, memory_size, head_size, num_heads, embed):
#         super(Transformer, self).__init__()
#         self.embed = embed
        
#         # Project input to a dimension that's compatible with num_heads
#         # Make d_model a multiple of num_heads
#         self.d_model = num_heads * head_size  # This ensures d_model is divisible by num_heads
#         self.input_proj = nn.Linear(input_size, self.d_model)
        
#         self.transformer = TransformerController(
#             d_model=self.d_model,  # Use the projected dimension
#             nhead=num_heads,
#             num_layers=1,
#             dim_feedforward=4 * self.d_model  # Scale with d_model instead of input_size
#         )
#         self.fc_out = nn.Linear(self.d_model, output_size)
        
#     def forward(self, x_emb, hidden=None, memory=None):
#         # Project input to transformer dimension
#         x_proj = self.input_proj(x_emb)
        
#         # Pass through transformer
#         trans_out = self.transformer(x_proj)
        
#         # Project to output size
#         out = self.fc_out(trans_out)
#         return out, None, (None, None)

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


# def teacher_forcing_loss_emb(model, x_emb, y_ids_unpadded, criterion, teacher_force=True):
#     """
#     Teacher-forcing loss for step-by-step sequence generation.
    
#     Args:
#         model (nn.Module): The sequence-to-sequence model.
#         x_emb (torch.Tensor): [B, Lx, E], embedded input sequence.
#         y_ids (torch.LongTensor): [B, Ly], token IDs for the target sequence.
#         criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).

#     Returns:
#         torch.Tensor: Average loss per token.
#     """
#     B, Lx, E = x_emb.size()
#     # y_ids_trimmed = [y[y != 0] for y in y_ids]
#     # y_ids_unpadded = pad_sequence(y_ids_trimmed, batch_first=True, padding_value=0)

#     Ly = y_ids_unpadded.size(1)
#     device = x_emb.device


#     # Initialize hidden states
#     hidden = None
#     memory = None

#     y_emb_input = model.embed(y_ids_unpadded)[:, :-1, :]  # shape [B, L_y-1, E]

#     # Concatenate input and partial target
#     # e.g. shape [B, L_x + (L_y-1), E]
#     full_input = torch.cat([x_emb, y_emb_input], dim=1)
    
#     # Now feed this entire sequence to the LSTM "in one shot"
#     outputs, _, (h, c) = model(full_input)  # shape of outputs is [B, L_x + (L_y - 1), something]
#     logits = outputs[:, Lx-1:, :].contiguous()  # Get predictions starting from after input sequence
#     logits = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, num_classes]
    
#     # Reshape targets
#     targets = y_ids_unpadded.contiguous().view(-1)  # [batch_size * seq_len]
    
#     total_loss = criterion(logits, targets)
#     return total_loss/Ly 


    # # # Forward input sequence step-by-step
    # # for t in range(Lx):
    # #     inp_t = x_emb[:, t, :].unsqueeze(1)  # [B, 1, E]
    # #     logits, memory, hidden = model(inp_t, hidden=hidden, memory=memory)

    # # Now process target sequence step-by-step
    # total_loss = 0.0
    # num_tokens = 0

    # for t in range(Ly):  
    #     logits = logits.squeeze(1)  # [B, vocab_size]
    #     # Determine whether to continue teacher forcing
    #     pred_t = logits.argmax(dim=-1)  # Predicted token IDs
    #     counter=0
    #     while pred_t.eq(0).all() and counter>10: # its allowed to think for 10 iters per token.. after that incur loss
    #         counter+=1
    #         inp_t = model.embed(pred_t).unsqueeze(1)
    #         logits, memory, hidden = model(inp_t, hidden=hidden, memory=memory)
    #         logits = logits.squeeze(1)  # [B, vocab_size]
    #         # Determine whether to continue teacher forcing
    #         pred_t = logits.argmax(dim=-1)  # Predicted token IDs
        
    #     # pdb.set_trace()
    #     # we made our prediction for this t so lets get some loss
    #     target_t = y_ids_unpadded[:, t ]  # Ground truth at step t+1 (excluding <bos>)
    #     step_loss = criterion(logits/4.0, target_t)
    #     total_loss += step_loss #.item()
    #     # correct += (target_t == pred_t).sum().item()
    #     num_tokens += (target_t != 0).sum().item()

    #     if not teacher_force: # if we are not teacher forcing, we should feed its own output back in
    #         target_t = pred_t
        
  
    #     # now we take the next truth and give it to the model and iter again
    #     target_t_emb = model.embed(target_t).unsqueeze(1)  # [B, 1, E]
  
    #     # Generate prediction for the current step
    #     logits, memory, hidden = model(target_t_emb, hidden=hidden, memory=memory) # TODO, hidden may need a transpose?

    # Average loss across all valid tokens
#    return total_loss/Ly #  / num_tokens if num_tokens > 0 else 0.0



# def teacher_forcing_loss_emb(
#     model,
#     x_emb,          # [B, Lx, E] embedded input
#     y_ids_unpadded, # [B, Ly] target IDs
#     criterion,
#     chunk_size=256*8
# ):
#     """
#     A "chunked" teacher-forcing approach that avoids storing large outputs.
#     Instead of feeding (x_emb + y_emb_input) all at once, we break it
#     into smaller chunks (up to `chunk_size` tokens each) and process them
#     in a rolling fashion, carrying over (hidden, memory) between chunks.

#     - The first (Lx-1) frames are "input frames" only and do not produce
#       predictions. After that, frames produce predictions that map to `y_ids_unpadded`.
#     - We accumulate cross-entropy chunk by chunk, skipping the frames
#       that belong to the "input only" region. 
#     - This avoids OOM for very large sequences, 
#       and also avoids storing a big `outputs_list` in VRAM.

#     Returns:
#         avg_loss: the total cross-entropy over all predicted tokens,
#                   divided by the total # of predicted tokens (i.e. Ly).
#     """

#     B, Lx, E = x_emb.shape         # x_emb is [B, Lx, E]
#     Ly = y_ids_unpadded.shape[1]   # y_ids_unpadded is [B, Ly]

#     # Build the teacher-forcing partial input for the target side
#     y_emb_input = model.embed(y_ids_unpadded)[:, :-1, :]  # shape [B, Ly-1, E]
#     # The full input is x_emb + partial target embeddings, total length = Lx + (Ly - 1)
#     full_input = torch.cat([x_emb, y_emb_input], dim=1)   
#     full_len = full_input.size(1)  # Lx + (Ly - 1)

#     # We'll do a rolling forward pass, chunk by chunk
#     hidden = None
#     memory = None

#     # We'll accumulate cross-entropy in a running sum
#     total_loss = 0.0
#     total_predicted_tokens = 0

#     pos = 0
#     while pos < full_len:
#         chunk_end = min(pos + chunk_size, full_len)
#         input_chunk = full_input[:, pos:chunk_end, :]  # shape [B, chunk_len, E]
#         chunk_len = input_chunk.size(1)

#         out_chunk, mem_new, hidden_new = model(
#             input_chunk, hidden=hidden, memory=memory
#         )
#         # out_chunk has shape [B, chunk_len, vocab_size or something]

#         # Update hidden/memory for next chunk
#         hidden = hidden_new
#         memory = mem_new

#         # ----------------------------------------------------------
#         # Determine how many frames in this chunk correspond to *predictions*
#         # We skip the frames from the "input only" region if pos < Lx-1.
#         # The first (Lx-1) frames do not produce predictions.
#         # local_pred_start = how many frames in this chunk are "input only"?
#         # E.g. if pos=0 and Lx-1=10, chunk_len=8 => local_pred_start= max(0, 10 - 0)=10, but that's >8 => no predictions
#         # The portion from local_pred_start.. chunk_len are valid predicted frames in out_chunk.
#         # local_pred_start = max(0, (Lx - 1) - pos).
#         #
#         local_pred_start = max(0, (Lx - 1) - pos)
#         local_pred_len = chunk_len - local_pred_start

#         if local_pred_len > 0:
#             # out_chunk_pred => [B, local_pred_len, vocab_size]
#             out_chunk_pred = out_chunk[:, local_pred_start:, :]

#             # Flatten for the criterion
#             out_chunk_pred = out_chunk_pred.reshape(B * local_pred_len, -1)

#             # Now find the corresponding targets in y_ids_unpadded
#             # The global predicted frames are from [pos + local_pred_start .. pos + chunk_len).
#             # But the first (Lx-1) frames do not produce a target => 
#             # effectively, target offset in y is 
#             #   startY = (pos + local_pred_start) - (Lx -1)
#             #   endY   = (pos + chunk_len)         - (Lx -1)
#             # We clamp these to [0.. Ly] because the model only predicts Ly tokens in total.
#             global_startY = (pos + local_pred_start) - (Lx - 1)
#             global_endY   = (pos + chunk_len)         - (Lx - 1)

#             # clamp
#             global_startY_clamped = max(0, global_startY)
#             global_endY_clamped   = min(Ly, global_endY)

#             # The # of tokens we actually have here
#             actual_lenY = global_endY_clamped - global_startY_clamped

#             if actual_lenY > 0:
#                 # The corresponding slice of y_ids_unpadded is [ global_startY_clamped : global_endY_clamped ]
#                 targets_slice = y_ids_unpadded[:, global_startY_clamped : global_endY_clamped]
#                 # Flatten
#                 targets_slice = targets_slice.reshape(-1)

#                 # If there's a mismatch in shape vs. out_chunk_pred => we might need to slice out_chunk_pred further
#                 # because local_pred_len might exceed actual_lenY if the doc ended.
#                 # So let's also clamp out_chunk_pred if actual_lenY < local_pred_len
#                 # difference = local_pred_len - actual_lenY
#                 if actual_lenY < local_pred_len:
#                     # out_chunk_pred has shape [B * local_pred_len, vocab]
#                     # we want only the first B*actual_lenY
#                     needed_elems = B * actual_lenY
#                     out_chunk_pred = out_chunk_pred[:needed_elems, :]

#                 # Compute partial cross-entropy
#                 partial_loss = criterion(out_chunk_pred, targets_slice)

#                 total_loss += partial_loss * actual_lenY  # sum of losses
#                 total_predicted_tokens += actual_lenY

#         pos = chunk_end  # move on

#     # end while

#     if total_predicted_tokens == 0:
#         # Means the entire input was smaller than (Lx-1), or we had no valid prediction frames
#         # Return 0 to avoid dividing by zero
#         return 0.0

#     # average cross-entropy across all predicted tokens
#     avg_loss = total_loss / total_predicted_tokens
#     return avg_loss

def teacher_forcing_loss_emb(model, x_ids, y_ids_unpadded, criterion, chunk_size=32, x_emb=None):

    try:
        
        if x_emb == None:
            x_emb = model.embed(x_ids)
            
        B, Lx, E = x_emb.shape
        Ly = y_ids_unpadded.shape[1]
        
    
        if isinstance(model, (Transformer, Mamba)):
            # Build partial target embedding (same as before)
            y_emb_input = model.embed(y_ids_unpadded)
            full_input_emb = torch.cat([x_emb, y_emb_input], dim=1)[:, :-1]
            full_input_ids = torch.cat([x_ids, y_ids_unpadded], dim=1)[:, 1:] #do fully autoregressive
            
            # Get outputs
            outputs, _, _ = model(full_input_emb)
            
            # Get logits starting from context
            # logits = outputs[:, (Lx-1):, :]
            logits = outputs
            
            # Flatten predictions
            logits_2d = logits.reshape(-1, logits.size(-1))
            
            # CHANGE: Use shifted targets to match RNN branch
            # gold = y_ids_unpadded[:, 1:].reshape(-1)  # Start from index 1
            gold = full_input_ids.reshape(-1)  # Start from index 0
            
            # Handle size mismatches (same as before)
            if gold.size(0) != logits_2d.size(0):
                pdb.set_trace()
                min_size = min(gold.size(0), logits_2d.size(0))
                gold = gold[:min_size]
                logits_2d = logits_2d[:min_size, :]
            # print(f"logits_2d size : {logits_2d.shape} gold size : {gold.shape}")
            # Calculate loss (similar to RNN branch)
            avg_loss = criterion(logits_2d, gold)
            
            return avg_loss  
        else:
            
            hidden = None
            memory = None
            total_loss = 0.0
            total_predicted_tokens = 0
            
            # Process input sequence first
            pos = 0
            # 1) If it's a Transformer or Mamba => single-pass causal approach
            while pos < Lx:
                chunk_end = min(pos + chunk_size, Lx)
                input_chunk = x_emb[:, pos:chunk_end, :]
                
                out_chunk, mem_new, hidden_new = model(input_chunk, hidden=hidden, memory=memory)
                hidden = hidden_new
                memory = mem_new
                pos = chunk_end
        
            # Now process target sequence chunk by chunk
            pos = 0
            while pos < Ly - 1:  # -1 because we don't embed the last target token
                chunk_end = min(pos + chunk_size, Ly - 1)
                # Only embed the current chunk of target sequence
                y_chunk = y_ids_unpadded[:, pos:chunk_end]
                y_emb_chunk = model.embed(y_chunk)
                
                out_chunk, mem_new, hidden_new = model(y_emb_chunk, hidden=hidden, memory=memory)
                
                # Update states
                hidden = hidden_new
                memory = mem_new
        
                # Compute loss for this chunk
                out_chunk = out_chunk.reshape(-1, out_chunk.size(-1))
                targets = y_ids_unpadded[:, pos+1:chunk_end+1].reshape(-1)  # shift by 1 for next-token prediction
                
                if targets.size(0) > 0:  # ensure we have targets
                    chunk_loss = criterion(out_chunk, targets)
                    total_loss += chunk_loss * targets.size(0)
                    total_predicted_tokens += targets.size(0)
                
                pos = chunk_end
        
            if total_predicted_tokens == 0:
                pdb.set_trace()
                return 0.0
        
            avg_loss = total_loss / total_predicted_tokens # you have to do the average of averages
            return avg_loss

    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM in teacher_forcing_loss_emb: {str(e)}")
        torch.cuda.empty_cache()
        raise  # Re-raise to be caught by train_micro_batch

    except Exception as e:
        print(f"Error in teacher_forcing_loss_emb: {str(e)}")
        torch.cuda.empty_cache()
        raise  # Re-raise to be caught by train_micro_batch



# def teacher_forcing_loss_emb_detatch_version(model, x_emb, y_ids_unpadded, criterion, chunk_size=256, backward=False):
#     B, Lx, E = x_emb.shape
#     Ly = y_ids_unpadded.shape[1]
#     total_loss = 0.0
#     total_predicted_tokens = 0
    
#     # Initialize states
#     hidden = None
#     memory = None
    
#     # Process input sequence first
#     pos = 0
#     while pos < Lx:
#         chunk_end = min(pos + chunk_size, Lx)
#         # Process input chunk
#         input_chunk = x_emb[:, pos:chunk_end, :]
#         out_chunk, mem_new, hidden_new = model(input_chunk, hidden=hidden, memory=memory)
        
#         # Update states with detached versions
#         if isinstance(hidden_new, tuple):
#             hidden = tuple(h for h in hidden_new)
#         else:
#             hidden = hidden_new if hidden_new is not None else None
#         memory = mem_new if mem_new is not None else None
        
#         del out_chunk, mem_new, hidden_new
#         torch.cuda.empty_cache()
#         pos = chunk_end

#     # Now process target sequence
#     pos = 0
#     while pos < Ly - 1:  # -1 because we predict next token
#         chunk_end = min(pos + chunk_size, Ly - 1)
        
#         # Get target chunk
#         y_chunk = y_ids_unpadded[:, pos:chunk_end]
#         y_emb_chunk = model.embed(y_chunk)
        
#         # Forward pass
#         out_chunk, mem_new, hidden_new = model(y_emb_chunk, hidden=hidden, memory=memory)
        
#         # Compute loss for this chunk
#         out_chunk_flat = out_chunk.reshape(-1, out_chunk.size(-1))
#         targets = y_ids_unpadded[:, pos+1:chunk_end+1].reshape(-1)  # shift by 1 for next-token prediction
        
#         if targets.size(0) > 0:  # ensure we have targets
#             if backward:
#                 chunk_loss = criterion(out_chunk_flat, targets)
#                 chunk_loss.backward()
#                 total_loss += chunk_loss * targets.size(0)
#             else:
#                 with torch.no_grad():
#                     chunk_loss = criterion(out_chunk_flat, targets)
#                     total_loss += chunk_loss * targets.size(0)
#             total_predicted_tokens += targets.size(0)
        
#         # Update states with detached versions
#         if isinstance(hidden_new, tuple):
#             hidden = tuple(h.detach() for h in hidden_new)
#         else:
#             hidden = hidden_new.detach() if hidden_new is not None else None
#         memory = mem_new.detach() if mem_new is not None else None
        
#         # Clean up
#         del out_chunk, out_chunk_flat, mem_new, hidden_new, y_emb_chunk, y_chunk
#         torch.cuda.empty_cache()
#         pos = chunk_end

#     return total_loss / total_predicted_tokens if total_predicted_tokens > 0 else 0.0



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
def mezo_char_layerwise(model, x_emb, y, criterion, epsilon=1e-3):
    """
    Layerwise MeZO with grouped parameter perturbations.

    Args:
        model (nn.Module): The model being trained.
        x_emb (torch.Tensor): Input embeddings [B, L, E].
        y (torch.LongTensor): Target tensor [B, L].
        criterion (nn.Module): Loss function.
        epsilon (float): Finite difference step size.

    Returns:
        float: Average loss over all layers.
    """
    named_params = list(model.named_parameters())
    layer_dict = group_params_by_layer(named_params)
    all_params = list(model.parameters())
    local_seed = torch.randint(0, 2**32, (1,)).item()

    with torch.inference_mode():
        # Zero out old gradients
        for p in all_params:
            if p.grad is not None:
                p.grad.zero_()
    
        total_loss = 0.0
        layer_count = 0

    
        for layer_name, param_list in layer_dict.items():
            #torch.cuda.empty_cache() 
            layer_count += 1

            torch.manual_seed(local_seed)
            #torch.cuda.empty_cache() 
            # + epsilon perturbation
            for i, (_, p) in enumerate(param_list):
                d = torch.randn_like(p)
                p.data.add_(epsilon * d)

            # Forward pass for positively perturbed model
            loss_plus = teacher_forcing_loss_emb(model, x_emb, y, criterion)

            #torch.cuda.empty_cache() 
            # -2 epsilon perturbation
            torch.manual_seed(local_seed)
            for i, (_, p) in enumerate(param_list):
                d = torch.randn_like(p)
                p.data.sub_(2.0 * epsilon * d)

            # Forward pass for negatively perturbed model
            loss_minus = teacher_forcing_loss_emb(model, x_emb, y, criterion)

            # Restore original parameters
            torch.manual_seed(local_seed)
            #torch.cuda.empty_cache() 
            
            for i, (_, p) in enumerate(param_list):
                d = torch.randn_like(p)
                p.data.add_(epsilon * d)


            # Compute gradient estimate for this layer
            grad_est = (loss_plus - loss_minus) / (2 * epsilon)

            # Accumulate gradients
            torch.manual_seed(local_seed)
            #torch.cuda.empty_cache() 
            
            for i, (_, p) in enumerate(param_list):
                d = torch.randn_like(p)
                if p.grad is None:
                    p.grad = grad_est * d
                else:
                    p.grad.add_(grad_est * d)

            avg_loss = 0.5 * (loss_plus.item() + loss_minus.item())
            total_loss += avg_loss

    # Average loss across layers
    if layer_count > 0:
        total_loss /= layer_count

    return total_loss



##############################################################################
# Activity-based Node Perturbation (ANP) : NOT TESTED! TODO 
##############################################################################
# def anp_single(model, x_ids, y, criterion, epsilon=1e-3, decorrelation_matrix=None, verbose=False):
#     """
#     Implements Activity-based Node Perturbation (ANP) for gradient-free training with decorrelation.

#     Args:
#         model (torch.nn.Module): The neural network model to train.
#         x_ids
#         y (torch.Tensor): Ground truth labels.
#         criterion (callable): Loss function.
#         epsilon (float): Perturbation magnitude.
#         decorrelation_matrix (torch.Tensor): Decorrelation matrix to be updated and persisted across time steps.
#         verbose (bool): If True, prints detailed debug information.

#     Returns:
#         float: Average loss across the perturbed passes.
#         torch.Tensor: Updated decorrelation matrix.
#     """
#     with torch.no_grad():
#         x_embeddings = model.embed(x_ids)

#         if verbose:
#             print("\n=== ANP SINGLE DEBUG ===")  # Debug output header
#             print(f"Input shape: {x_embeddings.shape}")  # Print input tensor shape
#             print(f"Target shape: {y.shape}")  # Print target tensor shape
    
#         # Initialize a random seed for reproducibility of perturbations
#         local_seed = torch.randint(0, 2**32, (1,)).item()  # Generate random seed
#         all_params = list(model.parameters())  # Get all model parameters
    
#         # Ensure no gradients persist
#         for p in all_params:
#             if p.grad is not None:
#                 p.grad.zero_()  # Zero out gradients
    
#         if verbose:
#             print(f"\nRandom seed: {local_seed}")  # Print random seed
#             print("Parameter shapes:")  # Print parameter shapes for debugging
#             for i, p in enumerate(all_params):
#                 if p.requires_grad:
#                     print(f"Param {i}: {p.shape}")  # Print each parameter shape
    
#         # === Decorrelation Mechanism ===
#         if decorrelation_matrix is None:
#             decorrelation_matrix = torch.eye(x_embeddings.shape[-1], device=x_embeddings.device)  # Initialize decorrelation matrix if not provided
    
#         def decorrelate(inputs, decorrelation_matrix):
#             # Apply decorrelation: x* = D * x
#             return torch.matmul(inputs, decorrelation_matrix)
    
#         decorrelated_inputs = decorrelate(x_embeddings, decorrelation_matrix)  # Apply decorrelation to inputs
    
#         def update_decorrelation_matrix(decorrelation_matrix, decorrelated_inputs):
#             # Update decorrelation matrix: ΔD = (x* x*^T - diag(x* x*)) D
#             batch_size, seq_len, feature_dim = decorrelated_inputs.shape
#             decorrelated_inputs_flat = decorrelated_inputs.reshape(batch_size * seq_len, feature_dim)
#             covariance = torch.matmul(decorrelated_inputs_flat.T, decorrelated_inputs_flat) / (batch_size * seq_len)
#             diagonal = torch.diag(torch.diag(covariance))
#             update = torch.matmul(covariance - diagonal, decorrelation_matrix)
#             return decorrelation_matrix - epsilon * update
    
#         decorrelation_matrix = update_decorrelation_matrix(decorrelation_matrix, decorrelated_inputs)
    
#         if verbose:
#             print("\nDecorrelated inputs computed and decorrelation matrix updated.")  # Debug output for decorrelation
    
#         # === Clean Pass ===
#         if verbose:
#             print("\n=== Clean Pass ===")  # Debug output for clean pass
    
#         outputs_clean = model(decorrelated_inputs)  # Forward pass without perturbation
#         pdb.set_trace()
#         if isinstance(outputs_clean, tuple):  # Ensure output is a tensor
#             outputs_clean = outputs_clean[0]
#         outputs_clean = outputs_clean.view(-1, outputs_clean.size(-1))  # Flatten output
#         y_flat = y.view(-1)  # Flatten target
        
#         loss_clean = criterion(outputs_clean, y_flat)  # Compute clean loss
    
#         # Capture pre-activations during the clean pass
#         pre_activations_clean = {}
#         for name, module in model.named_modules():
#             if hasattr(module, 'weight') and module.weight.requires_grad:
#                 pre_activations_clean[name] = module.weight.data.clone()
    
#         if verbose:
#             print(f"Clean pass loss: {loss_clean.item()}")
    
#         # === Noisy Pass ===
#         if verbose:
#             print("\n=== Noisy Pass ===")  # Debug output for noisy pass
    
#         torch.manual_seed(local_seed)  # Reset the seed for reproducibility
    
#         pre_activation_differences = {}
#         for i, (name, module) in enumerate(model.named_modules()):
#             if hasattr(module, 'weight') and module.weight.requires_grad:
#                 torch.manual_seed(local_seed + i)  # Reset seed for unique perturbations
#                 perturbation = torch.randn_like(module.weight) * epsilon  # Generate random perturbation
#                 module.weight.data.add_(perturbation)  # Apply perturbation to weights
    
#                 # Compute pre-activation difference
#                 pre_activation_differences[name] = module.weight.data.clone() - pre_activations_clean[name]
    
#         outputs_noisy = model(decorrelated_inputs)  # Forward pass with perturbed weights
#         if isinstance(outputs_noisy, tuple):
#             outputs_noisy = outputs_noisy[0]  # Use the logits tensor
#         outputs_noisy = outputs_noisy.view(-1, outputs_noisy.size(-1))  # Flatten output
#         loss_noisy = criterion(outputs_noisy, y_flat)  # Compute noisy loss
    
#         if verbose:
#             print(f"Noisy pass loss: {loss_noisy.item()}")
    
#         # Revert the weights to original state
#         for i, (name, module) in enumerate(model.named_modules()):
#             if hasattr(module, 'weight') and module.weight.requires_grad:
#                 torch.manual_seed(local_seed + i)  # Reset the seed
#                 perturbation = torch.randn_like(module.weight) * epsilon  # Generate identical perturbation
#                 module.weight.data.sub_(perturbation)  # Revert to original weights
    
#         # === Compute Gradients ===
#         grad_estimate = (loss_noisy - loss_clean) / (2.0 * epsilon)  # Estimate gradient
    
#         for name, module in model.named_modules():
#             if hasattr(module, 'weight') and module.weight.requires_grad:
#                 if name in pre_activation_differences:
#                     pre_diff = pre_activation_differences[name]
#                     normalized_pre_diff = pre_diff / (torch.norm(pre_diff) + 1e-8)  # Avoid division by zero
#                     if module.weight.grad is None:
#                         module.weight.grad = grad_estimate * normalized_pre_diff  # Initialize gradient
#                     else:
#                         module.weight.grad.add_(grad_estimate * normalized_pre_diff)  # Accumulate gradient
    
#         # === Average Loss ===
#         avg_loss = 0.5 * (loss_clean.item() + loss_noisy.item())  # Compute average loss
    
#         if verbose:
#             print(f"\nFinal average loss: {avg_loss}")  # Print average loss
#             print("=== ANP Single Complete ===\n")  # Debug output footer

#     return avg_loss, decorrelation_matrix  # Return average loss and updated decorrelation matrix



# def anp_single(model, x_ids, y, criterion, epsilon=1e-3, decorrelation_matrix=None, verbose=True):
#     with torch.no_grad():
#         x_emb = model.embed(x_ids)
#         B, Lx, E = x_emb.shape
        
#         if verbose:
#             print("=== Model Parameter Shapes ===")
#             for name, param in model.named_parameters():
#                 print(f"{name}: {param.shape}")
#             print("\n=== Input Shapes ===")
#             print(f"x_emb shape: {x_emb.shape}")
#             print(f"model.d_model: {model.d_model}")

#         if isinstance(model, (Transformer, Mamba)):
#             # Setup phase
#             Ly = y.size(1)
#             full_input_ids = torch.cat([x_ids, y], dim=1)
#             max_seq_len = max(full_input_ids.size(1), 15)
#             full_input_ids = full_input_ids[:, :max_seq_len]
#             full_input_emb = model.embed(full_input_ids)

#             # Initialize decorrelation matrix if needed
#             if decorrelation_matrix is None:
#                 decorrelation_matrix = torch.eye(full_input_emb.size(-1), device=x_emb.device)
            
#             # Store parameter-specific noises
#             param_noises = {}
            
#             # Clean forward pass
#             decorrelated_inputs = torch.matmul(full_input_emb, decorrelation_matrix)
#             outputs_clean, _, _ = model(decorrelated_inputs)
#             logits_clean = outputs_clean.reshape(-1, outputs_clean.size(-1))
#             targets = full_input_ids.reshape(-1)
#             min_size = min(logits_clean.size(0), targets.size(0))
#             logits_clean = logits_clean[:min_size, :]
#             targets = targets[:min_size]
#             loss_clean = criterion(logits_clean, targets)

#             # Noisy forward pass with careful parameter handling
#             torch.manual_seed(torch.randint(0, 2**32, (1,)).item())
            
#             # First, generate and store all noises
#             for name, param in model.named_parameters():
#                 if param.requires_grad:
#                     noise = torch.randn_like(param) * epsilon
#                     param_noises[name] = noise
#                     if verbose:
#                         print(f"Adding noise to {name}: param shape {param.shape}, noise shape {noise.shape}")
#                     param.data.add_(noise)

#             # Forward pass with noise
#             outputs_noisy, _, _ = model(decorrelated_inputs)
#             logits_noisy = outputs_noisy.reshape(-1, outputs_noisy.size(-1))
#             logits_noisy = logits_noisy[:min_size, :]
#             loss_noisy = criterion(logits_noisy, targets)

#             # Remove noise using stored values
#             for name, param in model.named_parameters():
#                 if param.requires_grad:
#                     if verbose:
#                         print(f"Removing noise from {name}: param shape {param.shape}, noise shape {param_noises[name].shape}")
#                     param.data.sub_(param_noises[name])

#             # Compute gradient estimate
#             grad_estimate = (loss_noisy - loss_clean) / (2.0 * epsilon)
            
#             # Apply gradient estimate
#             for param in model.parameters():
#                 if param.requires_grad and param.grad is not None:
#                     param.grad.add_(grad_estimate)

#             avg_loss = 0.5 * (loss_clean.item() + loss_noisy.item())
            
#         else:
#             if verbose:
#                 print("\n=== ANP SINGLE DEBUG ===")  # Debug output header
#                 print(f"Input shape: {x_emb.shape}")  # Print input tensor shape
#                 print(f"Target shape: {y.shape}")  # Print target tensor shape
        
#             # Initialize a random seed for reproducibility of perturbations
#             local_seed = torch.randint(0, 2**32, (1,)).item()  # Generate random seed
#             all_params = list(model.parameters())  # Get all model parameters
        
#             # Ensure no gradients persist
#             for p in all_params:
#                 if p.grad is not None:
#                     p.grad.zero_()  # Zero out gradients
        
#             if verbose:
#                 print(f"\nRandom seed: {local_seed}")  # Print random seed
#                 print("Parameter shapes:")  # Print parameter shapes for debugging
#                 for i, p in enumerate(all_params):
#                     if p.requires_grad:
#                         print(f"Param {i}: {p.shape}")  # Print each parameter shape
        
#             # === Decorrelation Mechanism ===
#             if decorrelation_matrix is None:
#                 decorrelation_matrix = torch.eye(x_emb.shape[-1], device=x_emb.device)  # Initialize decorrelation matrix if not provided
        
#             def decorrelate(inputs, decorrelation_matrix):
#                 # Apply decorrelation: x* = D * x
#                 return torch.matmul(inputs, decorrelation_matrix)
        
#             decorrelated_inputs = decorrelate(x_emb, decorrelation_matrix)  # Apply decorrelation to inputs
        
#             def update_decorrelation_matrix(decorrelation_matrix, decorrelated_inputs):
#                 # Update decorrelation matrix: ΔD = (x* x*^T - diag(x* x*)) D
#                 batch_size, seq_len, feature_dim = decorrelated_inputs.shape
#                 decorrelated_inputs_flat = decorrelated_inputs.reshape(batch_size * seq_len, feature_dim)
#                 covariance = torch.matmul(decorrelated_inputs_flat.T, decorrelated_inputs_flat) / (batch_size * seq_len)
#                 diagonal = torch.diag(torch.diag(covariance))
#                 update = torch.matmul(covariance - diagonal, decorrelation_matrix)
#                 return decorrelation_matrix - epsilon * update
        
#             decorrelation_matrix = update_decorrelation_matrix(decorrelation_matrix, decorrelated_inputs)
        
#             if verbose:
#                 print("\nDecorrelated inputs computed and decorrelation matrix updated.")  # Debug output for decorrelation
        
#             # === Clean Pass ===
#             if verbose:
#                 print("\n=== Clean Pass ===")  # Debug output for clean pass
        
#             outputs_clean = model(decorrelated_inputs)  # Forward pass without perturbation
#             pdb.set_trace()
#             if isinstance(outputs_clean, tuple):  # Ensure output is a tensor
#                 outputs_clean = outputs_clean[0]
#             outputs_clean = outputs_clean.view(-1, outputs_clean.size(-1))  # Flatten output
#             y_flat = y.view(-1)  # Flatten target
            
#             loss_clean = criterion(outputs_clean, y_flat)  # Compute clean loss
        
#             # Capture pre-activations during the clean pass
#             pre_activations_clean = {}
#             for name, module in model.named_modules():
#                 if hasattr(module, 'weight') and module.weight.requires_grad:
#                     pre_activations_clean[name] = module.weight.data.clone()
        
#             if verbose:
#                 print(f"Clean pass loss: {loss_clean.item()}")
        
#             # === Noisy Pass ===
#             if verbose:
#                 print("\n=== Noisy Pass ===")  # Debug output for noisy pass
        
#             torch.manual_seed(local_seed)  # Reset the seed for reproducibility
        
#             pre_activation_differences = {}
#             for i, (name, module) in enumerate(model.named_modules()):
#                 if hasattr(module, 'weight') and module.weight.requires_grad:
#                     torch.manual_seed(local_seed + i)  # Reset seed for unique perturbations
#                     perturbation = torch.randn_like(module.weight) * epsilon  # Generate random perturbation
#                     module.weight.data.add_(perturbation)  # Apply perturbation to weights
        
#                     # Compute pre-activation difference
#                     pre_activation_differences[name] = module.weight.data.clone() - pre_activations_clean[name]
        
#             outputs_noisy = model(decorrelated_inputs)  # Forward pass with perturbed weights
#             if isinstance(outputs_noisy, tuple):
#                 outputs_noisy = outputs_noisy[0]  # Use the logits tensor
#             outputs_noisy = outputs_noisy.view(-1, outputs_noisy.size(-1))  # Flatten output
#             loss_noisy = criterion(outputs_noisy, y_flat)  # Compute noisy loss
        
#             if verbose:
#                 print(f"Noisy pass loss: {loss_noisy.item()}")
        
#             # Revert the weights to original state
#             for i, (name, module) in enumerate(model.named_modules()):
#                 if hasattr(module, 'weight') and module.weight.requires_grad:
#                     torch.manual_seed(local_seed + i)  # Reset the seed
#                     perturbation = torch.randn_like(module.weight) * epsilon  # Generate identical perturbation
#                     module.weight.data.sub_(perturbation)  # Revert to original weights
        
#             # === Compute Gradients ===
#             grad_estimate = (loss_noisy - loss_clean) / (2.0 * epsilon)  # Estimate gradient
        
#             for name, module in model.named_modules():
#                 if hasattr(module, 'weight') and module.weight.requires_grad:
#                     if name in pre_activation_differences:
#                         pre_diff = pre_activation_differences[name]
#                         normalized_pre_diff = pre_diff / (torch.norm(pre_diff) + 1e-8)  # Avoid division by zero
#                         if module.weight.grad is None:
#                             module.weight.grad = grad_estimate * normalized_pre_diff  # Initialize gradient
#                         else:
#                             module.weight.grad.add_(grad_estimate * normalized_pre_diff)  # Accumulate gradient
    
#             # === Average Loss ===
#             avg_loss = 0.5 * (loss_clean.item() + loss_noisy.item())  # Compute average loss
#     return avg_loss, decorrelation_matrix

def anp_single(model, x_ids, y, criterion, epsilon=1e-3, decorrelation_matrix=None, verbose=False):
    """
    Implements Activity-based Node Perturbation (ANP) for gradient-free training with decorrelation,
    but uses teacher_forcing_loss_emb(...) to compute the forward loss for both 
    Transformer/Mamba and legacy RNN/NTM/DNC models.

    Args:
        model (torch.nn.Module): The neural network model to train.
        x_ids (torch.LongTensor): [B, L], input token IDs.
        y (torch.LongTensor): [B, L], ground truth token IDs.
        criterion (callable): Loss function (e.g. CrossEntropy).
        epsilon (float): Perturbation magnitude for the ANP steps.
        decorrelation_matrix (torch.Tensor): Matrix used to decorrelate the input embeddings.
        verbose (bool): If True, prints debug info about shapes, seeds, etc.

    Returns:
        avg_loss (float): 0.5 * (loss_clean + loss_noisy)
        decorrelation_matrix (torch.Tensor): Updated matrix after this step.
    """
    with torch.no_grad(): # DONT USE INFERENCE_MODE BC ANP NEEDS THE ACTIVATIONS! 
        # ---------------------------------------------------------------
        # 1) Get raw embeddings from x_ids
        # ---------------------------------------------------------------
        x_embeddings = model.embed(x_ids)

        if verbose:
            print("\n=== ANP SINGLE DEBUG ===")
            print(f" x_ids shape         : {x_ids.shape}")
            print(f" x_embeddings shape  : {x_embeddings.shape}")
            print(f" y shape             : {y.shape}")

        # ---------------------------------------------------------------
        # 2) Initialize random seed & zero old grads
        # ---------------------------------------------------------------
        local_seed = torch.randint(0, 2**32, (1,)).item()
        all_params = list(model.parameters())

        # Zero out old gradients
        for p in all_params:
            if p.grad is not None:
                p.grad.zero_()

        if verbose:
            print(f"\nRandom seed: {local_seed}")
            print("Parameter shapes:")
            for i, p in enumerate(all_params):
                if p.requires_grad:
                    print(f"  Param {i} -> shape {p.shape}")

        # ---------------------------------------------------------------
        # 3) Decorrelation
        # ---------------------------------------------------------------
        if decorrelation_matrix is None:
            # shape: [embed_dim, embed_dim]
            decorrelation_matrix = torch.eye(x_embeddings.shape[-1], device=x_embeddings.device)

        def decorrelate(inputs, D):
            # inputs: [B, L, E], D: [E, E]
            return torch.matmul(inputs, D)

        decorrelated_inputs = decorrelate(x_embeddings, decorrelation_matrix)
        B, seq_len, emb_dim = decorrelated_inputs.shape

        if verbose:
            print(f"Decorrelated input shape: {decorrelated_inputs.shape} (B,L,E={emb_dim})")

        def update_decorrelation_matrix(D, dec_inp):
            # dec_inp: [B, L, E]
            batch_size, seq_len, feature_dim = dec_inp.shape
            flat = dec_inp.reshape(batch_size * seq_len, feature_dim)
            # covariance [E, E]
            covariance = torch.matmul(flat.T, flat) / (batch_size * seq_len)
            diagonal = torch.diag(torch.diag(covariance))
            update_ = torch.matmul(covariance - diagonal, D)
            return D - epsilon * update_

        # Update decorrelation matrix
        decorrelation_matrix = update_decorrelation_matrix(decorrelation_matrix, decorrelated_inputs)

        if verbose:
            print(f"Updated decorrelation matrix shape: {decorrelation_matrix.shape}")

        # ---------------------------------------------------------------
        # 4) Clean Pass => no perturbation
        # ---------------------------------------------------------------
        if verbose:
            print("\n=== Clean Pass ===")

        # Use teacher_forcing_loss_emb w/ x_emb = decorrelated_inputs
        # The function does the forward pass internally (for RNN or Transformer).
        loss_clean = teacher_forcing_loss_emb(
            model,
            x_ids,               # This is going to be ignored via the embeddings
            y,                   
            criterion=criterion,
            x_emb=decorrelated_inputs  # ... override embeddings
        )

        if verbose:
            print(f"Clean pass loss: {loss_clean.item():.6f}")

        # Grab a snapshot of each weight for the "pre-activation difference" logic
        pre_activations_clean = {}
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight.requires_grad:
                pre_activations_clean[name] = module.weight.data.clone()

        # ---------------------------------------------------------------
        # 5) Noisy Pass => random perturbation of the model weights
        # ---------------------------------------------------------------
        if verbose:
            print("\n=== Noisy Pass ===")

        torch.manual_seed(local_seed)  # same seed => same noise directions
        pre_activation_differences = {}

        for i, (name, module) in enumerate(model.named_modules()):
            if hasattr(module, 'weight') and module.weight.requires_grad:
                # reset the seed for each param
                torch.manual_seed(local_seed + i)
                # random perturbation
                perturbation = torch.randn_like(module.weight) * epsilon
                module.weight.data.add_(perturbation)

                # store difference
                pre_activation_differences[name] = (
                    module.weight.data.clone() - pre_activations_clean[name]
                )

        # Now measure the "noisy" forward pass
        loss_noisy = teacher_forcing_loss_emb(
            model,
            x_ids,
            y,
            criterion=criterion,
            chunk_size=32,
            x_emb=decorrelated_inputs
        )

        if verbose:
            print(f"Noisy pass loss: {loss_noisy.item():.6f}")

        # ---------------------------------------------------------------
        # 6) Revert the model weights
        # ---------------------------------------------------------------
        torch.manual_seed(local_seed)
        for i, (name, module) in enumerate(model.named_modules()):
            if hasattr(module, 'weight') and module.weight.requires_grad:
                torch.manual_seed(local_seed + i)
                revert_perturb = torch.randn_like(module.weight) * epsilon
                module.weight.data.sub_(revert_perturb)

        # ---------------------------------------------------------------
        # 7) Compute gradient & apply to module.weight.grad
        # ---------------------------------------------------------------
        grad_estimate = (loss_noisy - loss_clean) / (2.0 * epsilon)
        if verbose:
            print(f"\nEstimated gradient: {grad_estimate.item():.6f}")

        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight.requires_grad:
                if name in pre_activation_differences:
                    diff_ = pre_activation_differences[name]
                    norm_ = torch.norm(diff_) + 1e-8
                    normalized_pre_diff = diff_ / norm_
                    if module.weight.grad is None:
                        module.weight.grad = grad_estimate * normalized_pre_diff
                    else:
                        module.weight.grad.add_(grad_estimate * normalized_pre_diff)

        # ---------------------------------------------------------------
        # 8) Final average loss
        # ---------------------------------------------------------------
        avg_loss = 0.5 * (loss_clean.item() + loss_noisy.item())
        if verbose:
            print(f"\nFinal average loss: {avg_loss:.6f}")
            print("=== ANP Single Complete ===\n")

    return avg_loss, decorrelation_matrix



##############################################################################
# Warm single mezo  (warm_single_mezo)
##############################################################################

def mezo_char_single_with_warm_start(
    model, 
    x_emb, 
    y, 
    criterion, 
    epsilon=1e-3, 
    verbose=False,
    max_perturbations=1,
    min_acceptable_loss_percent_diff=0.001,
    init_seed=None
):
    """
    Instrumented version of mezo_char_single with multiple perturbations and warm start based on last best direction found.
    
    We attempt up to `max_perturbations` different random directions.
    Each time:
      1) We do a plus pass and minus pass, reusing the same random seed
         to produce +/- directions consistently.
      2) We compute the absolute difference between (loss_plus) and (loss_minus),
         then normalize by the average => percent_diff = diff_val / avg_val.
      3) If percent_diff >= min_acceptable_loss_percent_diff, we immediately
         accept that direction, apply the gradient update, and return.
      4) Otherwise, we keep track of whichever attempt had the highest percent_diff,
         calling that our "best direction so far."
    If, after all attempts, none meets the threshold, we do one more pass with
    the best direction found so far and finalize an update using that.

    Args:
        model (nn.Module): The model being trained.
        x_emb (torch.Tensor): Input embeddings [B, L, E].
        y (torch.LongTensor): Target tensor [B, L].
        criterion (nn.Module): Loss function.
        epsilon (float): Finite difference step size.
        verbose (bool): Verbosity flag.
        max_perturbations (int): Maximum number of random directions to try.
        min_acceptable_loss_percent_diff (float):
            Minimum required percentage-difference in losses
            for us to accept the gradient update immediately.
        init_seed (int or None):
            Optional user-provided seed to try first. If None, we pick randomly.

    Returns:
        float: The final average loss (0.5 * (L+ + L-)) for whichever direction
               we actually used for the gradient update.
        int:   The seed used for the final update (so the caller can reuse it).
    """
    all_params = list(model.parameters())

    # We'll record the "best" direction so far
    best_percent_diff = -1.0
    best_seed = None
    best_loss_plus = None
    best_loss_minus = None

    # Whether we've found a direction that meets our threshold
    early_success = False
    final_seed_used = None
    final_loss = 0.0

    def plus_minus_pass(local_seed):
        """
        Helper function that:
          (a) Zeroes grads
          (b) Does the plus pass (+εd), minus pass (-εd), reverts, 
          (c) Returns (loss_plus, loss_minus, percent_diff).
        Does NOT apply any gradient. That’s done separately if accepted.
        """
        # 1) Zero grads
        for p in all_params:
            if p.grad is not None:
                p.grad.zero_()

        with torch.no_grad():
            #torch.cuda.empty_cache()

            # -------------------------
            #     PLUS PASS
            # -------------------------
            #torch.cuda.empty_cache()
            torch.manual_seed(local_seed)
            for p in all_params:
                if p.requires_grad:
                    d = torch.randn_like(p)
                    p.data.add_(epsilon * d)

            loss_plus = teacher_forcing_loss_emb(model, x_emb, y, criterion)

            # -------------------------
            #     MINUS PASS
            # -------------------------
            #torch.cuda.empty_cache()
            torch.manual_seed(local_seed)
            for p in all_params:
                if p.requires_grad:
                    d = torch.randn_like(p)
                    p.data.sub_(2.0 * epsilon * d)

            loss_minus = teacher_forcing_loss_emb(model, x_emb, y, criterion)

            # -------------------------
            #     REVERT PARAMS
            # -------------------------
            torch.manual_seed(local_seed)
            #torch.cuda.empty_cache()
            for p in all_params:
                if p.requires_grad:
                    d = torch.randn_like(p)
                    p.data.add_(epsilon * d)

        # Compute percent-diff
        diff_val = abs(loss_plus.item() - loss_minus.item())
        avg_val = 0.5 * (loss_plus.item() + loss_minus.item())
        percent_diff = (diff_val / avg_val) if avg_val > 0 else 0.0

        return loss_plus, loss_minus, percent_diff, avg_val

    def apply_grad(local_seed, loss_plus, loss_minus):
        """
        Given the final accepted direction (loss_plus, loss_minus),
        compute grad_est and apply p.grad for each param.
        """
        grad_est = (loss_plus - loss_minus) / (2.0 * epsilon)
        torch.manual_seed(local_seed)
        #torch.cuda.empty_cache()
        for p in all_params:
            if p.requires_grad:
                d = torch.randn_like(p)
                if p.grad is None:
                    p.grad = grad_est * d
                else:
                    p.grad.add_(grad_est * d)

    # Main loop: try up to max_perturbations times
    for attempt in range(max_perturbations):
        # Pick the seed: if attempt=0 and init_seed!=None => use init_seed
        if (attempt == 0) and (init_seed is not None):
            local_seed = init_seed
        else:
            local_seed = torch.randint(0, 2**32, (1,)).item()

        if verbose:
            print(f"\n=== MEZO Attempt {attempt+1}/{max_perturbations} ===")
            print(f"Using seed: {local_seed}")

        loss_plus, loss_minus, percent_diff, avg_val = plus_minus_pass(local_seed)

        if verbose:
            print(f"  loss_plus={loss_plus.item():.6f}, loss_minus={loss_minus.item():.6f}")
            print(f"  percent_diff={percent_diff:.4%}, min_needed={min_acceptable_loss_percent_diff:.4%}")

        # Update "best" if this attempt is better
        if percent_diff > best_percent_diff:
            best_percent_diff = percent_diff
            best_seed = local_seed
            best_loss_plus = loss_plus
            best_loss_minus = loss_minus

        # If it meets threshold => accept immediately, break
        if percent_diff >= min_acceptable_loss_percent_diff:
            apply_grad(local_seed, loss_plus, loss_minus)
            final_seed_used = local_seed
            final_loss = avg_val
            early_success = True
            break
        else:
            if verbose:
                print("  => Not sufficient. Trying a new seed...")

    # If we never “broke” early, we still do an update using the best direction
    if not early_success:
        if best_seed is not None:
            if verbose:
                print(f"\nNo direction above {min_acceptable_loss_percent_diff:.4%}, "
                      f"but we'll use the best attempt with seed={best_seed} (percent_diff={best_percent_diff:.4%}).")
            # We must re-run plus/minus to re-derive the same random directions, then apply grad.
            # We'll do a second pass with best_seed
            loss_plus, loss_minus, _, avg_val = plus_minus_pass(best_seed)
            apply_grad(best_seed, loss_plus, loss_minus)
            final_seed_used = best_seed
            final_loss = avg_val
        else:
            # This is extremely unlikely, but in case no best_seed was set
            if verbose:
                print("No valid attempts found and best_seed is None. No gradient update.")
            return 0.0, None

    if verbose:
        print(f"\n[MEZO SINGLE] Final seed used: {final_seed_used}, final avg loss: {final_loss:.6f}\n")

    return final_loss, final_seed_used

##############################################################################
# Fast Single MeZO using the new sampling approach: d ~ Normal(adam ratio, 1.0)
##############################################################################
def flatten_params(model):
    """
    Flatten all model parameters into a single 1D tensor (param_data).
    Also return a list of (shape, numel, device) to reconstruct each parameter.
    """
    param_tensors = []
    meta = []
    for p in model.parameters():
        flat = p.data.view(-1)
        param_tensors.append(flat)
        meta.append((p.shape, flat.numel(), p.device))
    param_data = torch.cat(param_tensors, dim=0)
    return param_data, meta


def unflatten_into_params(param_data, model, meta):
    """
    Unflatten 'param_data' (1D) back into each model parameter's .data
    using the (shape, numel, device) info in 'meta'.
    """
    offset = 0
    for p, (shape, numel, dev) in zip(model.parameters(), meta):
        slice_ = param_data[offset : offset + numel]
        offset += numel
        p.data = slice_.view(shape).to(dev)



def flatten_adam_ratio_data(model, optimizer):
    """
    For each parameter p, fetch the Adam state:
        exp_avg  (m_t)
        exp_avg_sq (v_t)
    and compute ratio = (exp_avg / sqrt(exp_avg_sq + 1e-8)).

    If the state doesn't exist (or mismatch shape),
    fallback to zeros or random as desired. Here we'll fallback to zeros.

    Returns a single 1D tensor 'ratio_data' concatenating all parameters.
    """
    ratio_list = []
    for p in model.parameters():
        state = optimizer.state.get(p, {})
        exp_avg = state.get("exp_avg", None)
        exp_avg_sq = state.get("exp_avg_sq", None)

        if (exp_avg is not None) and (exp_avg_sq is not None):
            # Flatten
            ratio_1d = (exp_avg / torch.sqrt(exp_avg_sq + 1e-8)).view(-1)
            ratio_list.append(ratio_1d)
        else:
            # Fallback => zeros
            ratio_list.append(torch.zeros(p.data.numel(), device=p.data.device))

    ratio_data = torch.cat(ratio_list, dim=0)
    return ratio_data



# def mezo_adaptive_sampling_parallel(
#     model,
#     x_emb,
#     y,
#     criterion,
#     optimizer,
#     epsilon=1e-3,
#     verbose=False,
#     adaptive=True,
#     fixed_size_perturbation=False
# ):
#     """
#     Parallelized MeZO using model replication to run plus/minus passes simultaneously.
#     Uses torch.jit.fork() for parallel execution while maintaining separate model states.
#     """
#     # First create two separate model instances for parallel execution
#     # We use deepcopy to ensure completely separate models with their own buffers
#     model_plus = copy.deepcopy(model)
#     model_minus = copy.deepcopy(model)
    
#     # Move the copied models to the same device as the original
#     device = next(model.parameters()).device
#     model_plus.to(device)
#     model_minus.to(device)

#     # Flatten parameters just once at the start
#     orig_param_data, meta = flatten_params(model)
#     param_data_orig = orig_param_data.clone()
#     ratio_data = flatten_adam_ratio_data(model, optimizer)

#     # Zero existing gradients
#     for p in model.parameters():
#         if p.grad is not None:
#             p.grad.zero_()

#     # Sample the perturbation direction
#     local_seed = torch.randint(0, 2**32, (1,)).item()
#     torch.manual_seed(local_seed)
#     z_data = torch.randn_like(ratio_data, device=device)
    
#     if adaptive:
#         d_data = ratio_data + z_data
#     else:
#         d_data = z_data
        
#     if fixed_size_perturbation:
#         norm = torch.norm(d_data, p=2)
#         d_data = (d_data / norm)

#     # Create parameter versions for both directions
#     param_data_plus = param_data_orig + epsilon * d_data
#     param_data_minus = param_data_orig - epsilon * d_data

#     # Apply perturbations to the respective models
#     unflatten_into_params(param_data_plus, model_plus, meta)
#     unflatten_into_params(param_data_minus, model_minus, meta)

#     # Define forward computation functions
#     def compute_plus():
#         with torch.inference_mode():
#             return teacher_forcing_loss_emb(model_plus, x_emb, y, criterion)

#     def compute_minus():
#         with torch.inference_mode():
#             return teacher_forcing_loss_emb(model_minus, x_emb, y, criterion)

#     # Execute forward passes in parallel using torch.jit
#     future_plus = torch.jit.fork(compute_plus)
#     future_minus = torch.jit.fork(compute_minus)
    
#     # Wait for both computations to complete
#     loss_plus = torch.jit.wait(future_plus)
#     loss_minus = torch.jit.wait(future_minus)

#     if verbose:
#         print(f"Plus pass loss: {loss_plus.item():.6f}")
#         print(f"Minus pass loss: {loss_minus.item():.6f}")

#     # Revert original model to its initial state
#     unflatten_into_params(param_data_orig, model, meta)

#     # Compute gradient estimate
#     grad_est = (loss_plus - loss_minus) / (2.0 * epsilon)
#     grad_data = grad_est * d_data
    
#     # Unflatten gradients into original model
#     offset = 0
#     for p, (shape, numel, dev) in zip(model.parameters(), meta):
#         slice_ = grad_data[offset : offset + numel]
#         offset += numel
#         if p.grad is None:
#             p.grad = slice_.view(shape).to(dev)
#         else:
#             p.grad.add_(slice_.view(shape).to(dev))
    
#     # Clean up
#     del grad_data, d_data
#     del model_plus, model_minus  # Explicitly delete copied models
#     torch.cuda.empty_cache()     # Free up GPU memory
    
#     avg_loss = 0.5 * (loss_plus.item() + loss_minus.item())
#     return avg_loss


import torch
import copy
from torch import nn
import torch.cuda



# def mezo_adaptive_sampling_parallel(
#     model,
#     x_emb,
#     y,
#     criterion,
#     optimizer,
#     epsilon=1e-3,
#     verbose=False,
#     adaptive=True,
#     fixed_size_perturbation=False
# ):
#     """
#     Parallelized MeZO using model replication to run plus/minus passes simultaneously.
#     Uses torch.jit.fork() for parallel execution while maintaining separate model states.
#     """
#     # First create two separate model instances for parallel execution
#     # We use deepcopy to ensure completely separate models with their own buffers
#     model_plus = copy.deepcopy(model)
#     model_minus = copy.deepcopy(model)
    
#     # Move the copied models to the same device as the original
#     device = next(model.parameters()).device
#     model_plus.to(device)
#     model_minus.to(device)

#     # Flatten parameters just once at the start
#     orig_param_data, meta = flatten_params(model)
#     param_data_orig = orig_param_data.clone()
#     ratio_data = flatten_adam_ratio_data(model, optimizer)

#     # Zero existing gradients
#     for p in model.parameters():
#         if p.grad is not None:
#             p.grad.zero_()

#     # Sample the perturbation direction
#     local_seed = torch.randint(0, 2**32, (1,)).item()
#     torch.manual_seed(local_seed)
#     z_data = torch.randn_like(ratio_data, device=device)
    
#     if adaptive:
#         d_data = ratio_data + z_data
#     else:
#         d_data = z_data
        
#     if fixed_size_perturbation:
#         norm = torch.norm(d_data, p=2)
#         d_data = (d_data / norm)

#     # Create parameter versions for both directions
#     param_data_plus = param_data_orig + epsilon * d_data
#     param_data_minus = param_data_orig - epsilon * d_data

#     # Apply perturbations to the respective models
#     unflatten_into_params(param_data_plus, model_plus, meta)
#     unflatten_into_params(param_data_minus, model_minus, meta)

#     # Define forward computation functions
#     def compute_plus():
#         with torch.inference_mode():
#             return teacher_forcing_loss_emb(model_plus, x_emb, y, criterion)

#     def compute_minus():
#         with torch.inference_mode():
#             return teacher_forcing_loss_emb(model_minus, x_emb, y, criterion)

#     # Execute forward passes in parallel using torch.jit
#     future_plus = torch.jit.fork(compute_plus)
#     future_minus = torch.jit.fork(compute_minus)
    
#     # Wait for both computations to complete
#     loss_plus = torch.jit.wait(future_plus)
#     loss_minus = torch.jit.wait(future_minus)

#     if verbose:
#         print(f"Plus pass loss: {loss_plus.item():.6f}")
#         print(f"Minus pass loss: {loss_minus.item():.6f}")

#     # Revert original model to its initial state
#     unflatten_into_params(param_data_orig, model, meta)

#     # Compute gradient estimate
#     grad_est = (loss_plus - loss_minus) / (2.0 * epsilon)
#     grad_data = grad_est * d_data
    
#     # Unflatten gradients into original model
#     offset = 0
#     for p, (shape, numel, dev) in zip(model.parameters(), meta):
#         slice_ = grad_data[offset : offset + numel]
#         offset += numel
#         if p.grad is None:
#             p.grad = slice_.view(shape).to(dev)
#         else:
#             p.grad.add_(slice_.view(shape).to(dev))
    
#     # Clean up
#     del grad_data, d_data
#     del model_plus, model_minus  # Explicitly delete copied models
#     torch.cuda.empty_cache()     # Free up GPU memory
    
#     avg_loss = 0.5 * (loss_plus.item() + loss_minus.item())
#     return avg_loss





def mezo_adaptive_sampling_fast(
    model,
    x_emb,
    y,
    criterion,
    optimizer,            # Adam or similar
    epsilon=1e-3,
    verbose=False,
    adaptive=True,
    fixed_size_perturbation=False
):
    """
    A fast version of MeZO that:
      1) Flattens all model parameters -> param_data
      2) Computes ratio_data = (exp_avg / sqrt(exp_avg_sq+1e-8)) for each param, flattened
      3) Samples a single big d_data ~ Normal(mean=ratio_data, std=1.0)
      4) Does plus pass (param_data += epsilon*d_data), minus pass (param_data -= 2*epsilon*d_data),
         and revert (param_data += epsilon*d_data) with minimal unflatten calls.
      5) Applies the finite difference gradient to p.grad.

    Returns:
        float: 0.5*(loss_plus + loss_minus)
    """
    with torch.inference_mode():
        if verbose:
            print("\n=== MEZO SINGLE ADAM FAST v2 DEBUG ===")
            print(f"Input shape: {x_emb.shape}, target shape: {y.shape}")
    
        # 1) Flatten parameters
        orig_param_data, meta = flatten_params(model)
        # We'll keep a copy of the original so we can revert easily
        param_data_orig = orig_param_data.clone()
    
        # 2) Flatten the ratio_data = exp_avg / sqrt(exp_avg_sq + 1e-8)
        ratio_data = flatten_adam_ratio_data(model, optimizer)
        device = orig_param_data.device
    
        # Zero existing grads
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
    
        # 3) local_seed for reproducibility
        local_seed = torch.randint(0, 2**32, (1,)).item()
        if verbose:
            print(f"\nRandom seed: {local_seed}")
            print(f"Flattened param_data shape: {orig_param_data.shape}, ratio_data shape: {ratio_data.shape}")
    
        # We'll define a small forward helper
        def forward_loss():
            with torch.inference_mode():
                loss = teacher_forcing_loss_emb(model, x_emb, y, criterion)
                #torch.cuda.empty_cache()
                return loss
    
        # 4) Sample d_data ~ Normal(ratio_data, std=1.0)
        torch.manual_seed(local_seed)
        z_data = torch.randn_like(ratio_data, device=device)  # standard normal
    
        if adaptive:
            d_data = ratio_data + z_data  # => Normal(mean=ratio_data, std=1.0)
        else:
            d_data = z_data
    
        if fixed_size_perturbation:
            norm = torch.norm(d_data, p=2)           # Compute the L2 norm
            if verbose:
                print(f"fixed_size_perturbation={fixed_size_perturbation} norm {norm}")
            d_data = (d_data / norm)  # Normalize 
        
        #  --- PLUS PASS ---
        param_data_plus = param_data_orig + epsilon * d_data
        unflatten_into_params(param_data_plus, model, meta)
        loss_plus = forward_loss()
        if verbose:
            print(f"Plus pass loss: {loss_plus.item():.6f}")
    
        #  --- MINUS PASS ---
        param_data_minus = param_data_orig - epsilon * d_data
        unflatten_into_params(param_data_minus, model, meta)
        loss_minus = forward_loss()
        if verbose:
            print(f"Minus pass loss: {loss_minus.item():.6f}")
    
        #  --- REVERT ---
        # We revert to original (for subsequent steps in training)
        unflatten_into_params(param_data_orig, model, meta)
    
        # 5) Compute grad_est = (loss_plus - loss_minus) / (2*epsilon)
        grad_est = (loss_plus - loss_minus) / (2.0 * epsilon)
        if verbose:
            print(f"\nEstimated gradient: {grad_est.item():.6f}")
    
        # Construct a single grad_data = grad_est * d_data
        grad_data = grad_est * d_data
        
        # Unflatten grad_data into p.grad
        offset = 0
        for p, (shape, numel, dev) in zip(model.parameters(), meta):
            slice_ = grad_data[offset : offset + numel]
            offset += numel
            if p.grad is None:
                p.grad = slice_.view(shape).to(dev)
            else:
                p.grad.add_(slice_.view(shape).to(dev))
    
        del grad_data,d_data
        #torch.cuda.empty_cache()
        avg_loss = 0.5 * (loss_plus.item() + loss_minus.item())
        if verbose:
            print(f"\nFinal average loss: {avg_loss:.6f}")
            print("=== MEZO SINGLE ADAM FAST v2 COMPLETE ===\n")
    
        return avg_loss


    
##############################################################################
# Single mezo adaptive sampling
##############################################################################
def mezo_adaptive_sampling(
    model,
    x_emb,
    y,
    criterion,
    optimizer,         # The Adam (or Adam-like) optimizer with exp_avg, exp_avg_sq
    epsilon=1e-3,
    verbose=False
):
    """
    Adam-based adaptive version of mezo_char_single that does NOT store directions.
    For each parameter p, we sample the probe direction from:
        d ~ Normal(mean = exp_avg, std = sqrt(exp_avg_sq + 1e-8))
    using the Adam state. If that state doesn't exist yet, we fallback
    to standard Gaussian torch.randn_like(p).

    We do three passes:
      1) Plus pass  (p += epsilon*d)
      2) Minus pass (p -= 2*epsilon*d)
      3) Revert     (p += epsilon*d)
    Each pass calls torch.manual_seed(local_seed) so that the same random calls
    produce identical directions, avoiding storage of 'd' in memory.

    Args:
        model (nn.Module): The model being trained.
        x_emb (torch.Tensor): [B, L, E], input embeddings.
        y (torch.Tensor): [B, L], target IDs.
        criterion (callable): Loss function (e.g., cross-entropy).
        optimizer (torch.optim.Optimizer):
            Adam or similar, with 'exp_avg', 'exp_avg_sq' in optimizer.state[p].
        epsilon (float): Magnitude of the finite difference steps.
        verbose (bool): Whether to print debug info.

    Returns:
        float: Average of the plus and minus losses (0.5*(L+ + L-)).
    """
    if verbose:
        print("\n=== MEZO SINGLE ADAM DEBUG ===")
        print(f"Input shape: {x_emb.shape}")
        print(f"Target shape: {y.shape}")

    # We'll still pick a local seed to ensure reproducible directions
    local_seed = torch.randint(0, 2**32, (1,)).item()

    # Gather parameters
    all_params = list(model.parameters())

    with torch.no_grad():
        #torch.cuda.empty_cache()

        if verbose:
            print(f"\nRandom seed: {local_seed}")
            print("Parameter shapes:")
            for i, p in enumerate(all_params):
                if p.requires_grad:
                    print(f"Param {i}: {p.shape}")

        # 0) Zero out existing grads
        for p in all_params:
            if p.grad is not None:
                p.grad.zero_()

        def calc_d(optimizer):
            state = optimizer.state.get(p, {})
            exp_avg = state.get("exp_avg", None)     # Adam first moment
            exp_avg_sq = state.get("exp_avg_sq", None)  # Adam second moment
            if (exp_avg is not None) and (exp_avg_sq is not None):
                std_t = torch.sqrt(exp_avg_sq + 1e-8)  # shape same as p
                d = torch.normal(mean=exp_avg/std_t, std=1.)
                if d.shape != p.shape:
                    # Fallback if shape mismatch
                    d = torch.randn_like(p)
            else:
                d = torch.randn_like(p)
            return d
        # ------------------------------------------------------
        #               PLUS PASS
        # ------------------------------------------------------
        if verbose:
            print("\n=== Plus Pass ===")
        #torch.cuda.empty_cache()
        torch.manual_seed(local_seed)  # Ensures consistent random calls

        # For each parameter, sample direction from N(exp_avg, sqrt(exp_avg_sq+1e-8)),
        # or fallback to torch.randn_like(p) if no state is found
        for i, p in enumerate(all_params):
            if not p.requires_grad:
                continue

            # p.data += epsilon * d
            p.data.add_(epsilon * calc_d(optimizer))

            if verbose and i < 3:
                print(f"Param {i} ADAM-perturbation stats: mean(d)={d.mean().item():.6f}, std(d)={d.std().item():.6f}")

        loss_plus = teacher_forcing_loss_emb(model, x_emb, y, criterion)
        if verbose:
            print(f"Plus pass loss: {loss_plus.item()}")

        # ------------------------------------------------------
        #               MINUS PASS
        # ------------------------------------------------------
        if verbose:
            print("\n=== Minus Pass ===")
        #torch.cuda.empty_cache()
        torch.manual_seed(local_seed)  # Same seed => same random calls => same d

        for i, p in enumerate(all_params):
            if not p.requires_grad:
                continue

            # p.data -= 2.0 * epsilon * d
            p.data.sub_(2.0 * epsilon * calc_d(optimizer))

        loss_minus = teacher_forcing_loss_emb(model, x_emb, y, criterion)
        if verbose:
            print(f"Minus pass loss: {loss_minus.item()}")

        # ------------------------------------------------------
        #               REVERT
        # ------------------------------------------------------
        torch.manual_seed(local_seed)
        #torch.cuda.empty_cache()

        for i, p in enumerate(all_params):
            if not p.requires_grad:
                continue

            # p.data += epsilon * d
            p.data.add_(epsilon * calc_d(optimizer))

        # ------------------------------------------------------
        #               APPLY GRADIENT
        # ------------------------------------------------------
        grad_est = (loss_plus - loss_minus) / (2.0 * epsilon)
        if verbose:
            print(f"\nEstimated gradient: {grad_est.item()}")

        # We do a fresh random pass for the actual gradient direction
        torch.manual_seed(local_seed)
        #torch.cuda.empty_cache()

        for i, p in enumerate(all_params):
            if not p.requires_grad:
                continue

            # p.grad += grad_est * d
            if p.grad is None:
                p.grad = grad_est * calc_d(optimizer)
            else:
                p.grad.add_(grad_est * calc_d(optimizer))

            if verbose and i < 3:
                print(f"\nParam {i} gradient stats:")
                print(f"Mean(grad): {p.grad.mean().item():.6f}, Std(grad): {p.grad.std().item():.6f}")

        # Final average loss
        avg_loss = 0.5 * (loss_plus.item() + loss_minus.item())
        if verbose:
            print(f"\nFinal average loss: {avg_loss}")
            print("=== MEZO SINGLE ADAM COMPLETE ===\n")

    return avg_loss



##############################################################################
# Single mezo
##############################################################################
def mezo_char_single(model, x_emb, y, criterion, epsilon=1e-3, verbose=False):
    """Instrumented version of mezo_char_single"""
    if verbose:
        print("\n=== MEZO SINGLE DEBUG ===")
        print(f"Input shape: {x_emb.shape}")
        print(f"Target shape: {y.shape}")

    local_seed = torch.randint(0, 2**32, (1,)).item()
    all_params = list(model.parameters())
    with torch.inference_mode():
        #torch.cuda.empty_cache() 
                
    
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
        #torch.cuda.empty_cache() 
        torch.manual_seed(local_seed)
        
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
        #torch.cuda.empty_cache() 
        torch.manual_seed(local_seed)
        for p in all_params:
            if p.requires_grad:
                d = torch.randn_like(p)
                p.data.sub_(2.0 * epsilon * d)
    
        loss_minus = teacher_forcing_loss_emb(model, x_emb, y, criterion)
        if verbose:
            print(f"Minus pass loss: {loss_minus.item()}")
    
        # Revert parameters
        torch.manual_seed(local_seed)
        #torch.cuda.empty_cache() 
        
        for p in all_params:
            if p.requires_grad:
                d = torch.randn_like(p)
                p.data.add_(epsilon * d)
    
        # Compute and apply gradients
        grad_est = (loss_plus - loss_minus) / (2.0 * epsilon)
        if verbose:
            print(f"\nEstimated gradient: {grad_est}")
    
        torch.manual_seed(local_seed)
        #torch.cuda.empty_cache() 
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



#########
# Generation functions for your models
#########


# def generate_sequence_batched(model, x_ids, embed, char_to_id, id_to_char, max_seq_len=20, device='cuda', 
#                                criterion=None, y_ids=None, bos_token='<bos>', eos_token='<eos>', 
#                                pad_token='<PAD>', teacher_force=False, verbose=False):
#     """
#     Generate sequences step-by-step using a sequence model with autoregressive or teacher-forced generation.
#     The input x_ids already contains the <bos> token followed by the prompt.
    
#     Args:
#         model: The sequence model
#         x_ids: Input token IDs including <bos> [batch_size, input_seq_len]
#         embed: Embedding layer
#         char_to_id: Dictionary mapping characters to token IDs
#         id_to_char: Dictionary mapping token IDs to characters
#         max_seq_len: Maximum length of generated sequence
#         device: Device to run the model on
#         criterion: Loss function (optional)
#         y_ids: Target token IDs for teacher forcing and loss calculation [batch_size, target_seq_len]
#         bos_token: Beginning of sequence token
#         eos_token: End of sequence token
#         pad_token: Padding token
#         teacher_force: Whether to use teacher forcing
#         verbose: Whether to print generation progress
#     Returns:
#         Tuple containing generated strings, full sequences, token IDs, probabilities and metrics
#     """
#     B, Lx = x_ids.size()
#     device = x_ids.device
    
#     # Special token IDs
#     bos_id = char_to_id[bos_token]
#     eos_id = char_to_id[eos_token]
#     pad_id = char_to_id[pad_token]
    
#     # Embed the input sequence which already includes <bos>
#     x_emb = embed(x_ids)  # [B, Lx, E]
    
#     # Initialize states and outputs
#     memory = None
#     hidden = None
#     generated_ids = [[] for _ in range(B)]
#     probs_batch = [[] for _ in range(B)]
#     all_logits = []
#     all_targets = []
    
#     # Get initial prediction from the input sequence
    
#     logits, memory, hidden = model(x_emb, hidden=hidden, memory=memory)
#     last_logits = logits[:, -1, :]  # Get predictions for next token after input
    
#     # Track which sequences have finished generating
#     finished = torch.zeros(B, dtype=torch.bool, device=device)
    
#     # Generation loop
#     for step in range(max_seq_len):
#         # Calculate probabilities for current step
#         probs = torch.softmax(last_logits, dim=-1)
        
#         # Store logits for loss calculation if targets provided
#         if y_ids is not None and step < y_ids.size(1):
#             all_logits.append(last_logits)
#             all_targets.append(y_ids[:, step])
        
#         # Select next token based on teacher forcing or model prediction
#         if teacher_force and y_ids is not None and step < y_ids.size(1):
#             next_token = y_ids[:, step]
#             next_emb = embed(next_token.unsqueeze(1))  # [B, 1, E]
#         else:
#             next_token = torch.argmax(last_logits, dim=-1)  # [B]
#             next_emb = embed(next_token.unsqueeze(1))  # [B, 1, E]
        
#         # Store generated tokens and probabilities
#         for b in range(B):
#             if not finished[b]:
#                 token_id = next_token[b].item()
#                 generated_ids[b].append(token_id)
#                 probs_batch[b].append(probs[b, token_id].item())
                
#                 # Check if sequence is finished
#                 if token_id == eos_id:
#                     finished[b] = True
        
#         # Break if all sequences have finished
#         if finished.all():
#             break
            
#         # Get next prediction using the selected token
#         logits, memory, hidden = model(next_emb, hidden=hidden, memory=memory)
#         last_logits = logits[:, -1, :]  # [B, vocab_size]
    
#     # Calculate metrics if targets provided
#     avg_loss = 0.0
#     avg_token_level_accuracy = 0.0
#     avg_sample_level_accuracy = 0.0
    
#     if y_ids is not None and criterion is not None:
#         # Stack logits and targets
#         stacked_logits = torch.stack(all_logits, dim=1)  # [B, seq_len, vocab_size]
#         stacked_targets = torch.stack(all_targets, dim=1)  # [B, seq_len]
        
#         # Calculate loss
#         loss = criterion(stacked_logits.view(-1, stacked_logits.size(-1)), 
#                         stacked_targets.view(-1))
#         avg_loss = loss.item()
        
#         # Calculate accuracies
#         predictions = torch.argmax(stacked_logits, dim=-1)  # [B, seq_len]
#         mask = (stacked_targets != bos_id) & (stacked_targets != eos_id) & (stacked_targets != pad_id)

#         # Calculate token-level accuracy, ignoring special tokens
#         token_correct = (predictions == stacked_targets) & mask
#         # token_correct = (predictions == stacked_targets).float()
#         avg_token_level_accuracy = token_correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0

#         # avg_token_level_accuracy = token_correct.mean().item()
        
#         # Sample-level accuracy (all tokens correct)
#         sample_correct = token_correct.all(dim=1).float()
#         avg_sample_level_accuracy = sample_correct.mean().item()
    
#     # Convert generated IDs to strings
#     generated_strs = []
#     generated_strs_with_all_special_tokens = []
    
#     for seq_ids in generated_ids:
#         # Remove EOS token and everything after it
#         if eos_id in seq_ids:
#             seq_ids = seq_ids[:seq_ids.index(eos_id)]
            
#         # Convert to string without special tokens
#         clean_str = tensor_to_string(torch.tensor(seq_ids), id_to_char)
#         generated_strs.append(clean_str)
        
#         # Convert to string with special tokens
#         full_str = tensor_to_string(torch.tensor([bos_id] + seq_ids + [eos_id]), id_to_char)
#         generated_strs_with_all_special_tokens.append(full_str)
    
#     return (
#         generated_strs,
#         generated_strs_with_all_special_tokens,
#         generated_ids,
#         probs_batch,
#         avg_loss,
#         avg_token_level_accuracy,
#         avg_sample_level_accuracy,
#     )

def generate_sequence_batched(model, x_ids, embed, char_to_id, id_to_char, max_seq_len=20, device='cuda', 
                            criterion=None, y_ids=None, bos_token='<bos>', eos_token='<eos>', 
                            pad_token='<PAD>', teacher_force=False, verbose=False):
    """Same docstring as before"""
    B, Lx = x_ids.size()
    device = x_ids.device
    
    # Special token IDs
    bos_id = char_to_id[bos_token]
    eos_id = char_to_id[eos_token]
    pad_id = char_to_id[pad_token]
    
    x_emb = embed(x_ids)  # [B, Lx, E]
    
    if isinstance(model, (Transformer, Mamba)):
    #     # LLM-style generation
    #     generated_ids = [[] for _ in range(B)]
    #     probs_batch = [[] for _ in range(B)]
    #     all_logits = []
    #     all_targets = []
    #     finished = torch.zeros(B, dtype=torch.bool, device=device)
        
    #     # Initial state (kv_cache for transformer, hidden for mamba)
    #     state = None
    #     curr_input = x_emb
        
    #     for step in range(max_seq_len):
    #         # Forward pass with state
    #         logits, state, _ = model(curr_input, hidden=state)
    #         last_logits = logits[:, -1, :]
            
    #         # Calculate probabilities
    #         probs = torch.softmax(last_logits, dim=-1)
            
    #         # Store logits if we have targets
    #         if y_ids is not None and step < y_ids.size(1):
    #             all_logits.append(last_logits)
    #             all_targets.append(y_ids[:, step])
            
    #         # Select next token
    #         if teacher_force and y_ids is not None and step < y_ids.size(1):
    #             next_token = y_ids[:, step]
    #         else:
    #             next_token = torch.argmax(last_logits, dim=-1)
            
    #         # Store generated tokens and probabilities
    #         for b in range(B):
    #             if not finished[b]:
    #                 token_id = next_token[b].item()
    #                 generated_ids[b].append(token_id)
    #                 probs_batch[b].append(probs[b, token_id].item())
    #                 if token_id == eos_id:
    #                     finished[b] = True
            
    #         if finished.all():
    #             break
                
    #         # Prepare next input (just the new token for LLMs)
    #         next_emb = embed(next_token.unsqueeze(1))
    #         curr_input = next_emb
        # LLM-style generation
        generated_ids = x_ids.tolist()  # Initialize with input tokens
        probs_batch = [[] for _ in range(B)]
        all_logits = []
        all_targets = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for step in range(max_seq_len):
            try:
                # Convert current sequence to embeddings
                curr_ids = torch.tensor(generated_ids, device=device)
                curr_emb = embed(curr_ids)  # [B, L_so_far, E]
                
                # Forward pass
                logits, _, _ = model(curr_emb)
                last_logits = logits[:, -1, :]  # Get logits for the last token
                
                # Calculate probabilities
                probs = torch.softmax(last_logits, dim=-1)
                
                # Store logits and targets if teacher-forcing
                if y_ids is not None and step < y_ids.size(1):
                    all_logits.append(last_logits)
                    all_targets.append(y_ids[:, step])
                
                # Select next token
                if teacher_force and y_ids is not None and step < y_ids.size(1):
                    next_token = y_ids[:, step]
                else:
                    next_token = torch.argmax(last_logits, dim=-1)  # Greedy decoding
                
                # Update generated sequences
                for b in range(B):
                    if not finished[b]:
                        token_id = next_token[b].item()
                        generated_ids[b].append(token_id)
                        probs_batch[b].append(probs[b, token_id].item())
                        if token_id == eos_id:
                            finished[b] = True
                
                if finished.all():
                    break
            except Exception as e:
                print(e)

            
    else:
        # Original RNN-style generation (keep existing code)
        memory = None
        hidden = None
        generated_ids = [[] for _ in range(B)]
        probs_batch = [[] for _ in range(B)]
        all_logits = []
        all_targets = []
        
        logits, memory, hidden = model(x_emb, hidden=hidden, memory=memory)
        last_logits = logits[:, -1, :]
        
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for step in range(max_seq_len):
            probs = torch.softmax(last_logits, dim=-1)
            
            if y_ids is not None and step < y_ids.size(1):
                all_logits.append(last_logits)
                all_targets.append(y_ids[:, step])
            
            if teacher_force and y_ids is not None and step < y_ids.size(1):
                next_token = y_ids[:, step]
                next_emb = embed(next_token.unsqueeze(1))
            else:
                next_token = torch.argmax(last_logits, dim=-1)
                next_emb = embed(next_token.unsqueeze(1))
            
            for b in range(B):
                if not finished[b]:
                    token_id = next_token[b].item()
                    generated_ids[b].append(token_id)
                    probs_batch[b].append(probs[b, token_id].item())
                    if token_id == eos_id:
                        finished[b] = True
            
            if finished.all():
                break
                
            logits, memory, hidden = model(next_emb, hidden=hidden, memory=memory)
            last_logits = logits[:, -1, :]
    
    # Rest of function (metrics calculation and string conversion) stays exactly the same
    avg_loss = 0.0
    avg_token_level_accuracy = 0.0
    avg_sample_level_accuracy = 0.0
    
    if y_ids is not None and criterion is not None:
        stacked_logits = torch.stack(all_logits, dim=1)
        stacked_targets = torch.stack(all_targets, dim=1)
        
        loss = criterion(stacked_logits.view(-1, stacked_logits.size(-1)), 
                        stacked_targets.view(-1))
        avg_loss = loss.item()
        
        predictions = torch.argmax(stacked_logits, dim=-1)
        mask = (stacked_targets != bos_id) & (stacked_targets != eos_id) & (stacked_targets != pad_id)
        token_correct = (predictions == stacked_targets) & mask
        avg_token_level_accuracy = token_correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0
        
        sample_correct = token_correct.all(dim=1).float()
        avg_sample_level_accuracy = sample_correct.mean().item()
    
    # Convert to strings
    generated_strs = []
    generated_strs_with_all_special_tokens = []
    
    for seq_ids in generated_ids:
        if eos_id in seq_ids:
            seq_ids = seq_ids[:seq_ids.index(eos_id)]
            
        clean_str = tensor_to_string(torch.tensor(seq_ids), id_to_char)
        generated_strs.append(clean_str)
        
        full_str = tensor_to_string(torch.tensor(seq_ids + [eos_id]), id_to_char)
        generated_strs_with_all_special_tokens.append(full_str)
    
    return (
        generated_strs,
        generated_strs_with_all_special_tokens,
        generated_ids,
        probs_batch,
        avg_loss,
        avg_token_level_accuracy,
        avg_sample_level_accuracy,
    )

class WarmupScheduler:
        def __init__(self, optimizer, warmup_steps, base_lr, final_lr):
            """
            Args:
                optimizer (torch.optim.Optimizer): The optimizer whose LR needs to be warmed up.
                warmup_steps (int): Number of warmup steps.
                base_lr (float): Initial learning rate (e.g., 0.0).
                final_lr (float): Target learning rate after warmup.
            """
            self.optimizer = optimizer
            self.warmup_steps = warmup_steps
            self.base_lr = base_lr
            self.final_lr = final_lr
            self.current_step = 0
    
        def step(self):
            """Perform one step of warmup."""
            self.current_step += 1
            if self.current_step <= self.warmup_steps:
                # Linearly interpolate between base_lr and final_lr
                warmup_lr = self.base_lr + (self.final_lr - self.base_lr) * (self.current_step / self.warmup_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                print(f"warmup_lr, {warmup_lr}")
    
        def finished(self):
            """Check if warmup is complete."""
            print(f"finished warmup_lr!")
            return self.current_step >= self.warmup_steps



def generate_task_data(num_samples, task, context_len, maxn, train=True, ds=None):
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
    elif task=="reverse":
        return generate_reverse_task_str(num_samples, context_len, train=train)
    elif task=="sort":
        return generate_sort_task_str(num_samples, context_len, train=train)
    elif task=="factorial":
        return generate_factorial_task_str(num_samples, context_len, max_n=maxn, train=train)
    elif task=="owt":
        return generate_openwebtext_task_str(num_samples, context_len, ds, max_n=maxn, train=train, min_total_seq_len=2*context_len) # I will sample x_id as 0:context_len and then eval on context_len:2*context_len.. TODO this is hacky and I hate it.. but ICML deadline! :(
    else:
        raise ValueError(f"Unknown task {task}")



# ++++++++ CURRICULUM LOGIC ++++++++
# We'll do a naive approach:
# For "copy" => start with input_sample_length=2, each time we see train_acc>0.95 for a consecutive # of times => +1
# For "add" => start with max_num=5, each time train_acc>0.95 => max_num+=5
# We'll track consecutive_succ
# to start the curriculum
# if copy => start with input_sample_length=2 (we ignore user param if they want, or we do min(2, user param) for train)
# if add => start with max_num=5 if user param is bigger
def maybe_update_curriculum(train_acc, current_context, current_maxnum, consecutive_succ):
    # nonlocal consecutive_succ
    threshold= 0.25
    if train_acc> threshold:
        consecutive_succ+=1
    else:
        consecutive_succ=0
    # if we pass threshold 5 times in a row => increment difficulty
    if consecutive_succ>=4:
        consecutive_succ=0
        
        new_ct= current_context+1
        print(f"[CURRICULUM] copy: increasing input_sample_length => {new_ct}")
    
        new_mn= current_maxnum+5
        print(f"[CURRICULUM]: increasing max_num => {new_mn}")
        return new_ct, new_mn, 0
    return current_context, current_maxnum, consecutive_succ

# def train_micro_batch(model, x_emb, y_ids, criterion, optimizer, mezo_flavor, args, mezo_state=None, decorrelation_matrix=None, best_seed=None):
    
#     """
#     x_emb => [micro_batch_size, max_seq_len, embed_dim]
#     y_ids => [micro_batch_size, max_seq_len]
#     returns => float loss
#     """
#     try:
#         # if optimizer == "mezo":
#         if args.tie_epsilon_to_lr_ratio>-1:
#             args.epsilon = args.tie_epsilon_to_lr_ratio * optimizer.param_groups[0]['lr']
#             model.eval()
#         if mezo_flavor == "mezo_layerwise":
#             loss_val= mezo_char_layerwise(model, x_emb, y_ids, criterion, epsilon=args.epsilon)
#         elif mezo_flavor == "warm_single_mezo": 
#             loss_val, best_seed = mezo_char_single_with_warm_start(
#                                     model, 
#                                     x_emb, 
#                                     y_ids, 
#                                     criterion, 
#                                     epsilon=args.epsilon, 
#                                     max_perturbations=1,
#                                     min_acceptable_loss_percent_diff=0.001,
#                                     init_seed=best_seed,
#                                     verbose=False,
#                                 )
#         elif mezo_flavor == "rolling_mezo": 
#             loss_val = mezo_char_single_rolling(model, x_emb, y_ids, criterion, mezo_state) # NOT TESTED! TODO

#         elif mezo_flavor == "anp":

#             loss_val, decorrelation_matrix = anp_single(model, x_emb, y_ids, criterion, epsilon=args.epsilon, decorrelation_matrix=decorrelation_matrix, verbose=False)
            
#         elif mezo_flavor == "mezo_single":
#             loss_val= mezo_char_single(model, x_emb, y_ids, criterion, epsilon=args.epsilon)
#         elif mezo_flavor == "mezo_adaptive_sampling":
#             loss_val= mezo_adaptive_sampling(model, x_emb, y_ids, criterion, optimizer, epsilon=args.epsilon)
            
#         # else:
#         #     raise Exception("No flavor")
            
    
#         else:
#             model.train()
#             optimizer.zero_grad()
#             # out, _, _= model(x_emb)
#             # B,L,V= out.size()
#             # loss= criterion(out.view(B*L, V), y_ids.view(B*L))
#             loss = teacher_forcing_loss_emb(model, x_emb, y_ids, criterion, backward=True)
#             loss.backward()
#             loss_val= loss.item()
#         return loss_val, decorrelation_matrix, best_seed
#     except torch.cuda.OutOfMemoryError as e:
#         print(f"CUDA OOM: {str(e)}")
#         torch.cuda.empty_cache()
#         return float('nan'), decorrelation_matrix, best_seed  # WandB will ignore NaN values in plots

#     except Exception as e:
#         print(f"Unexpected error in train_micro_batch: {str(e)}")
#         torch.cuda.empty_cache()
#         return float('nan'), decorrelation_matrix, best_seed  


def prepare_model_for_fast_inference(model, dummy_data, optim="sgd"):
    """
    1) model.eval()
    2) flatten_parameters() on any LSTM module
    3) possibly torch.compile() if PyTorch >=2.0
    """
    import torch
    import torch.backends.cudnn as cudnn
    
    cudnn.benchmark = True
    
    
    # 1) Run a dummy pass:
    model(dummy_data)  # or however your LSTM is called
    
    # 2) Then flatten
    for module in model.modules():
        if isinstance(module, torch.nn.LSTM):
            module.flatten_parameters()
    
    # 3) Now compile
    if optim=="sgd":
        model = torch.compile(model, mode="reduce-overhead")
    # Flatten LSTM parameters if any
    for m in model.modules():
        if isinstance(m, torch.nn.LSTM):
            m.flatten_parameters()

    
    return model




##############################################################################
# Main
##############################################################################
def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="ntm", choices=["ntm","dnc","tra", "tdnc", "tntm", "simplelstm", "mamba"])
    parser.add_argument("--task", type=str, default="copy",
                        choices=["copy","repeat_copy","associative_recall","add","sub","mul","div","fib","factorial","owt"])
    parser.add_argument("--input_sample_length", type=int, default=150,
                        help="Base length for generating tasks. We'll do a simple curriculum on some tasks.")
    parser.add_argument("--max_seq_len", type=int, default=150,
                        help="For generation.")

    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--macro_batch_size", type=int, default=1)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--max_num", type=int, default=110,
                        help="This is the max number in the domain to use in training for arithmetic tasks. Min in the train domain is 0. We'll do a simple curriculum for arithmetic if task in all. i.e. [add,sub,mul,div].")

    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--memory_size", type=int, default=128)
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=1)

    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd","mezo"])
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--tie_epsilon_to_lr_ratio", type=float, default=-1)
    parser.add_argument("--epsilon", type=float, default=1e-2, help="MeZO eps.")

    parser.add_argument("--mezo_flavor", type=str, default="None", choices=["mezo_single","mezo_layerwise", "mezo_rolling", "anp", "warm_single_mezo", "mezo_adaptive_sampling", "mezo_adaptive_sampling_fast", "mezo_single_fast", "None", "mezo_adaptive_sampling_parallel"])

    parser.add_argument("--fixed_size_perturbation", action="store_true")
    
    parser.add_argument("--cosine_lr", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=100)

    parser.add_argument("--grad_norm", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="If >0, grad norm clipped.")

    
    parser.add_argument("--pad_bias", type=float, default=0.0, help="Initial logit bias for <PAD> in final layer. NOT IMPLEMENTED YET")
    parser.add_argument("--log_interval", type=int, default=300)
    parser.add_argument("--wandb_proj", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--minimum_starting_point_context_length", type=int, default=100, help="min seq length fed into the model to start the curriculum learning")
    
    args= parser.parse_args()


    verbose = False 

    total_samples_per_iter = args.micro_batch_size * args.macro_batch_size

    # pick device
    if torch.cuda.is_available():
        torch.cuda.init()
        gpu_index= pick_gpu_with_most_free_mem()
        device= torch.device(f"cuda:{gpu_index}")
        print(f"[INFO] Using GPU: {gpu_index}")
    else:
        device= torch.device("cpu")
        print("[INFO] Using CPU")

    # build vocab
    vocab_list, char_to_id, id_to_char = get_char_vocab()
    vocab_size= len(vocab_list)

    if args.optimizer == "sgd":
        mezo_flavor = "sgd"
    else:
        mezo_flavor = args.mezo_flavor
        
    vram_stats = calculate_vram_usage_direct(
        args.arch,
        args.hidden_size,
        args.memory_size,
        args.head_size,
        args.num_heads,
        args.input_size,
        vocab_size,
        args.micro_batch_size,
        args.input_sample_length,
        mezo_flavor,
        True
    )
    total_estimated_vram_gb = vram_stats["total_estimated_gb"]

    # wandb
    if args.wandb_proj is not None:
        msg = vars(args)
        msg["total_estimated_vram_gb"] = total_estimated_vram_gb
        msg["total_samples_per_iter"] = total_samples_per_iter
        # setattr(args, "total_estimated_vram_gb", total_estimated_vram_gb)
        # setattr(args, "total_samples_per_iter", total_samples_per_iter)
    
        # For neptune:
        # run = neptune.init_run(
        #                     project="fchaubard/mezornn",
        #                     api_token=neptune_api_token,
        #                     name=args.wandb_run_name
        #                 )  # your credentials
        # run["parameters"] = args

        # For custom mongo db:
        # run = init_run_mongo(
        #             db_url=db_url,
        #             db_name=db_name,
        #             project=db_project,
        #             name=args.wandb_run_name,
        #             params=msg
        #         )



        # For wandb:
        wandb.init(project=args.wandb_proj, name=args.wandb_run_name)
        wandb.config.update(msg)
        
        print(f"Logging to run:{args.wandb_run_name}")
        print(f"w/ config:{msg}")

    # embed
    torch.cuda.reset_peak_memory_stats(device)
    # embed= nn.Embedding(vocab_size, args.input_size, padding_idx=0).to(device)
    embed= nn.Embedding(vocab_size, args.input_size).to(device)
    nn.init.orthogonal_(embed.weight)
    
    # model
    if args.arch == "ntm":
        model = NTM(args.input_size, vocab_size, args.hidden_size, args.memory_size, args.head_size, args.num_heads, embed).to(device)
    elif args.arch == "dnc":
        model = DNC(args.input_size, vocab_size, args.hidden_size, args.memory_size, args.head_size, args.num_heads, embed).to(device)
    elif args.arch == "tdnc":
        # TODO NOT TESTED YET
        model = TransformerMemoryDNC(args.input_size, vocab_size, args.hidden_size, args.memory_size, args.head_size, args.num_heads, embed).to(device)
    elif args.arch == "mamba":
        # TODO NOT TESTED YET
        model = Mamba(args.input_size, vocab_size, args.hidden_size, args.memory_size, args.head_size, args.num_heads, embed).to(device)
    elif args.arch == "tntm":
        # TODO NOT TESTED YET
        model = TransformerMemoryNTM(args.input_size, vocab_size, args.hidden_size, args.memory_size, args.head_size, args.num_heads, embed).to(device)    
    elif args.arch == "simplelstm":
        model = SimpleLSTM(args.input_size, vocab_size, args.hidden_size, embed).to(device)  
    else: # "tra"
        model = Transformer(args.input_size, vocab_size, args.hidden_size, args.memory_size, args.head_size, args.num_heads, embed).to(device) 
        
    # build optimizer
    params= list(model.parameters())+ list(embed.parameters())
    if args.optimizer=="sgd": 
        optimizer= optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay,  betas=(0.9, 0.999), eps=1e-08, amsgrad=False )
    else:
        # mezo uses the same optimizer for momentum, we'll do param.grad => momentum => param.data
        # optimizer= optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        optimizer= optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay,  betas=(0.9, 0.999), eps=1e-08, amsgrad=False )

    scheduler= None
    if args.cosine_lr:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iters, eta_min=args.learning_rate/20)    
    else: 
        # default is LR plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',  # Minimize the validation loss
                    factor=0.5,  # Reduce the LR by a factor of 0.5
                    patience=100,  # Number of validation epochs with no improvement
                    verbose=True,  # Log the change in LR
                    min_lr=1e-8   # Minimum learning rate
                )
        
    
    criterion= nn.CrossEntropyLoss(ignore_index=0)
    global_step=0

    # track time
    train_start_time= time.time()
    consecutive_succ=0
    gen_efficiency_gen_loss = 0
    gen_efficiency_val_loss = 0
    decorrelation_matrix = None
    
   
    minimum_starting_point_context_length = args.minimum_starting_point_context_length
    current_context_len = minimum_starting_point_context_length
    current_max_num= 15 # used only in arithmetic functions

    # lets make sure the user knows how the curriculum works
    assert current_max_num <= args.max_num, "Must have max_num > 15"
    assert current_context_len <= args.input_sample_length, f"Must have input_sample_length > 1: current_context_len:{current_context_len} and input_sample_length:{args.input_sample_length}"

    mezo_state = None
    if args.mezo_flavor == "mezo_rolling": 
        # TODO, not tested
        mezo_state = init_rolling_mezo(model, args.epsilon)

    # Warmup scheduler
    warmup_scheduler = WarmupScheduler(
        optimizer=optimizer,
        warmup_steps= args.warmup_steps,
        base_lr=1e-9,
        final_lr=args.learning_rate
    )

    best_seed = None

    print("Optimizing the model")
    
    dummy_data = torch.zeros(args.micro_batch_size,
        args.minimum_starting_point_context_length,
        args.input_size,
        device=device
    )

    
    # verbose = True
    if args.optimizer=="sgd":
        # optimize the model
        prepare_model_for_fast_inference(model, dummy_data)
    else:
        # you can use half prec if using 
        # model = model.half()
        # dummy_data = dummy_data.half()
        model.eval()
        with torch.no_grad():
            prepare_model_for_fast_inference(model, dummy_data)
    print(torch.cuda.max_memory_allocated(device) / (1024 ** 3), "GiB")
    print("Memory stats:")
    print(torch.cuda.memory_summary(device))
    print("Done")
    ####################################
    # TRAIN LOOP
    ####################################
    #for iteration in range(1, args.max_iters+1):
    iteration = -1

    ds = None
    if args.task == "owt":
        iterr = 0
        while True:
            try:
                ds = load_dataset(
                    "haritzpuerto/the_pile_00_OpenWebText2",
                    split="train",
                    # cache_dir="/hf_cache/",
                    # use_auth_token=False,  # Disable authentication to avoid API calls
                    download_mode="reuse_dataset_if_exists"  # Reuse the cached dataset
                )
                break
            except Exception as e:
                print("Hugging face issue...")
                print(e)
                time.sleep(5)
                iterr+=1
                if iterr>100:
                    raise Exception("HUGGING FACE ISSUES AGAIN!")
        print("Got the OWT dataset!")
                

    while True:
        #torch.cuda.reset_peak_memory_stats(device)        
        
        iteration+=1
        iter_start_time= time.time()
        
        # generate data
        # we do a curriculum for train
        this_sample_context_length = 1 + np.random.randint( 
                                            minimum_starting_point_context_length, 
                                            max(1,current_context_len+1)
                        ) # always at least generate 1.
        
        this_sample_max_num = 1+np.random.randint(0,max(1,current_max_num)) # always at least generate 1.
        x_strs, y_strs= generate_task_data(total_samples_per_iter, args.task,
                                           this_sample_context_length,
                                           this_sample_max_num,
                                           train=True,
                                            ds = ds
                                          )
        
        model.zero_grad()
        embed.zero_grad()

        micro_loss_sum= 0.0

        # micro/macro approach
        for micro_i in range(args.macro_batch_size):
            # try:
                start_idx= micro_i* args.micro_batch_size
                end_idx= start_idx+ args.micro_batch_size
                cur_x= x_strs[start_idx:end_idx]
                cur_y= y_strs[start_idx:end_idx]
                
                # take all the excess padding out just in case
                # x_ids= str_to_tensor(cur_x, char_to_id, args.input_sample_length+1).to(device)
                # x_ids_trimmed = [x[x != 0] for x in x_ids]
                # x_ids = pad_sequence(x_ids_trimmed, batch_first=True, padding_value=0)
    
                # y_ids= str_to_tensor(cur_y, char_to_id, args.input_sample_length+1).to(device)
                # y_ids_trimmed = [y[y != 0] for y in y_ids]
                # y_ids = pad_sequence(y_ids_trimmed, batch_first=True, padding_value=0)
                
                
                x_ids= str_to_tensor(cur_x, char_to_id).to(device)
                y_ids= str_to_tensor(cur_y, char_to_id).to(device)

                # x_emb= embed(x_ids)
                x_emb =  x_ids # REALLY HACKKY! NEED TO FIX BUT JUST DOING THIS TO MAKE TEACHER FORCING WORK FOR TRANSFORMERS
                
                # loss_val, decorrelation_matrix, best_seed = train_micro_batch(model, x_emb, y_ids, criterion, optimizer, mezo_flavor, args, mezo_state, decorrelation_matrix, best_seed)
                
                if args.tie_epsilon_to_lr_ratio>-1:
                    args.epsilon = args.tie_epsilon_to_lr_ratio * optimizer.param_groups[0]['lr']
    
                if args.optimizer=="mezo":
                    with torch.inference_mode():
                        # x_emb = x_emb.half()
                        model.eval()
                        if mezo_flavor == "mezo_layerwise":
                            loss_val= mezo_char_layerwise(model, x_emb, y_ids, criterion, epsilon=args.epsilon)
                        elif mezo_flavor == "warm_single_mezo": 
                            loss_val, best_seed = mezo_char_single_with_warm_start(
                                                    model, 
                                                    x_emb, 
                                                    y_ids, 
                                                    criterion, 
                                                    epsilon=args.epsilon, 
                                                    max_perturbations=1,
                                                    min_acceptable_loss_percent_diff=0.001,
                                                    init_seed=best_seed,
                                                    verbose=False,
                                                )
                        elif mezo_flavor == "rolling_mezo": 
                            loss_val = mezo_char_single_rolling(model, x_emb, y_ids, criterion, mezo_state) # NOT TESTED! TODO
                
                        elif mezo_flavor == "anp":
                
                            loss_val, decorrelation_matrix = anp_single(model, x_emb, y_ids, criterion, epsilon=args.epsilon, decorrelation_matrix=decorrelation_matrix, verbose=False)
                            
                        elif mezo_flavor == "mezo_single":
                            loss_val= mezo_char_single(model, x_emb, y_ids, criterion, epsilon=args.epsilon)
            
                        
                        elif mezo_flavor == "mezo_single_fast":
                            loss_val= mezo_adaptive_sampling_fast(model, x_emb, y_ids, criterion, optimizer, epsilon=args.epsilon,adaptive=False, fixed_size_perturbation=args.fixed_size_perturbation)
                        elif mezo_flavor == "mezo_adaptive_sampling":
                            if iteration<=args.warmup_steps:
                                loss_val= mezo_char_single(model, x_emb, y_ids, criterion, epsilon=args.epsilon,fixed_size_perturbation=False)
                            
                            else:
                                loss_val= mezo_adaptive_sampling(model, x_emb, y_ids, criterion, optimizer, epsilon=args.epsilon,fixed_size_perturbation=args.fixed_size_perturbation)
                                
                        elif mezo_flavor == "mezo_adaptive_sampling_fast":
                            if iteration<=args.warmup_steps:
                                loss_val= mezo_adaptive_sampling_fast(model, x_emb, y_ids, criterion, optimizer, epsilon=args.epsilon, adaptive=False)
                                
                            else:
                                loss_val= mezo_adaptive_sampling_fast(model, x_emb, y_ids, criterion, optimizer, epsilon=args.epsilon, adaptive=True, fixed_size_perturbation=False)
                        # elif mezo_flavor == "mezo_adaptive_sampling_parallel":
                        # JUST A BIT FASTER BUT NOT CONVERGING AS WELL SO JUST LEAVE IT ALONE..    
                        #     if iteration<=args.warmup_steps:
                        #         loss_val = mezo_adaptive_sampling_parallel(
                        #             model,
                        #             x_emb,
                        #             y_ids,
                        #             criterion,
                        #             optimizer,
                        #             epsilon=args.epsilon,
                        #             adaptive=False,
                        #             fixed_size_perturbation=True
                        #         )
                                
                        #     else:
                        #         loss_val = mezo_adaptive_sampling_parallel(
                        #             model,
                        #             x_emb,
                        #             y_ids,
                        #             criterion,
                        #             optimizer,
                        #             epsilon=args.epsilon,
                        #             adaptive=True,
                        #             fixed_size_perturbation=True
                        #         )
                                

                        
                    
                        else:
                            raise Exception("No flavor")
                            
                
                else:
                    model.train()
                    optimizer.zero_grad()
                    # out, _, _= model(x_emb)
                    # B,L,V= out.size()
                    # loss= criterion(out.view(B*L, V), y_ids.view(B*L))
                    loss = teacher_forcing_loss_emb(model, x_emb, y_ids, criterion)
                    
                    loss.backward()
                    
                    loss_val= loss.item()
    
                micro_loss_sum+= loss_val
                #torch.cuda.empty_cache()
    
    
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
            # except torch.cuda.OutOfMemoryError as e:
            #     print(f"CUDA OOM: {str(e)}")
            #     torch.cuda.empty_cache()
            #     return float('nan'), decorrelation_matrix, best_seed  # WandB will ignore NaN values in plots
        
            # except Exception as e:
            #     print(f"Unexpected error: {str(e)}")
            #     torch.cuda.empty_cache()
            #     return float('nan'), decorrelation_matrix, best_seed  
                

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
        
        # # finally we step! 
        # if False:
        #     # check to see if any of the grads are 0 that means training is not happening.
        #     params = list(model.parameters()) + list(embed.parameters())

        #     for i, param in enumerate(params):
        #         if param.grad is not None:
        #             grad_abs_sum = param.grad.abs().sum().item()
        #             if grad_abs_sum == 0:
        #                 print(f"Param {i}: Gradient sum is 0. Name: {param.shape}")
        #         else:
        #             print(f"Param {i}: No gradient calculated (None). Name: {param.shape}")
        #     pdb.set_trace()
        optimizer.step()

        train_loss_mezo = micro_loss_sum / args.macro_batch_size

        # Warmup step, TODO, not sure how much this adds TBH.. should ablate
        if iteration <= args.warmup_steps:
            warmup_scheduler.step()
        elif scheduler is not None:
            if args.cosine_lr:
                scheduler.step()
            else:
                scheduler.step(train_loss)
                
    
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
        vram_inferred = torch.cuda.max_memory_allocated(device)/1024**3

        msg= (f"Iter={iteration}, train_loss={train_loss_mezo:.3f}, "
              f"LR={lr_current:.6f}, eps={args.epsilon:.6f}, vram_inferred={vram_inferred:.6f} GB iter_time={iteration_time:.2f}s, total_time={total_elapsed/60:.2f}m, "
              f"context_len={current_context_len}, max_num={current_max_num}, gen_eff_token={gen_efficiency_gen_loss}, gen_eff_sample={gen_efficiency_val_loss}")

        print(msg)
        sys.stdout.flush()

        if train_loss_mezo>1000:
            raise Exception(f"Ending training train_loss_mezo diverging: {train_loss_mezo}")

        ####################################
        # VALIDATION LOOP
        ####################################
        # validation every log_interval
        if iteration % args.log_interval == 0:

            # compute train accuracy on last micro-batch
            with torch.inference_mode():
                    # x_ids = str_to_tensor(cur_x, char_to_id, args.max_seq_len).to(device)  # [B, Lx]
                    # y_ids = str_to_tensor(cur_y, char_to_id, args.max_seq_len).to(device)  # [B, Ly]
                    x_ids = str_to_tensor(cur_x, char_to_id).to(device)  # [B, Lx]
                    y_ids = str_to_tensor(cur_y, char_to_id).to(device)  # [B, Ly]
    
                    # iterate on the train just to show accuracy as mezo makes it difficult to get a good gauge on accuracy as the model is not ever truly represented

                    
                    # fallback to your old code for RNN
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
    
                    # Generate Val Samples that are hold out numbers and seq length
                
        
                    this_sample_context_length_for_val = 1 + np.random.randint( 
                                            minimum_starting_point_context_length, 
                                            max(1,current_context_len+1)
                        ) # always at least generate 1.
        
                    ### TURN THIS BACK ON IF YOU WANT TO SAMPLE FULL LENGTH.. BUT ITS SLOW..
                    # this_sample_context_length_for_val = 1+np.random.randint(args.input_sample_length, args.input_sample_length)
        
                    this_sample_max_num_for_val = 1+np.random.randint(0,max(args.max_num,args.max_num)) # always at least generate 1.
                    
                    # TODO, maybe this should be different? can update later if we want
                    val_samples = 5 # total_samples_per_iter 
                    
                    # Generate a validation batch (vx, vy)
                    vx, vy= generate_task_data(val_samples, 
                                                   args.task,
                                                   this_sample_context_length_for_val, 
                                                   this_sample_max_num_for_val,
                                                   train=False,
                                                   ds = ds
                                              )
                    
            
                    # Convert to tensors
                    # vx_ids= str_to_tensor(vx, char_to_id, args.max_seq_len).to(device)
                    # vy_ids= str_to_tensor(vy, char_to_id, args.max_seq_len).to(device)
                    vx_ids= str_to_tensor(vx, char_to_id).to(device)
                    vy_ids= str_to_tensor(vy, char_to_id).to(device)
            
                    # --------------------------------------------------------------------
                    # 1) Optionally, do a teacher-forced pass to measure "val_loss"
                    # --------------------------------------------------------------------

                    # vx_emb = embed(vx_ids)
                    vx_emb = vx_ids # VERY HACKKY TODO SAME REASON AS ABOVE
                
                    #             vy_emb = embed(vy_ids)[:, :-1, :]  # Exclude last token from input since we'll predict it
                    #             v_full = torch.cat([vx_emb, vy_emb], dim=1)  # Concatenate along sequence length dimension
                                
                    #             Bx, Lx, Vx = vx_emb.size()
                    #             model.eval()
                    #             outputs, _, _ = model(v_full)
                    #             B2,L2,V2= outputs.size()
                                
                    #             logits = outputs[:, Lx-1:, :].contiguous()  # Get predictions starting from after input sequence
                    #             logits = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, num_classes]
                    
                    # # Reshape targets
                    #             targets = vy_ids.contiguous().view(-1)  # [batch_size * seq_len]
                
                    #             val_loss= criterion(logits, targets)
                    val_loss = teacher_forcing_loss_emb(model, vx_emb, vy_ids, criterion)

                
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

            # For debugging, let's print a few val random samples
            sample_indices= random.sample(range(len(generated_strs)), min(3,len(generated_strs)))
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

            gen_efficiency_gen_loss = val_gen_loss * 100.0 / (total_estimated_vram_gb * (total_elapsed / 3600.0))
            gen_efficiency_val_loss = val_loss * 100.0 / (total_estimated_vram_gb * (total_elapsed / 3600.0))
            print(f"Generation loss: {val_gen_loss}, Generation accuracy: {val_gen_acc}, Generation sample accuracy: {val_gen_acc_sample}")
            print(f"Generalization Efficiency:")
            print(f"    VRAM: {total_estimated_vram_gb}")
            print(f"    Wall-Clock Hrs: {total_elapsed / 3600.0}")
            print(f"    val_gen_acc: {val_gen_acc}")
            print(f"    val_gen_acc_sample: {val_gen_acc_sample}")
            print(f"    Gen Eff (token): {gen_efficiency_gen_loss}")
            print(f"    Gen Eff (sample): {gen_efficiency_val_loss}")
                
            print("="*30)
            print("="*30)

            print("[END DEBUG]\n")
            
            # check to see if we should update curriculum
            new_ctx, new_mn, consecutive_succ= maybe_update_curriculum(train_acc, current_context_len, current_max_num, consecutive_succ)
            if current_context_len!=new_ctx:
                print(f"!!!!! UPDATING CURRICULUM FROM {current_context_len} to {new_ctx}")
                print(f"!!!!! UPDATING CURRICULUM FROM {current_context_len} to {new_ctx}")
                print(f"!!!!! UPDATING CURRICULUM FROM {current_context_len} to {new_ctx}")
                # pdb.set_trace()
            current_context_len= new_ctx
            current_max_num= new_mn

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
                    "total_time_hours": total_elapsed / 3600.0,
                    "curr_context_len": current_context_len,
                    "curr_max_num": current_max_num,
                    "val_loss": val_loss.item(),        # teacher-forced
                    "val_gen_loss": val_gen_loss,       # from generation
                    "val_gen_acc": val_gen_acc,
                    "total_estimated_vram_gb":total_estimated_vram_gb,
                    "GPU(GB)-Hours":total_estimated_vram_gb*total_elapsed / 3600.0,
                    "gen_efficiency_gen_loss":gen_efficiency_gen_loss,
                    "gen_efficiency_val_loss":gen_efficiency_val_loss,
                    "weight_decay_loss": weight_decay_loss.item(),
                    "vram_inferred":vram_inferred
                    # "vram_usage":vram_usage,
                }
                print("="*30)
                print("VAL STATS")
                print(msg)
                try:
                    # if wandb
                    wandb.log(msg, step=iteration)
    
                    # if neptune or mdb
                    # for key, value in msg.items():
                    #     run[f"{key}"].log(value, step=iteration)
                    
                   

                except Exception as e:
                        print(f"logging failed at iteration {iteration}. Error: {str(e)}")

                    



    print("Finished.") 
    if args.wandb_proj is not None:
        # Finish the run
        run.finish()


if __name__=="__main__":
    main()
