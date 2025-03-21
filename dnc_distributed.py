#!/usr/bin/env python3: 
# python -m torch.distributed.launch --nproc_per_node=10 dnc_distributed.py 
# python -m torch.distributed.launch --nproc_per_node=10 --master_port=29501 dnc_distributed.py  

import os, argparse, time
import string
import torch
import torch.nn as nn
import torch.distributed as dist
from ntm_with_modern_training_runs import teacher_forcing_loss_emb_parallel
import datetime
import wandb
import math
from datasets import load_dataset
import random

CHUNK_SIZE = 1048576

class DNC(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, memory_size, head_size, num_heads, embed, device=None):
        super(DNC, self).__init__()
        with torch.inference_mode():
            # Set the device for initialization
            self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Move the model to the specified device immediately
            self.to(self.device)
            
            self.input_size = input_size
            self.output_size = output_size  # This should be vocab_size
            self.hidden_size = hidden_size
            self.memory_size = memory_size
            self.head_size = head_size
            self.num_reads = num_heads
            self.embed = embed
    
            # Input normalization
            controller_input_size = input_size + self.num_reads * self.head_size
            self.input_norm = nn.LayerNorm(controller_input_size, device=self.device)
            
            # Controller with normalization
            self.controller = nn.LSTM(controller_input_size, hidden_size, batch_first=True, device=self.device)
            self.controller_norm = nn.LayerNorm(hidden_size, device=self.device)
    
            # Memory operation layers with normalization
            self.fc_read_keys = nn.Linear(hidden_size, self.num_reads * self.head_size, device=self.device)
            self.fc_write_keys = nn.Linear(hidden_size, self.head_size, device=self.device)
            self.fc_write_strength = nn.Linear(hidden_size, 1, device=self.device)
            self.fc_erase_vector = nn.Linear(hidden_size, self.head_size, device=self.device)
            self.fc_add_vector = nn.Linear(hidden_size, self.head_size, device=self.device)
    
            self.read_keys_norm = nn.LayerNorm(head_size, device=self.device)
            self.write_keys_norm = nn.LayerNorm(head_size, device=self.device)
            self.memory_norm = nn.LayerNorm(head_size, device=self.device)
    
            # Output projection with normalization - project directly to vocab size
            total_output_size = hidden_size + self.num_reads * self.head_size
            self.pre_output_norm = nn.LayerNorm(total_output_size, device=self.device)
            self.fc_proj = nn.Linear(total_output_size, output_size, device=self.device) # Project directly to vocab size
    
            # Initialize parameters on GPU
            self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters directly on the GPU"""
        # Initialize LSTM params
        for name, p in self.controller.named_parameters():
            if 'weight' in name:
                init_tensor = torch.zeros_like(p, device=self.device)
                nn.init.orthogonal_(init_tensor)
                p.data.copy_(init_tensor)
            elif 'bias' in name:
                p.data.zero_()
    
        # Initialize memory operation layers
        for name, p in self.named_parameters():
            if 'fc_' in name and 'weight' in name:
                init_tensor = torch.zeros_like(p, device=self.device)
                nn.init.xavier_uniform_(init_tensor)
                p.data.copy_(init_tensor)
            elif 'fc_' in name and 'bias' in name:
                p.data.zero_()

    def _read_memory(self, memory, read_keys):
        """Read from memory using normalized attention."""
        # Normalize memory and keys
        memory_normalized = self.memory_norm(memory)
        read_keys_norm = self.read_keys_norm(read_keys.view(-1, self.head_size)).view(-1, self.num_reads, self.head_size)

        # Compute attention weights (no scaling, let LayerNorm handle it)
        read_weights = torch.einsum('bnh,bmh->bnm', read_keys_norm, memory_normalized)
        read_weights = torch.softmax(read_weights, dim=2)
        read_vectors = torch.einsum('bnm,bmh->bnh', read_weights, memory)
        
        # Free temporary tensors immediately
        del memory_normalized, read_keys_norm, read_weights
        
        return read_vectors

    def _write_memory(self, memory, write_keys, write_str, erase_vec, write_vec):
        """Write to memory using normalized attention with in-place chunked updates to limit temporary VRAM usage."""
        # Normalize memory and keys
        memory_normalized = self.memory_norm(memory)
        write_keys_norm = self.write_keys_norm(write_keys)
    
        # Compute write weights
        write_scores = torch.einsum('bh,bmh->bm', write_keys_norm, memory_normalized)
        write_weights = torch.softmax(write_scores, dim=1).unsqueeze(1)  # Shape: [B, 1, memory_size]
        
        # Free no-longer-needed tensors
        del memory_normalized, write_scores, write_keys_norm
    
        # Scale write weights by write strength (in-place)
        write_weights.mul_(write_str.unsqueeze(1))
        
        # Compute erase and write contributions
        erase = torch.einsum('bnm,bh->bmh', write_weights, erase_vec)
        write = torch.einsum('bnm,bh->bmh', write_weights, write_vec)
        
        # Free write_weights as it's no longer needed
        del write_weights
    
        # Update memory in small chunks along the memory dimension
        chunk_size = 128  # Adjust based on your memory constraints
        for i in range(0, memory.size(1), chunk_size):
            cur_slice = slice(i, i + chunk_size)
            mem_chunk = memory[:, cur_slice, :]
            erase_chunk = erase[:, cur_slice, :]
            write_chunk = write[:, cur_slice, :]
            
            # Instead of computing "1 - erase_chunk" over the whole memory,
            # allocate a small temporary buffer for the current chunk.
            temp_buf = mem_chunk.new_empty(erase_chunk.size())
            temp_buf.fill_(1.0)
            temp_buf.sub_(erase_chunk)  # Now temp_buf = 1 - erase_chunk (in-place)
            
            # In-place update of the memory chunk:
            # mem_chunk = mem_chunk * (1 - erase_chunk) + write_chunk
            mem_chunk.mul_(temp_buf)
            mem_chunk.add_(write_chunk)
            
            # Free the temporary buffer for this chunk
            del temp_buf
    
        # Clean up intermediate tensors
        del erase, write
    
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
            
            # Controller forward pass
            out_ctrl, hidden = self.controller(controller_input.unsqueeze(1), hidden)
            h = self.controller_norm(out_ctrl.squeeze(1))

            # Generate memory parameters
            read_keys = self.fc_read_keys(h).view(B, self.num_reads, self.head_size)
            write_keys = self.fc_write_keys(h)
            write_str = torch.sigmoid(self.fc_write_strength(h))
            erase_vec = torch.sigmoid(self.fc_erase_vector(h))
            write_vec = torch.tanh(self.fc_add_vector(h))

            # Update memory with in-place write
            memory = self._write_memory(memory, write_keys, write_str, erase_vec, write_vec)
            read_vectors = self._read_memory(memory, read_keys)
            read_vec = read_vectors.reshape(B, -1)

            # Project output directly to logits (vocab size)
            output = torch.cat([h, read_vec], dim=-1)
            output = self.pre_output_norm(output)
            logits = self.fc_proj(output)
            outputs.append(logits.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs, memory, hidden




# =============================================================================
# Logging Helpers
# =============================================================================
def current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def log_start(stage, rank):
    print(f"[Rank {rank}] {current_time()} - START {stage}\n", end='', flush=True)

def log_end(stage, rank):
    print(f"[Rank {rank}] {current_time()} - END {stage}\n", end='', flush=True)

def log_msg(stage, rank, msg):
    print(f"[Rank {rank}] {current_time()} - {stage}: {msg}\n", end='', flush=True)

# =============================================================================
# Persistent Group: Create once for all ranks except rank 0
# =============================================================================
def create_group_except_rank0():
    world_size = dist.get_world_size()
    ranks = list(range(1, world_size))
    return dist.new_group(ranks=ranks)

# =============================================================================
# Helper: Broadcast within a given group (persistent group used)
# =============================================================================
def broadcast_in_group(tensor, src_rank, group):
    dist.broadcast(tensor, src=src_rank, group=group)
    # dist.barrier()
    return tensor

# =============================================================================
# Helper: Reduce Mean within a group
# =============================================================================
def reduce_mean_in_group(tensor, dst_rank, group):
    dist.reduce(tensor, dst=dst_rank, op=dist.ReduceOp.SUM, group=group)
    if dist.get_rank() == dst_rank:
        tensor.div_(len(group.ranks))
    dist.barrier()
    return tensor


def reduce_mean_in_group(tensor, dst_rank, group, world_size):
    """
    Reduce mean across a specific process group
    
    Args:
        tensor: The tensor to reduce
        dst_rank: Destination rank (global rank numbering)
        group: Process group to use
        world_size: Total world size
    """
    # Get current rank
    rank = dist.get_rank()
    
    group_size = world_size - 2  # For a group excluding rank 0 and rank 1 which is clean
    
    # Only participate if you're in the group
    if rank != 0:          
        # Perform the reduction
        dist.reduce(tensor, dst=1, op=dist.ReduceOp.SUM, group=group)
        if dist.get_rank() == dst_rank:
            # dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
            tensor.div_(group_size)
    
    return tensor

# =============================================================================
# List-Level In-Place Operations (on lists of tensors)
# =============================================================================
def list_inplace_add(param_list, add_list, alpha=1.0, chunk_size=CHUNK_SIZE):
    for p, a in zip(param_list, add_list):
        d = p.numel()
        p_view = p.view(-1)
        a_view = a.view(-1)
        for start in range(0, d, chunk_size):
            end = min(d, start+chunk_size)
            p_view[start:end].add_(a_view[start:end], alpha=alpha)

def list_inplace_sub(param_list, sub_list, alpha=1.0, chunk_size=CHUNK_SIZE):
    for p, a in zip(param_list, sub_list):
        d = p.numel()
        p_view = p.view(-1)
        a_view = a.view(-1)
        for start in range(0, d, chunk_size):
            end = min(d, start+chunk_size)
            p_view[start:end].sub_(a_view[start:end], alpha=alpha)


# =============================================================================
# Main Class: Distributed Zero–Order Adaptive Sampling with Custom Adam
# =============================================================================
#
# Roles:
#   - Rank 0 ("adam rank"): does not create a model; obtains parameter meta from Rank 1,
#         then initializes adam_m and adam_v (on GPU) based on that meta.
#   - Rank 1 ("clean rank"): creates the full model and input data.
#   - Dirty ranks (>=2): create the model structure only (their parameters will be overwritten).

class MezoAdaptiveSamplingParallel:
    def __init__(self, 
                 model, 
                 learning_rate=0.001, 
                 probe_dropout_rate=0.99, 
                 epsilon=0.001,
                 beta1=0.9, 
                 beta2=0.999, 
                 meta_perturbations=1,
                 verbose=True):
        self.learning_rate = learning_rate
        self.probe_dropout_rate = probe_dropout_rate
        self.epsilon = epsilon  # initial; later tied 1:1 with lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.meta_perturbations = meta_perturbations
        self.verbose = verbose

        # On Rank 1 and dirty ranks, model is provided.
        self.model = model  

        rank = dist.get_rank()
        
        self.param_list = list(self.model.parameters())
        self.d = sum(p.numel() for p in self.param_list)
        
        self.adam_m = None  # this will exist only on rank 0
        self.adam_v = None  # this will exist only on rank 0
        
        self.adam_ratio_list = None # this will exist only on rank 1
        self.probe_list = None  # this will exist only on rank 2+

        # Create a persistent group for all ranks except rank 0.
        if rank == 0: # adam rank, hold both adam moments
            self.group_except_zero = None
            self.adam_m = [torch.zeros_like(p) for p in self.param_list]
            del self.param_list
            self.param_list = None
            self.adam_v = [torch.zeros_like(p) for p in self.adam_m]
        else:
            self.group_except_zero = create_group_except_rank0()
            if rank == 1: # clean rank, hold model + adam_ratio
                self.adam_ratio_list = [torch.zeros_like(p, dtype=p.dtype, device=p.device) for p in self.param_list]
            elif rank>2: # dirty ranks, hold model + probe
                self.probe_list = [torch.zeros_like(p, dtype=p.dtype, device=p.device) for p in self.param_list]

    def distributed_step(self, x_ids, y, criterion, iteration, warmup_iters=100):
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = self.param_list[0].device if self.param_list is not None else torch.device(f"cuda:{rank}")
        
        current_lr = self.learning_rate 
        self.epsilon = current_lr
        
        dist.barrier()
    
        # -------------------- Stage 0: Broadcast Clean theta_t from Rank 1 to Dirty Ranks --------------------
        if self.verbose:
            log_start("Stage 0", rank)
        if rank >= 1:
            for p in self.param_list:
                broadcast_in_group(p, src_rank=1, group=self.group_except_zero)
        if self.verbose:
            if rank >= 1:
                weight_decay_loss = 0.0
                for param in self.model.parameters():
                    weight_decay_loss += torch.sum(param )  # using 1e-2 as dummy weight_decay
                log_msg("Stage 0: weight_decay_loss: ", rank, f"{weight_decay_loss}.")
        if self.verbose:
            log_end("Stage 0", rank)
        dist.barrier()
        
        # -------------------- Stage 0c: Broadcast x_ids and y from Rank 1 to Dirty Ranks --------------------
        if self.verbose:
            log_start("Stage 0c", rank)
        
        if rank >= 1:
            if rank == 1:
                x_shape = torch.tensor([x_ids.size(0), x_ids.size(1)], device=device, dtype=torch.long)
                y_shape = torch.tensor([y.size(0), y.size(1)], device=device, dtype=torch.long)
                x_ids = x_ids.to(device)
                y = y.to(device)
                if self.verbose:
                    log_msg("Stage 0c", rank, f"Broadcasting shapes - x: {x_shape}, y: {y_shape}")
                    log_msg("Stage 0c", rank, f"Broadcasting data - x: {x_ids.shape}, y: {y.shape}")
            else:
                x_shape = torch.zeros(2, device=device, dtype=torch.long)
                y_shape = torch.zeros(2, device=device, dtype=torch.long)
        
            broadcast_in_group(x_shape, src_rank=1, group=self.group_except_zero)
            broadcast_in_group(y_shape, src_rank=1, group=self.group_except_zero)
            
            if rank >= 2:
                x_ids = torch.zeros((x_shape[0].item(), x_shape[1].item()), device=device, dtype=torch.long)
                y = torch.zeros((y_shape[0].item(), y_shape[1].item()), device=device, dtype=torch.long)
                if self.verbose:
                    log_msg("Stage 0c", rank, f"Created receiving tensors - x: {x_ids.shape}, y: {y.shape}")
            
            broadcast_in_group(x_ids, src_rank=1, group=self.group_except_zero)
            broadcast_in_group(y, src_rank=1, group=self.group_except_zero)
            
            if self.verbose:
                if rank == 1:
                    log_msg("Stage 0c", rank, f"Broadcast completed - x: {x_ids.shape}, y: {y.shape}")
                elif rank >= 2:
                    log_msg("Stage 0c", rank, f"Received data - x: {x_ids.shape}, y: {y.shape}, x[0,0]: {x_ids[0,0]}")
        
        dist.barrier()
        if self.verbose:
            log_end("Stage 0c", rank)
    
        for meta_pert in range(self.meta_perturbations):
    
            if self.verbose:
                log_msg(f"Perturbation {meta_pert+1} of {self.meta_perturbations}", rank, "")
    
            # -------------------- Stage 3: Dirty Ranks Compute Their Dirty theta_t --------------------
            # Stage 3a: Each dirty rank samples its own probe = eps * N(0,1).
            if self.verbose:
                log_start("Stage 3a", rank)
            if rank >= 2:
                probe_list = []
                for p in self.param_list:
                    probe = torch.zeros_like(p)
                    probe.normal_(mean=0, std=1)
                    probe.mul_(self.epsilon)
                    probe_list.append(probe)
                self.probe_list = probe_list
            if self.verbose:
                log_end("Stage 3a", rank)
            dist.barrier()
    
            # Stage 3b: Dirty ranks receive eps * adam_ratio from Rank 0 and add it to probe.
            if self.verbose:
                log_start("Stage 3b", rank)
            
            if rank == 0:
                # Precompute scalars for m_hat and v_hat scaling
                scaling1 = 1.0 / (1 - self.beta1 ** (iteration + 1))
                scaling2 = 1.0 / (1 - self.beta2 ** (iteration + 1))
            
                for idx, p in enumerate(self.adam_m):
                    m_flat = p.view(-1)
                    v_flat = self.adam_v[idx].view(-1)
            
                    # We'll allocate a buffer to hold each chunk's ratio
                    ratio_buf = m_flat.new_empty(CHUNK_SIZE)
            
                    # Process in chunks
                    for start in range(0, m_flat.numel(), CHUNK_SIZE):
                        end = min(m_flat.numel(), start + CHUNK_SIZE)
                        chunk_size = end - start
            
                        # 1) Copy the relevant slice of m_flat
                        ratio_buf[:chunk_size].copy_(m_flat[start:end])
                        # 2) Scale in-place for m_hat
                        ratio_buf[:chunk_size].mul_(scaling1)
            
                        # 3) Compute v_hat chunk
                        temp_v = v_flat[start:end].clone()  # small chunk
                        temp_v.mul_(scaling2)
                        temp_v.sqrt_()
                        temp_v.add_(1e-8)
            
                        # 4) ratio = m_hat / v_hat
                        ratio_buf[:chunk_size].div_(temp_v)
            
                        # Broadcast the ratio chunk to other ranks
                        dist.broadcast(ratio_buf[:chunk_size], src=0, async_op=True)
            
            elif rank == 1:
                # Instead of allocating a full-size temp, allocate per-chunk (dropped afterwards)
                for idx, p in enumerate(self.adam_ratio_list):
                    p_flat = p.view(-1)
                    for start in range(0, p_flat.numel(), CHUNK_SIZE):
                        end = min(p_flat.numel(), start + CHUNK_SIZE)
                        temp_chunk = p_flat.new_zeros(end - start)
                        dist.broadcast(temp_chunk, src=0, async_op=True)
                        # The received data is intentionally dropped 
            
            elif rank >= 2:
                for idx, p in enumerate(self.probe_list):
                    p_flat = p.view(-1)
                    for start in range(0, p_flat.numel(), CHUNK_SIZE):
                        end = min(p_flat.numel(), start + CHUNK_SIZE)
                        temp_chunk = p_flat.new_zeros(end - start)
                        dist.broadcast(temp_chunk, src=0, async_op=True)
                        # Add the received chunk (scaled) into the probe
                        p_flat[start:end].add_(temp_chunk * self.epsilon)
            
            if self.verbose:
                log_end("Stage 3b", rank)
            # dist.barrier()

            
        
            # Stage 3c: Dirty ranks apply a random dropout mask to their probe.
            if self.verbose:
                log_start("Stage 3c", rank)
            if rank >= 2:
                for probe in self.probe_list:
                    if self.probe_dropout_rate > 0:
                        mask = (torch.rand_like(probe) > self.probe_dropout_rate).float()
                        probe.mul_(mask)
                        del mask
            # dist.barrier()
            if self.verbose:
                log_end("Stage 3c", rank)
        
            # Stage 3d: Dirty theta_t = clean theta_t + modified probe.
            if self.verbose:
                log_start("Stage 3d", rank)
            if rank >= 2:
                for param, probe in zip(self.param_list, probe_list):
                    param.add_(probe)
            # dist.barrier()
            if self.verbose:
                log_end("Stage 3d", rank)
        
            # -------------------- Stage 4: Forward Pass --------------------
            if self.verbose:
                log_start("Stage 4", rank)
            if rank == 1:
                loss = teacher_forcing_loss_emb_parallel(self.model, x_ids, y, criterion)
                if self.verbose:
                    log_msg("Stage 4", rank, f"Clean loss = {loss}")
            elif rank >= 2:
                loss = teacher_forcing_loss_emb_parallel(self.model, x_ids, y, criterion)
                if self.verbose:
                    log_msg("Stage 4", rank, f"Dirty loss = {loss}")
            else:
                loss = torch.tensor(0.0, device=device)
            dist.barrier()
            if self.verbose:
                log_end("Stage 4", rank)
        
            # -------------------- Stage 5: Gather Losses onto Rank 1 --------------------
            if self.verbose:
                log_start("Stage 5", rank)
            if rank == 1:
                gathered_losses = {1: loss}
                for r in range(2, world_size):
                    temp = torch.tensor(0.0, device=device)
                    dist.recv(temp, src=r)
                    gathered_losses[r] = temp
                if self.verbose:
                    log_msg("Stage 5", rank, f"Gathered losses: {gathered_losses}")
            elif rank >= 2:
                dist.send(loss, dst=1)
            dist.barrier()
            if self.verbose:
                log_end("Stage 5", rank)
        
            # -------------------- Stage 6: Compute grad_est per Dirty Rank and Scatter --------------------
            if self.verbose:
                log_start("Stage 6", rank)
            if rank == 1:
                clean_loss = gathered_losses[1]
                grad_est_dict = {}
                for r in range(2, world_size):
                    grad_est = (gathered_losses[r] - clean_loss) / (self.epsilon + 1e-8)
                    grad_est_dict[r] = grad_est/(self.epsilon + 1e-8)
                for r, ge in grad_est_dict.items():
                    dist.send(ge, dst=r)
                self.grad_est_dict = grad_est_dict
                if self.verbose:
                    log_msg("Stage 6", rank, f"Computed grad_est per dirty rank: {grad_est_dict}")
            elif rank >= 2:
                grad_est = torch.tensor(0.0, device=device)
                dist.recv(grad_est, src=1)
                self.grad_est = grad_est
                if self.verbose:
                    log_msg("Stage 6", rank, f"Received grad_est = {grad_est}")
            dist.barrier()
            if self.verbose:
                log_end("Stage 6", rank)
        
            # -------------------- Stage 8: On Dirty Ranks, Scale Their Probe with grad_est and take eps out of it --------------------
            if self.verbose:
                log_start("Stage 8", rank)
            if rank >= 2:
                for probe in self.probe_list:
                    probe.mul_(self.grad_est)
                if self.verbose:
                    log_msg("Stage 8", rank, "Scaled probe by grad_est in place.")
            dist.barrier()
            if self.verbose:
                log_end("Stage 8", rank)
        
            # -------------------- Stage 9: Reduce Scaled Probes from Dirty Ranks to Rank 1 --------------------
            if self.verbose:
                log_start("Stage 9", rank)
            if rank >= 2:
                for probe in self.probe_list:
                    reduce_mean_in_group(probe, dst_rank=1, group=self.group_except_zero, world_size=world_size)
            if rank == 1:
                for i, probe in enumerate(self.adam_ratio_list):
                    if meta_pert == 0:
                        probe.zero_()
                    reduced_probe = reduce_mean_in_group(probe.clone(), dst_rank=1, group=self.group_except_zero, world_size=world_size)
                    self.adam_ratio_list[i] = reduced_probe
                self.avg_probe_list = self.adam_ratio_list
                if self.verbose:
                    log_msg("Stage 9", rank, "Averaged scaled probes from dirty ranks.")
            dist.barrier()
            if self.verbose:
                log_end("Stage 9", rank)
                
            # -------------------- Stage 9b: Prep for another meta-perturbation --------------------
            if self.meta_perturbations - meta_pert - 1 > 0:
                if self.verbose:
                    log_start("Stage 9b", rank)
                if rank >= 2:
                    for param, probe in zip(self.param_list, probe_list):
                        probe.div_(grad_est + 1e-8)
                        param.sub_(probe)
                dist.barrier()
                if self.verbose:
                    log_end("Stage 9b", rank)
        
        # -------------------- Stage 10: Rank 1 Streams Averaged Probe to Rank 0; Rank 0 Updates Adam State --------------------
        if self.verbose:
            log_start("Stage 10a", rank)
        if rank == 1:
            for idx, p in enumerate(self.avg_probe_list):
                p_flat = p.view(-1)
                for start in range(0, p_flat.numel(), CHUNK_SIZE):
                    end = min(p_flat.numel(), start + CHUNK_SIZE)
                    # Clone the chunk to ensure a contiguous block is sent
                    temp_chunk = p_flat[start:end].clone()
                    dist.send(temp_chunk, dst=0)
            if self.verbose:
                log_msg("Stage 10a", rank, "Streamed averaged probe to Rank 0.")
        elif rank == 0:
            for idx, p in enumerate(self.adam_m):
                p_flat = p.view(-1)
                v_flat = self.adam_v[idx].view(-1)
                for start in range(0, p_flat.numel(), CHUNK_SIZE):
                    end = min(p_flat.numel(), start + CHUNK_SIZE)
                    temp_chunk = p_flat.new_zeros(end - start)
                    dist.recv(temp_chunk, src=1)
                    p_flat[start:end].mul_(self.beta1).add_(temp_chunk, alpha=1 - self.beta1)
                    v_flat[start:end].mul_(self.beta2).addcmul_(temp_chunk, temp_chunk, value=1 - self.beta2)
            if self.verbose:
                log_msg("Stage 10a", rank, "Updated adam_v and adam_m")
        

        # -------------------- Stage 10b: Stream adam_ratio from rank 0 to rank 1 --------------------
        if self.verbose:
            log_start("Stage 10b", rank)
        if rank == 0:
            # Precompute scaling factors in-place (they're scalars)
            scaling1 = 1.0 / (1 - self.beta1 ** (iteration + 1))
            scaling2 = 1.0 / (1 - self.beta2 ** (iteration + 1))
            for idx, p in enumerate(self.adam_m):
                # Flatten the parameter tensors for easier chunk processing.
                m_flat = p.view(-1)
                v_flat = self.adam_v[idx].view(-1)
                # Preallocate a buffer for the ratio chunk of size CHUNK_SIZE.
                ratio_buf = m_flat.new_empty(CHUNK_SIZE)
                # Process tensor in chunks.
                for start in range(0, m_flat.numel(), CHUNK_SIZE):
                    end = min(m_flat.numel(), start + CHUNK_SIZE)
                    chunk_size = end - start
                    # Copy the current m chunk into the buffer.
                    ratio_buf[:chunk_size].copy_(m_flat[start:end])
                    # In-place scale: m_hat_chunk = m_chunk * scaling1
                    ratio_buf[:chunk_size].mul_(scaling1)
                    # Process the corresponding v chunk:
                    # Compute v_hat_chunk = (v_chunk * scaling2).sqrt() + epsilon
                    # We allocate a temporary buffer for the v chunk; since it's small, this overhead is minimal.
                    temp = v_flat[start:end].clone()
                    temp.mul_(scaling2)
                    temp.sqrt_()
                    temp.add_(1e-8)
                    # In-place compute the ratio for the chunk: ratio = m_hat_chunk / v_hat_chunk
                    ratio_buf[:chunk_size].div_(temp)
                    # Send the computed chunk.
                    dist.send(ratio_buf[:chunk_size], dst=1)
            if self.verbose:
                log_msg("Stage 10b", rank, "Streamed adam_ratio chunks to Rank 1.")
        elif rank == 1:
            for idx, p in enumerate(self.adam_ratio_list):
                p_flat = p.view(-1)
                for start in range(0, p_flat.numel(), CHUNK_SIZE):
                    end = min(p_flat.numel(), start + CHUNK_SIZE)
                    chunk_size = end - start
                    # Allocate a small buffer for receiving the chunk.
                    temp_chunk = p_flat.new_empty(chunk_size)
                    dist.recv(temp_chunk, src=0)
                    # In-place copy into the correct segment.
                    p_flat[start:end].copy_(temp_chunk)
            if self.verbose:
                log_msg("Stage 10b", rank, "Reconstructed adam_ratio from received chunks.")
        dist.barrier()
        if self.verbose:
            log_end("Stage 10b", rank)


        # -------------------- Stage 11: On Rank 1, Update theta_t with New adam_ratio --------------------
        if self.verbose:
            log_start("Stage 11", rank)
        if rank == 1:
            list_inplace_sub(self.param_list, self.adam_ratio_list, alpha=current_lr)
            if self.verbose:
                log_msg("Stage 11", rank, f"Updated theta_t with learning rate {current_lr:.6f} in place.")
            log_msg(" ", rank, f"gathered_losses {gathered_losses}.")
            log_msg(" ", rank, f"grad_est_dict {grad_est_dict}.")
        dist.barrier()
        if self.verbose:
            log_end("Stage 11", rank)

        if rank == 1:
            return loss
        else:
            return None



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
    

    if vebose:
        print(f"   Max lengths: max={max_out_len}")
        
        min_in_len = min(len(i) for i in in_list)
        min_out_len = min(len(i) for i in out_list)
        max_in_len = max(len(i) for i in in_list)
        max_out_len = max(len(i) for i in out_list)
        
        print(f"Sequence length stats:")
        print(f"Input lengths: min={min_in_len}, max={max_in_len}")
        print(f"Output lengths: min={min_out_len}, max={max_out_len}")
        print(f"Min total lengths: {[len(i) + len(j) for i,j in zip(in_list, out_list)]}")

   
    return in_list, out_list




def save_distributed_checkpoint(optimizer, run_name, save_dir, rank):
    """
    Save checkpoint in a distributed manner:
    - Rank 0: Save adam_m and adam_v tensors
    - Rank 1: Save model state and metadata
    
    Args:
        optimizer: The optimizer that contains the model and adam parameters
        run_name: Name of the current run/experiment
        save_dir: Directory where to save the checkpoint
        rank: Current process rank
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Rank 0 saves adam_m and adam_v tensors
    if rank == 0:
        adam_m_path = os.path.join(save_dir, f"{run_name}_adam_m")
        adam_v_path = os.path.join(save_dir, f"{run_name}_adam_v")
        
        torch.save(optimizer.adam_m, adam_m_path)
        torch.save(optimizer.adam_v, adam_v_path)
        
        print(f"Rank {rank}: Adam moment tensors saved to {save_dir}")
    
    # Rank 1 saves model state
    elif rank == 1:
        model_path = os.path.join(save_dir, f"{run_name}_model.pt")
        
        # Save model checkpoint
        model_checkpoint = {
            'model_state_dict': optimizer.model.state_dict(),
            'model_args': {
                'input_size': optimizer.model.input_size,
                'output_size': optimizer.model.output_size,
                'hidden_size': optimizer.model.hidden_size,
                'memory_size': optimizer.model.memory_size,
                'head_size': optimizer.model.head_size,
                'num_reads': optimizer.model.num_reads,
                'embed': optimizer.model.embed
            }
        }
        
        torch.save(model_checkpoint, model_path)
        print(f"Rank {rank}: Model state saved to {model_path}")
    else: # all other ranks just pass
        pass 



def load_distributed_checkpoint(optimizer, run_name, save_dir, device, rank):
    """
    Load a distributed checkpoint:
    - Rank 0: Load adam_m and adam_v tensors
    - Rank 1: Load model state and metadata
    
    Args:
        optimizer: The optimizer that contains the model and adam parameters
        run_name: Name of the run/experiment to load
        save_dir: Directory where the checkpoint is saved
        device: Device to load the tensors and model on
        rank: Current process rank
    
    Returns:
        success: Boolean indicating if the loading was successful
    """
    # call it like this: success = load_distributed_checkpoint(optimizer, run_name, save_dir, device, rank)

    # Rank 0 loads adam_m and adam_v tensors
    if rank == 0:
        adam_m_path = os.path.join(save_dir, f"{run_name}_adam_m")
        adam_v_path = os.path.join(save_dir, f"{run_name}_adam_v")
        
        if os.path.exists(adam_m_path) and os.path.exists(adam_v_path):
            optimizer.adam_m = torch.load(adam_m_path, map_location=device)
            optimizer.adam_v = torch.load(adam_v_path, map_location=device)
            print(f"Rank {rank}: Adam moment tensors loaded from {save_dir}")
            return True
        else:
            print(f"Warning: Adam moment tensors not found at {save_dir}")
            return False
    
    # Rank 1 loads model state
    elif rank == 1:
        model_path = os.path.join(save_dir, f"{run_name}_model.pt")
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            
            # Load model state dictionary
            optimizer.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Verify model args if needed
            model_args = checkpoint['model_args']
            current_args = {
                'input_size': optimizer.model.input_size,
                'output_size': optimizer.model.output_size,
                'hidden_size': optimizer.model.hidden_size,
                'memory_size': optimizer.model.memory_size,
                'head_size': optimizer.model.head_size,
                'num_reads': optimizer.model.num_reads
            }
            
            # Optional check for architecture consistency
            for key, value in current_args.items():
                if key in model_args and model_args[key] != value:
                    print(f"Warning: Loaded model has different {key}: {model_args[key]} vs current {value}")
            
            print(f"Rank {rank}: Model state loaded from {model_path}")
            return True
        else:
            print(f"Warning: Model checkpoint not found at {model_path}")
            return False
    
    # For other ranks, just return True
    return True



# =============================================================================
# Main Routine
# =============================================================================
def main():
    import os
    os.environ["WANDB_API_KEY"] = ""

    # # train on real data 
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed training.")
    # parser.add_argument("--mode", type=str, choices=["test", "train"], default="train", help="Run mode: test or train.")
    # parser.add_argument("--max_iters", type=int, default=1e10, help="Maximum iterations for training.")
    # parser.add_argument("--schedule_patience", type=int, default=1, help="Maximum iterations for training.")
    # parser.add_argument("--learning_rate", type=float, default=0.0001, help="Base learning rate (and eps, tied 1:1).")
    # parser.add_argument("--beta1", type=float, default=0.999, help="Base learning rate (and eps, tied 1:1).")
    # parser.add_argument("--beta2", type=float, default=0.99999, help="Base learning rate (and eps, tied 1:1).")
    # parser.add_argument("--epsilon", type=float, default=0.001, help="Perturbation scale epsilon (tied to learning rate).")
    # parser.add_argument("--probe_dropout_rate", type=float, default=0.999, help="Dropout rate for probe vector.")
    # parser.add_argument("--wandb_proj", type=str, default="DNC-SINGLE-DISTRIBUTED", help="WandB project name (optional)")
    # parser.add_argument("--wandb_run", type=str, default="", help="WandB run name (optional)")
    # parser.add_argument("--warmup_iters", type=int, default=100, help="Warmup iterations.")
    # parser.add_argument("--cosine_wavelength", type=int, default=1000, help="Cosine LR wavelength, init to very high.")
    # parser.add_argument("--val_iters", type=int, default=1000, help="Val iters, when we run val and log to wandb, and potentially checkpoint.")
    # parser.add_argument("--meta_perturbations", type=int, default=1, help="Number of Perturbations for all ranks per step.")
    # # parser.add_argument("--scatter_batch", type=str, choices=["true", "false"], default="false", help="whether each perturbation should be on a different batch, if true, we sample (world_size-2)*batch_size x_ids and y per iter and scatter it to .")
    
    # parser.add_argument(
    #     "--precision", 
    #     type=str, 
    #     choices=["fp32", "fp16", "bf16", "fp8", "int8"], 
    #     default="fp32", 
    #     help="Set precision mode: fp32, fp16, bf16, fp8, int8."
    # )
    
    # # New CLI arguments for model configuration
    # parser.add_argument("--model_scale", type=int, default=1, help="Scaling factor for model dimensions.")
    # parser.add_argument("--vocab_size", type=int, default=150, help="Vocabulary size.")
    # parser.add_argument("--num_heads", type=int, default=1, help="# dnc heads.")
    # parser.add_argument("--memory_size", type=int, default=2900, help="memory_size.")
    # parser.add_argument("--hidden_size", type=int, default=2900, help="hidden_size.")
    # parser.add_argument("--input_size", type=int, default=2900, help="Input size.")
    # parser.add_argument("--head_size", type=int, default=2900, help="head_size .")
    # parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    # parser.add_argument("--seq_len", type=int, default=1000, help="Sequence length.")
    # args = parser.parse_args()






    

    # # TEST OVERFIT FAST DEMO! 
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed training.")
    parser.add_argument("--mode", type=str, choices=["test", "train"], default="test", help="Run mode: test or train.")
    parser.add_argument("--max_iters", type=int, default=1e10, help="Maximum iterations for training.")
    parser.add_argument("--schedule_patience", type=int, default=1, help="Maximum iterations for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Base learning rate (and eps, tied 1:1).")
    parser.add_argument("--beta1", type=float, default=0.9, help="Base learning rate (and eps, tied 1:1).")
    parser.add_argument("--beta2", type=float, default=0.9, help="Base learning rate (and eps, tied 1:1).")
    parser.add_argument("--epsilon", type=float, default=0.001, help="Perturbation scale epsilon (tied to learning rate).")
    parser.add_argument("--probe_dropout_rate", type=float, default=0.99, help="Dropout rate for probe vector.")
    parser.add_argument("--wandb_proj", type=str, default="ZeroProp_Final_Runs", help="WandB project name (optional)")
    parser.add_argument("--wandb_run", type=str, default="", help="WandB run name (optional)")
    parser.add_argument("--warmup_iters", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--cosine_wavelength", type=int, default=100000000, help="Cosine LR wavelength, init to very high.")
    parser.add_argument("--val_iters", type=int, default=10, help="Val iters, when we run val and log to wandb, and potentially checkpoint.")
    parser.add_argument("--meta_perturbations", type=int, default=1, help="Number of Perturbations for all ranks per step.")
    
    
    # parser.add_argument(
    #     "--precision", 
    #     type=str, 
    #     choices=["fp32", "fp16", "bf16", "fp8", "int8"], 
    #     default="fp32", 
    #     help="Set precision mode: fp32, fp16, bf16, fp8, int8."
    # )

    # model_size = 100 == 1,603,290,101 and takes 35GB
    # model_size = 10 == 16,0329,010 and takes 35GB
    
    # New CLI arguments for model configuration
    parser.add_argument("--model_scale", type=int, default=1, help="Scaling factor for model dimensions.")
    parser.add_argument("--vocab_size", type=int, default=150, help="Vocabulary size.")
    parser.add_argument("--num_heads", type=int, default=1, help="# dnc heads.")
    parser.add_argument("--memory_size", type=int, default=2900, help="memory_size.")
    parser.add_argument("--hidden_size", type=int, default=2900, help="hidden_size.")
    parser.add_argument("--input_size", type=int, default=2900, help="Input size.")
    parser.add_argument("--head_size", type=int, default=2900, help="head_size .")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--seq_len", type=int, default=1000, help="Sequence length.")


    args = parser.parse_args()

    #####################################################################################
    # SETUP TRAINING
    #####################################################################################
    run_name = f"{args.mode}_lr{args.learning_rate}_scale{args.model_scale}_pdrop{args.probe_dropout_rate}"
    
    # Add model architecture details
    model_params = f"_h{args.hidden_size}"
    
    # Add training configuration
    train_params = f"_bs{args.batch_size}_seq{args.seq_len}_b1_{args.beta1}_b2_{args.beta2}"
    
    # Add optimization details
    opt_params = f"_coswav_{args.cosine_wavelength}_wu{args.warmup_iters}_mp{args.meta_perturbations}_pat{args.schedule_patience}"
    
    # Combine all parts
    if args.wandb_run=="":
         args.wandb_run = run_name + model_params + train_params + opt_params
    
    vocab_list, char_to_id, id_to_char = get_char_vocab()
    vocab_size = len(vocab_list)
    
    args.vocab_size = vocab_size

    # Derived values based on model_scale
    args.hidden_size = args.hidden_size * args.model_scale
    args.memory_size = args.memory_size * args.model_scale
    args.head_size = args.head_size * args.model_scale
    args.input_size = args.input_size * args.model_scale

    # TEMP OVERRIDE FOR NOW SO WE CAN DEBUG
    # args.wandb_proj = None
    verbose = False
    
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{args.local_rank}")

    assert torch.cuda.is_available(), f"Rank {rank}: CUDA not available!"
    print(f"Rank {rank} using device {torch.cuda.current_device()}")

    # set the random seed differently per rank
    torch.manual_seed(torch.initial_seed() + rank) 

    log_msg("Trying first dist.barrier(), if hanging here, no infiniband likely on node, need to turn off p2p",rank,"if so, run export NCCL_P2P_DISABLE=1")
    dist.barrier()
    

    log_start("INIT MODEL", rank)

    with torch.inference_mode():
        # time.sleep(rank) # we stagger the launch of DNC formation prevent RAM issues
        embed = nn.Embedding(args.vocab_size, args.input_size,device=device)
        model = DNC(input_size=args.input_size, output_size=args.vocab_size, hidden_size=args.hidden_size, memory_size=args.memory_size, head_size=args.head_size, num_heads=args.num_heads, embed=embed, device=device)
        # model.controller.flatten_parameters()
        model.eval()
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    if rank != 0:

        x_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device, dtype=torch.long) # PLACEHOLDER
        y = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device) # PLACEHOLDER

        
        if rank == 1:
            num_params = sum(p.numel() for p in model.parameters())
            num_layers = len(list(model.children()))
            print(f"[Init] Model has {num_params} parameters across {num_layers} layers.")
            

    

    elif rank == 0:
        x_ids, y = None,  None
        pass
        
    dist.barrier()
    log_end("INIT MODEL", rank)
    
    distributed_adaptive = MezoAdaptiveSamplingParallel(
        model=model,
        learning_rate=args.learning_rate,
        probe_dropout_rate=args.probe_dropout_rate,
        epsilon=args.epsilon,
        beta1=args.beta1,
        beta2=args.beta2,
        meta_perturbations=args.meta_perturbations,
        verbose=verbose
    )
    dist.barrier()
    
    if rank == 1:
        # Loss EMA tracking - one fast, one slow
        loss_ema_fast = None
        loss_ema_slow = None
        ema_alpha_fast = 0.9  # Faster EMA
        ema_alpha_slow = 0.999  # Slower EMA
        
        # Cosine learning rate scheduling parameters
        base_lr = args.learning_rate
        min_lr = base_lr * 0.001
        cosine_wavelength = args.cosine_wavelength #1000  # Length of each cosine cycle
        schedule_iteration = 0
        patience_counter = 0
        
        # Track previous loss
        prev_loss = float('inf')
        
    
    if rank == 1 and args.wandb_proj is not None and wandb is not None:
        # wandb.init(project=args.wandb_proj, name=args.wandb_run)
        wandb.init(project=args.wandb_proj,name=args.wandb_run)
        wandb.config.update( vars(args) )

        print("[Rank 1] Initialized wandb logging.")

    #####################################################################################
    # Load the dataset
    #####################################################################################
    if rank == 1:
        # generate OWT 
        ds = None
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
                
    #####################################################################################
    # START TRAINING
    #####################################################################################
    val_loss = 1e4 # placeholder value
    start_time = datetime.datetime.now()
    with torch.inference_mode():
        
        for i in range(int(args.max_iters) ):
            #####################################################################################
            # Sample x_ids,y from dataset
            #####################################################################################
            if rank == 1 and args.mode == "train":

                # GENERATE TRAIN BATCH
                x_strs, y_strs = generate_openwebtext_task_str(
                                            args.batch_size,
                                            1,
                                            ds,
                                            train = True,
                                            min_total_seq_len=args.seq_len,
                                            vebose = False
                                        )  
                
                x_ids = str_to_tensor(x_strs, char_to_id).to(device)
                y = str_to_tensor(y_strs, char_to_id).to(device)

                if verbose:
                    print(f" x_ids {x_ids.shape} y {y.shape}" )
                    
            dist.barrier()
                    
            #####################################################################################
            # TRAIN THE MODEL
            #####################################################################################
            train_loss = distributed_adaptive.distributed_step(x_ids, y, criterion, iteration=i, warmup_iters=args.warmup_iters)


            #####################################################################################
            # CHECKPOINT THE MODEL RARELY
            #####################################################################################
            if args.mode == "train" and (i+1) % (args.val_iters*1000) == 0:
                
                save_distributed_checkpoint(distributed_adaptive, 
                                            args.wandb_run, 
                                            "rnn_checkpoints", 
                                            rank)
                dist.barrier()
            
                
            
            #####################################################################################
            # UPDATE THE LEARNING RATE WITH OUR FAST/SLOW EMA COSINE LR SCHEDULE
            #####################################################################################
            if rank == 1:
                if loss_ema_fast is None:
                    # Initialize both EMAs with the current loss value
                    loss_ema_fast = train_loss
                    loss_ema_slow = train_loss

                
                loss_ema_fast = ema_alpha_fast * loss_ema_fast + (1 - ema_alpha_fast) * train_loss
                loss_ema_slow = ema_alpha_slow * loss_ema_slow + (1 - ema_alpha_slow) * train_loss
                
                if i < args.warmup_iters:
                    # Linear warmup
                    distributed_adaptive.learning_rate = base_lr * (i / args.warmup_iters)
                else:
                    
                    # Check if the fast EMA is higher than the slow EMA (by a small threshold)
                    # if loss_ema_fast > (loss_ema_slow + 1e-5):
                    #     patience_counter += 1
                    # else:
                    #     patience_counter = 0  # reset if condition is not met
                
                    # # Only step the cosine schedule if we have been patient enough
                    # if patience_counter >= args.schedule_patience:
                    #     schedule_iteration += 1
                    #     patience_counter = 0  # reset the counter after stepping
                
                    # Compute the position within the cosine cycle.
                    # Here, schedule_iteration is used to determine how far we are along the cycle.
                    # cycle_iteration = schedule_iteration % cosine_wavelength
                    
                    # progress = schedule_iteration / cosine_wavelength
                    
                    progress = i / cosine_wavelength
                    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                    distributed_adaptive.learning_rate = min_lr + (base_lr - min_lr) * cosine_factor
                
                            
                

                #####################################################################################
                # RUN VALIDATION
                #####################################################################################
                if rank == 1 and (i+1) % args.val_iters == 0:

                    if args.mode == "train":
        
                        # GENERATE VAL BATCH and run
                        x_strs, y_strs = generate_openwebtext_task_str(
                                                    args.batch_size,
                                                    1,
                                                    ds,
                                                    train = False,
                                                    min_total_seq_len=args.seq_len,
                                                    vebose = False
                                                )  
                        
                        x_ids = str_to_tensor(x_strs, char_to_id).to(device)
                        y = str_to_tensor(y_strs, char_to_id).to(device)
                        val_loss = teacher_forcing_loss_emb_parallel(model, x_ids, y, criterion)
                    
                    #####################################################################################
                    # log to wandb every val_iters iterations.
                    #####################################################################################
                    if args.wandb_proj is not None and wandb is not None:
                        # Compute a dummy weight decay loss (if applicable)
                        weight_decay_loss = 0.0
                        for param in model.parameters():
                            if param.requires_grad:
                                weight_decay_loss += (1e-2 / 2) * torch.sum(param ** 2)  # using 1e-2 as dummy weight_decay
    
    
                        
                        log_data = {
                            "train_loss": train_loss, 
                            "val_loss": val_loss, 
                            "loss_ema_fast":loss_ema_fast,
                            "loss_ema_slow":loss_ema_slow,
                            "lr": distributed_adaptive.learning_rate,
                            "weight_decay_loss": weight_decay_loss.item(),
                        }
                        
                        try:
                            wandb.log(log_data, step=i)
                        except Exception as e:
                            print(e)
                    
                
                #####################################################################################
                # Log to stdout
                #####################################################################################
                print("="*50)
                print(f"[Train] Iteration {i}, train_loss = {train_loss}, loss_ema_fast = {loss_ema_fast}, loss_ema_slow = {loss_ema_slow}, lr = {distributed_adaptive.learning_rate}, val_loss = {val_loss}")
                
               

            dist.barrier()
            if rank==1:
                if train_loss < 0.1:

                    end_time = datetime.datetime.now()
                    print("="*50)
                    print("="*50)
                    print("="*50)
                    log_msg("FINISHED TRAINING", rank, f"in {i} iterations acheived {train_loss} loss in {end_time - start_time} seconds.")
                    print(f"[Init] Model has {num_params} parameters across {num_layers} layers.")
            
                    print("="*50)
                    print("="*50)
                    print("="*50)
                    time.sleep(200)
                    break

                    
                
            
        dist.destroy_process_group()
        

if __name__ == "__main__":
    main()
