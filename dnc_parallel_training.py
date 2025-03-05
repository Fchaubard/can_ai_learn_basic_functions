import os
import sys
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
from copy import deepcopy
import numpy as np


# Import helpers from original script
from ntm_with_modern_training_runs import (
    get_char_vocab, str_to_tensor, teacher_forcing_loss_emb,
    generate_task_data, generate_sequence_batched, NTM, Transformer, Mamba,
    SimpleLSTM,
    pick_gpu_with_most_free_mem, calculate_vram_usage_direct,
    maybe_update_curriculum, prepare_model_for_fast_inference, WarmupScheduler
)



############
# DNC
##############
class DNC(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, memory_size, head_size, num_heads, embed):
        super(DNC, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
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

        # Output projection with normalization
        total_output_size = hidden_size + self.num_reads * self.head_size
        self.pre_output_norm = nn.LayerNorm(total_output_size)
        self.fc_proj = nn.Linear(total_output_size, output_size)

        self.reset_parameters()
        self.read_vec_buffer = None

    def reset_parameters(self):
        # Initialize LSTM params
        for name, p in self.controller.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0)
    
        # Initialize memory operation layers
        for name, p in self.named_parameters():
            if 'fc_' in name and 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif 'fc_' in name and 'bias' in name:
                nn.init.constant_(p, 0)

    def _read_memory(self, memory, read_keys):
        # Enable faster operations
        torch.backends.cudnn.benchmark = True
        
        # Normalize memory and keys
        memory_normalized = self.memory_norm(memory)
        read_keys_reshaped = read_keys.view(-1, self.head_size)
        read_keys_norm = self.read_keys_norm(read_keys_reshaped)
        read_keys = read_keys_norm.view(-1, self.num_reads, self.head_size)

        # Optimized attention calculation
        attn_scores = torch.bmm(read_keys, memory_normalized.transpose(1, 2))
        read_weights = torch.softmax(attn_scores, dim=2)
        read_vectors = torch.bmm(read_weights, memory)
        
        return read_vectors

    def _write_memory(self, memory, write_keys, write_str, erase_vec, write_vec):
        # Enable faster operations
        torch.backends.cudnn.benchmark = True
        
        # Normalize memory and keys
        memory_normalized = self.memory_norm(memory)
        write_keys = self.write_keys_norm(write_keys)
        
        # Calculate attention dotproduct 
        # [B, 1, H] x [B, M, H].transpose(1,2) -> [B, 1, M]
        attn_scores = torch.bmm(write_keys.unsqueeze(1), memory_normalized.transpose(1, 2))
        write_weights = torch.softmax(attn_scores, dim=2)  # [B, 1, M]
        
        # Scale by write strength [B, 1] -> [B, 1, 1]
        write_weights = write_weights * write_str.unsqueeze(2)  # [B, 1, M]
        
        # Compute erase and write operations using einsum for proper broadcasting
        erase = torch.einsum('bij,bk->bik', write_weights, erase_vec)  # [B, M, H]
        write = torch.einsum('bij,bk->bik', write_weights, write_vec)  # [B, M, H]
        
        # Update memory (in place to save memory)
        memory.mul_(1.0 - erase).add_(write)
        return memory

    

    def forward(self, x_emb, hidden=None, memory=None):
        torch.backends.cudnn.benchmark = True  # Enable faster operations
        
        B, L, E = x_emb.size()
        device = x_emb.device

        # Initialize states if needed
        if hidden is None:
            h0 = torch.zeros(1, B, self.hidden_size, device=device)
            c0 = torch.zeros(1, B, self.hidden_size, device=device)
            hidden = (h0, c0)

        if memory is None:
            memory = torch.zeros(B, self.memory_size, self.head_size, device=device)

        # Pre-allocate buffer for read vectors
        if self.read_vec_buffer is None or self.read_vec_buffer.size(0) != B:
            self.read_vec_buffer = torch.zeros(B, self.num_reads * self.head_size, device=device)
        read_vec = self.read_vec_buffer
        
        # Pre-allocate output buffer
        outputs_list = []

        # Process sequence elements
        for t in range(L):
            # Combine input with read vector
            controller_input = torch.cat([x_emb[:, t, :], read_vec], dim=-1)
            controller_input = self.input_norm(controller_input)
            
            # Controller forward pass
            out_ctrl, hidden = self.controller(controller_input.unsqueeze(1), hidden)
            h = self.controller_norm(out_ctrl.squeeze(1))

            # Generate memory operation parameters
            read_keys = self.fc_read_keys(h).view(B, self.num_reads, self.head_size)
            write_keys = self.fc_write_keys(h)
            write_str = torch.sigmoid(self.fc_write_strength(h))
            erase_vec = torch.sigmoid(self.fc_erase_vector(h))
            write_vec = torch.tanh(self.fc_add_vector(h))

            # Memory operations
            memory = self._write_memory(memory, write_keys, write_str, erase_vec, write_vec)
            read_vectors = self._read_memory(memory, read_keys)
            read_vec = read_vectors.reshape(B, -1)

            # Output projection
            output = torch.cat([h, read_vec], dim=-1)
            output = self.pre_output_norm(output)
            logits = self.fc_proj(output)
            outputs_list.append(logits.unsqueeze(1))

        # Combine all outputs
        outputs = torch.cat(outputs_list, dim=1)
        return outputs, memory, hidden


# Helper functions for parameter manipulation
def unflatten_into_params(param_data, model, meta):
    """
    Unflatten 'param_data' (1D) back into each model parameter's .data
    using the (shape, numel, device) info in 'meta'.
    """
    offset = 0
    for p, (shape, numel, dev) in zip(model.parameters(), meta):
        slice_ = param_data[offset : offset + numel]
        offset += numel
        p.data.copy_(slice_.view(shape))
        
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

def flatten_adam_ratio_data(model, optimizer):
    """
    For each parameter p, fetch the Adam state:
        exp_avg  (m_t)
        exp_avg_sq (v_t)
    and compute ratio = (exp_avg / sqrt(exp_avg_sq + 1e-8)).

    If the state doesn't exist (or mismatch shape),
    fallback to zeros as desired.

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

def teacher_forcing_loss_emb_parallel(model, x_ids, y_ids_unpadded, criterion, chunk_size=32, x_emb=None):
    """
    Returns a 0D GPU tensor for the loss instead of a Python float.
    This prevents implicit GPU->CPU sync during distributed runs.
    """
    torch.backends.cudnn.benchmark = True

    if x_emb is None:
        x_emb = model.embed(x_ids)
        
    B, Lx, E = x_emb.shape
    Ly = y_ids_unpadded.shape[1]

    # If it's a Transformer or Mamba, the logic is simpler (check for specific model types)
    if hasattr(model, '__class__') and model.__class__.__name__ in ('Transformer', 'Mamba'):
        y_emb_input = model.embed(y_ids_unpadded)
        full_input_emb = torch.cat([x_emb, y_emb_input], dim=1)[:, :-1]
        full_input_ids = torch.cat([x_ids, y_ids_unpadded], dim=1)[:, 1:]
        full_input_emb = full_input_emb.contiguous()
        full_input_ids = full_input_ids.contiguous()

        outputs, _, _ = model(full_input_emb)
        logits = outputs.contiguous()
        logits_2d = logits.reshape(-1, logits.size(-1))
        gold = full_input_ids.reshape(-1)

        # Adjust shapes if mismatch
        min_size = min(gold.size(0), logits_2d.size(0))
        gold = gold[:min_size]
        logits_2d = logits_2d[:min_size, :]

        avg_loss = criterion(logits_2d, gold)
        return avg_loss  # 0D tensor on GPU

    # Otherwise, handle RNN-based approach:
    else:
        hidden = None
        memory = None
        total_loss = torch.zeros((), device=x_emb.device)  # 0D GPU tensor
        total_predicted_tokens = 0

        # Process input sequence in bigger chunks
        chunk_size = max(chunk_size, 1024)
        for pos in range(0, Lx, chunk_size):
            chunk_end = min(pos + chunk_size, Lx)
            input_chunk = x_emb[:, pos:chunk_end, :]

            out_chunk, mem_new, hidden_new = model(input_chunk, hidden=hidden, memory=memory)
            hidden = hidden_new
            memory = mem_new

        # Process target sequence
        for pos in range(0, Ly - 1, chunk_size):
            chunk_end = min(pos + chunk_size, Ly - 1)
            y_chunk = y_ids_unpadded[:, pos:chunk_end]
            y_emb_chunk = model.embed(y_chunk)

            out_chunk, mem_new, hidden_new = model(y_emb_chunk, hidden=hidden, memory=memory)
            hidden = hidden_new
            memory = mem_new

            out_chunk = out_chunk.reshape(-1, out_chunk.size(-1))
            targets = y_ids_unpadded[:, pos+1:chunk_end+1].reshape(-1)

            if targets.size(0) > 0:
                chunk_loss = criterion(out_chunk, targets)
                # Accumulate in GPU tensor
                total_loss += chunk_loss * targets.size(0)
                total_predicted_tokens += targets.size(0)

        if total_predicted_tokens > 0:
            avg_loss = total_loss / total_predicted_tokens
        else:
            avg_loss = torch.zeros((), device=x_emb.device)

        return avg_loss  # 0D tensor on GPU







class MezoDistributedSampling:
    # Update the __init__ method in MezoDistributedSampling class
    def __init__(
        self,
        model,
        num_perturbations=1,
        probe_dropout_rate=0.0,
        verbose=True
    ):
        # Initialize distributed environment if not already done
        if not dist.is_initialized():
            self._init_distributed()
        
        # Get rank and world size information
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Check if we have enough GPUs for the requested perturbations
        if num_perturbations + 1 > self.world_size:
            raise ValueError(f"Not enough GPUs. Requested {num_perturbations+1} models, but only {self.world_size} GPUs available.")
        
        # Set up parameters
        self.num_perturbations = num_perturbations
        self.probe_dropout_rate = probe_dropout_rate
        self.verbose = verbose  # Keep verbose for all ranks
        
        # Setup local model for each rank
        self.device = f'cuda:{self.rank}'
        
        # First share model dimensions from rank 0
        if self.rank == 0:
            # Store main model on rank 0 only (no duplication)
            self.model_main = model
            self.local_model = model  # Use the same object
            
            # Extract model dimensions
            input_size = model.input_size
            output_size = model.output_size
            hidden_size = model.hidden_size
            memory_size = model.memory_size
            head_size = model.head_size
            num_heads = model.num_reads
            
            # Pack dimensions into a tensor
            dims = torch.tensor([input_size, output_size, hidden_size, memory_size, head_size, num_heads], 
                               dtype=torch.long, device=self.device)
            
            # Get meta info for parameter reshaping
            _, self.meta = flatten_params(model)
            
            print(f"Created model on rank {self.rank}\n")
        else:
            # Initialize placeholder tensor for receiving dimensions
            dims = torch.zeros(6, dtype=torch.long, device=self.device)
            self.meta = None
            self.model_main = None
        
        # Broadcast dimensions to all ranks
        dist.broadcast(dims, 0)
        
        # Create properly-sized models on non-rank-0 processes
        if self.rank != 0:
            # Unpack dimensions
            input_size = dims[0].item()
            output_size = dims[1].item()
            hidden_size = dims[2].item()
            memory_size = dims[3].item()
            head_size = dims[4].item()
            num_heads = dims[5].item()
            
            # Create embedding with correct dimensions
            embed = nn.Embedding(output_size, input_size).to(self.device)
            nn.init.orthogonal_(embed.weight)
            
            # Create DNC model with matching dimensions
            self.local_model = DNC(
                input_size=input_size,
                output_size=output_size,
                hidden_size=hidden_size,
                memory_size=memory_size,
                head_size=head_size,
                num_heads=num_heads,
                embed=embed
            ).to(self.device)
            self.local_model.eval()
            
            if self.verbose:
                print(f"Created placeholder DNC model on rank {self.rank} with matching dimensions\n")
        
        # Broadcast meta information from rank 0 to all ranks
        self._broadcast_meta()
        
        if self.verbose:
            num_params = sum(p.numel() for p in self.local_model.parameters())
            print(f"[Init] Rank {self.rank}: Model has {num_params} parameters.\n")
            print(f"[Init] Rank {self.rank}: Using {self.world_size} GPUs for distributed training.\n")
            print(f"[Init] Rank {self.rank}: Number of perturbations: {self.num_perturbations}\n")
            
    
    def _init_distributed(self):
        """Initialize the distributed environment."""
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Initialize process group
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
        
        # Set device based on local rank
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    
    def _broadcast_meta(self):
        """
        Broadcast meta information from rank 0 to all ranks.
        Meta information contains shapes and sizes of model parameters.
        """
        # First, broadcast the size of meta data
        if self.rank == 0:
            meta_str = str([(list(shape), numel) for shape, numel, _ in self.meta])
            meta_bytes = meta_str.encode('utf-8')
            size_tensor = torch.tensor([len(meta_bytes)], dtype=torch.long, device=self.device)
        else:
            size_tensor = torch.tensor([0], dtype=torch.long, device=self.device)
        
        dist.broadcast(size_tensor, 0)
        
        # Now, broadcast the meta data itself
        if self.rank == 0:
            meta_tensor = torch.ByteTensor(list(meta_bytes)).to(self.device)
        else:
            meta_tensor = torch.ByteTensor(size_tensor.item()).to(self.device)
        
        dist.broadcast(meta_tensor, 0)
        
        # Deserialize meta data on non-root ranks
        if self.rank != 0:
            meta_bytes = bytes(meta_tensor.cpu().numpy().tolist())
            meta_str = meta_bytes.decode('utf-8')
            meta_list = eval(meta_str)
            
            # Convert to meta format
            self.meta = [(tuple(shape), numel, self.device) for shape, numel in meta_list]




    def _broadcast_data_optimized(self, theta_cpu, theta_plus_adam_cpu, adam_ratio_cpu, seeds, x_ids, y):
        """
        Broadcast data from rank 0 to all other ranks with memory optimization.
        This includes theta, theta+adam_ratio, seeds, and dataset.
         
        Args:
            theta_cpu: Model parameters (valid on rank 0)
            theta_plus_adam_cpu: Theta + Adam ratio (valid on rank 0)
            adam_ratio_cpu: Adam ratio (valid on rank 0)
            seeds: Random seeds (valid on rank 0)
            x_ids: Input token IDs (valid on rank 0)
            y: Target token IDs (valid on rank 0)
            
        Returns:
            Tuple of broadcasted data on all ranks
        """
        # Create and broadcast size information
        if self.rank == 0:
            param_size = torch.tensor(theta_cpu.numel(), dtype=torch.long, device=self.device)
            seed_size = torch.tensor(len(seeds), dtype=torch.long, device=self.device)
            x_size_0 = torch.tensor(x_ids.size(0), dtype=torch.long, device=self.device)
            x_size_1 = torch.tensor(x_ids.size(1), dtype=torch.long, device=self.device)
            y_size_0 = torch.tensor(y.size(0), dtype=torch.long, device=self.device)
            y_size_1 = torch.tensor(y.size(1), dtype=torch.long, device=self.device)
        else:
            param_size = torch.tensor(0, dtype=torch.long, device=self.device)
            seed_size = torch.tensor(0, dtype=torch.long, device=self.device)
            x_size_0 = torch.tensor(0, dtype=torch.long, device=self.device)
            x_size_1 = torch.tensor(0, dtype=torch.long, device=self.device)
            y_size_0 = torch.tensor(0, dtype=torch.long, device=self.device)
            y_size_1 = torch.tensor(0, dtype=torch.long, device=self.device)
        
        # Broadcast sizes
        dist.broadcast(param_size, 0)
        dist.broadcast(seed_size, 0)
        dist.broadcast(x_size_0, 0)
        dist.broadcast(x_size_1, 0)
        dist.broadcast(y_size_0, 0)
        dist.broadcast(y_size_1, 0)
        
        # Allocate tensors for receiving data
        if self.rank != 0:
            theta_cpu = torch.zeros(param_size.item(), device='cpu')
            theta_plus_adam_cpu = torch.zeros(param_size.item(), device='cpu')
            adam_ratio_cpu = torch.zeros(param_size.item(), device='cpu')
            seed_tensor = torch.zeros(seed_size.item(), dtype=torch.long, device=self.device)
            x_ids = torch.zeros((x_size_0.item(), x_size_1.item()), dtype=torch.long, device=self.device)
            y = torch.zeros((y_size_0.item(), y_size_1.item()), dtype=torch.long, device=self.device)
        else:
            seed_tensor = torch.tensor(seeds, dtype=torch.long, device=self.device)
        
        # Move to device for broadcasting
        if self.rank == 0:
            theta_gpu = theta_cpu.to(self.device)
            theta_plus_adam_gpu = theta_plus_adam_cpu.to(self.device)
            adam_ratio_gpu = adam_ratio_cpu.to(self.device)
        else:
            theta_gpu = torch.zeros(param_size.item(), device=self.device)
            theta_plus_adam_gpu = torch.zeros(param_size.item(), device=self.device)
            adam_ratio_gpu = torch.zeros(param_size.item(), device=self.device)
        
        # Broadcast data (one at a time to save GPU memory)
        dist.broadcast(theta_gpu, 0)
        dist.broadcast(theta_plus_adam_gpu, 0)
        dist.broadcast(adam_ratio_gpu, 0)
        dist.broadcast(seed_tensor, 0)
        dist.broadcast(x_ids, 0)
        dist.broadcast(y, 0)
        
        # Move back to CPU for memory efficiency
        if self.rank != 0:
            theta_cpu = theta_gpu.cpu()
            theta_plus_adam_cpu = theta_plus_adam_gpu.cpu()
            adam_ratio_cpu = adam_ratio_gpu.cpu()
            seeds = seed_tensor.tolist()
        
        # Clean up GPU tensors
        theta_gpu = None
        theta_plus_adam_gpu = None
        adam_ratio_gpu = None
        torch.cuda.empty_cache()
        
        return theta_cpu, theta_plus_adam_cpu, adam_ratio_cpu, seeds, x_ids, y
    
    def mezo_distributed_sampling(
        self,
        x_ids,
        y,
        criterion,
        optimizer,
        epsilon=1e-3
    ):
        """
        Distributed implementation of Mezo Adaptive Sampling.
        Implements the "one_way" version where rank 0 does clean forward pass and other ranks do perturbed passes.
        """
        with torch.inference_mode():
            # Synchronize all processes at the start
            dist.barrier()
            start_time = time.time()
            
            # -------------------------------------------------------------
            # Step 1-4: Create batch data, distribute model and adam ratio
            # -------------------------------------------------------------
            if self.rank == 0:
                # Get original parameters
                orig_param_data, _ = flatten_params(self.model_main)
                
                # Get adam ratio on CPU
                ratio_data = flatten_adam_ratio_data(self.model_main, optimizer).cpu()
                
                # Move parameters to CPU for broadcasting
                orig_param_data_cpu = orig_param_data.cpu()
                
                # Generate random seeds for each rank
                seeds = [random.randint(0, 2**32-1) for _ in range(self.world_size)]
                
                # Zero gradients in the main model
                for p in self.model_main.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
                
                # Clean up original tensor
                orig_param_data = None
                torch.cuda.empty_cache()
            else:
                # These will be filled by broadcast
                orig_param_data_cpu = None
                ratio_data = None
                seeds = None
            
            # Broadcast data to all ranks
            orig_param_data_cpu, ratio_data, seeds, x_ids, y = self._broadcast_data(
                orig_param_data_cpu, ratio_data, seeds, x_ids, y
            )
            
            # Set random seed for this rank
            torch.manual_seed(seeds[self.rank])
            np.random.seed(seeds[self.rank])
            random.seed(seeds[self.rank])
            
            # -------------------------------------------------------------
            # Step 5-6: Parameter perturbation
            # -------------------------------------------------------------
            
            # Move original parameters to device
            orig_param_data = orig_param_data_cpu.to(self.device)
            
            # Apply parameters based on rank
            if self.rank == 0:
                # Rank 0 uses original params (no perturbation)
                unflatten_into_params(orig_param_data, self.local_model, self.meta)
                probe = None
            elif 1 <= self.rank <= self.num_perturbations:
                # Create random probe
                probe = torch.randn_like(orig_param_data)
                
                # Apply dropout if needed
                if self.probe_dropout_rate > 0.0:
                    mask = (torch.rand_like(probe) > self.probe_dropout_rate).float()
                    probe *= mask
                
                # Scale probe by epsilon
                probe *= epsilon
                
                # Apply perturbed parameters to model
                param_data = orig_param_data + probe
                unflatten_into_params(param_data, self.local_model, self.meta)
            else:
                # Ranks beyond num_perturbations don't participate
                unflatten_into_params(orig_param_data, self.local_model, self.meta)
                probe = None
            
            # Clean up tensors
            orig_param_data = None
            torch.cuda.empty_cache()
            
            # Synchronize before forward pass
            dist.barrier()
            
            # -------------------------------------------------------------
            # Step 7: Forward passes
            # -------------------------------------------------------------
            
            # Record start time for forward pass
            fwd_start_time = time.time()
            if self.verbose:
                print(f"[Rank {self.rank}] Starting forward pass at {fwd_start_time:.3f}\n")
            
            # Perform forward pass
            if self.rank <= self.num_perturbations:
                loss = teacher_forcing_loss_emb_parallel(
                    self.local_model, x_ids, y, criterion, chunk_size=1024
                )
            else:
                loss = torch.tensor(0.0, device=self.device)
            
            # Record end time for forward pass
            fwd_end_time = time.time()
            fwd_duration = fwd_end_time - fwd_start_time
            if self.verbose:
                print(f"[Rank {self.rank}] Completed forward pass at {fwd_end_time:.3f}, took {fwd_duration:.3f}s\n")
            
            # Synchronize after forward passes
            dist.barrier()
            
            # -------------------------------------------------------------
            # Step 8: Gather losses
            # -------------------------------------------------------------
            
            losses = self._gather_losses(loss)
            clean_loss_val = losses[0].item()
            
            # Rank 0 processes losses for logging
            if self.rank == 0:
                grad_ests = []
                for i in range(1, self.num_perturbations + 1):
                    if i < len(losses):
                        grad_est = (losses[i].item() - clean_loss_val) / epsilon
                        grad_ests.append(grad_est)
                
                perturbed_losses = [losses[i].item() for i in range(1, min(len(losses), self.num_perturbations + 1))]
                avg_perturbed = sum(perturbed_losses) / len(perturbed_losses) if perturbed_losses else 0.0
                avg_loss = (clean_loss_val + avg_perturbed) / 2
                
                if self.verbose:
                    print(f"Clean loss: {clean_loss_val:.4f}, Avg perturbed: {avg_perturbed:.4f}, Avg loss: {avg_loss:.4f}")
                    if grad_ests:
                        print(f"Grad estimates: {', '.join([f'{g:.6f}' for g in grad_ests])}")
            
            # -------------------------------------------------------------
            # Step 9: Scale probes by gradient estimates
            # -------------------------------------------------------------
            
            if 1 <= self.rank <= self.num_perturbations:
                perturbed_loss_val = losses[self.rank].item()
                grad_est = (perturbed_loss_val - clean_loss_val) / epsilon
                
                # Debug output for large gradient estimates
                if abs(grad_est) > 1000 and self.verbose:
                    print(f"[WARNING] Rank {self.rank} has large gradient estimate: {grad_est:.6f}")
                    print(f"  Clean loss: {clean_loss_val}, Perturbed loss: {perturbed_loss_val}, Epsilon: {epsilon}")
                
                # Scale probe by gradient estimate
                probe.mul_(grad_est)
            
            # -------------------------------------------------------------
            # Step 10: Average probes
            # -------------------------------------------------------------
            
            weighted_avg_probe = self._reduce_probes(probe)
            
            # Clean up
            probe = None
            torch.cuda.empty_cache()
            
            # -------------------------------------------------------------
            # Step 11-12: Update model
            # -------------------------------------------------------------
            
            if self.rank == 0:
                # Apply gradient to model parameters
                offset = 0
                for p, (shape, numel, _) in zip(self.model_main.parameters(), self.meta):
                    slice_ = weighted_avg_probe[offset:offset + numel].view(shape)
                    if p.grad is None:
                        p.grad = slice_.to(p.device)
                    else:
                        p.grad.copy_(slice_.to(p.device))
                    offset += numel
                
                # Update model with optimizer step
                optimizer.step()
                optimizer.zero_grad()
            
            # Final synchronization
            dist.barrier()
            
            # -------------------------------------------------------------
            # Step 13: Log timing info
            # -------------------------------------------------------------
            
            end_time = time.time()
            if self.rank == 0:
                iter_time = end_time - start_time
                if self.verbose:
                    print(f"Iteration completed in {iter_time:.4f}s")
                return avg_loss
            else:
                return 0.0





    def _broadcast_tensor(self, tensor, src_rank):
        """
        Utility function to broadcast a tensor from src_rank to all other ranks,
        handling the case where tensor size is unknown on non-src ranks.
        
        Args:
            tensor: Tensor to broadcast (should be valid on src_rank)
            src_rank: Source rank for broadcast
            
        Returns:
            Broadcasted tensor on all ranks
        """
        # First broadcast tensor size
        if self.rank == src_rank:
            size_tensor = torch.tensor(tensor.numel(), dtype=torch.long, device=self.device)
        else:
            size_tensor = torch.tensor(0, dtype=torch.long, device=self.device)
        
        dist.broadcast(size_tensor, src_rank)
        
        # Resize tensor on non-src ranks
        if self.rank != src_rank:
            tensor = torch.zeros(size_tensor.item(), device=self.device)
        
        # Broadcast tensor data
        dist.broadcast(tensor, src_rank)
        
        return tensor


    
    
    def _broadcast_data(self, param_data, ratio_data, seeds, x_ids, y):
        """
        Broadcast data from rank 0 to all other ranks.
        This includes model parameters, adam ratio, random seeds, and dataset.
        
        Args:
            param_data: Model parameters (valid on rank 0)
            ratio_data: Adam ratio (valid on rank 0)
            seeds: Random seeds (valid on rank 0)
            x_ids: Input token IDs (valid on rank 0)
            y: Target token IDs (valid on rank 0)
            
        Returns:
            Tuple of broadcasted data on all ranks
        """
        # Create and broadcast size information (use async operations)
        if self.rank == 0:
            param_size = torch.tensor(param_data.numel(), dtype=torch.long, device=self.device)
            ratio_size = torch.tensor(ratio_data.numel(), dtype=torch.long, device=self.device)
            seed_size = torch.tensor(len(seeds), dtype=torch.long, device=self.device)
            x_size_0 = torch.tensor(x_ids.size(0), dtype=torch.long, device=self.device)
            x_size_1 = torch.tensor(x_ids.size(1), dtype=torch.long, device=self.device)
            y_size_0 = torch.tensor(y.size(0), dtype=torch.long, device=self.device)
            y_size_1 = torch.tensor(y.size(1), dtype=torch.long, device=self.device)
        else:
            param_size = torch.tensor(0, dtype=torch.long, device=self.device)
            ratio_size = torch.tensor(0, dtype=torch.long, device=self.device)
            seed_size = torch.tensor(0, dtype=torch.long, device=self.device)
            x_size_0 = torch.tensor(0, dtype=torch.long, device=self.device)
            x_size_1 = torch.tensor(0, dtype=torch.long, device=self.device)
            y_size_0 = torch.tensor(0, dtype=torch.long, device=self.device)
            y_size_1 = torch.tensor(0, dtype=torch.long, device=self.device)
        
        # Broadcast sizes in parallel using async operations
        handles = []
        handles.append(dist.broadcast(param_size, 0, async_op=True))
        handles.append(dist.broadcast(ratio_size, 0, async_op=True))
        handles.append(dist.broadcast(seed_size, 0, async_op=True))
        handles.append(dist.broadcast(x_size_0, 0, async_op=True))
        handles.append(dist.broadcast(x_size_1, 0, async_op=True))
        handles.append(dist.broadcast(y_size_0, 0, async_op=True))
        handles.append(dist.broadcast(y_size_1, 0, async_op=True))
        
        # Wait for all size broadcasts to complete
        for handle in handles:
            handle.wait()
        
        # Allocate tensors for receiving data
        if self.rank != 0:
            param_data = torch.zeros(param_size.item(), device=self.device)
            ratio_data = torch.zeros(ratio_size.item(), device=self.device)
            seed_tensor = torch.zeros(seed_size.item(), dtype=torch.long, device=self.device)
            x_ids = torch.zeros((x_size_0.item(), x_size_1.item()), dtype=torch.long, device=self.device)
            y = torch.zeros((y_size_0.item(), y_size_1.item()), dtype=torch.long, device=self.device)
        else:
            # Move CPU tensors to GPU for broadcasting
            param_data_gpu = param_data.to(self.device)
            ratio_data_gpu = ratio_data.to(self.device)
            seed_tensor = torch.tensor(seeds, dtype=torch.long, device=self.device)
        
        # Broadcast data (all on GPU now, using async where possible)
        handles = []
        if self.rank == 0:
            handles.append(dist.broadcast(param_data_gpu, 0, async_op=True))
            handles.append(dist.broadcast(ratio_data_gpu, 0, async_op=True))
        else:
            handles.append(dist.broadcast(param_data, 0, async_op=True))
            handles.append(dist.broadcast(ratio_data, 0, async_op=True))
        
        # Wait for these critical broadcasts to complete
        for handle in handles:
            handle.wait()
        
        # Broadcast remaining data
        handles = []
        handles.append(dist.broadcast(seed_tensor, 0, async_op=True))
        handles.append(dist.broadcast(x_ids, 0, async_op=True))
        handles.append(dist.broadcast(y, 0, async_op=True))
        
        # Continue with CPU operations while broadcasts complete
        if self.rank != 0:
            param_data_cpu = param_data.cpu()
            ratio_data_cpu = ratio_data.cpu()
        else:
            param_data_cpu = param_data  # Already on CPU
            ratio_data_cpu = ratio_data  # Already on CPU
            # Clean up GPU tensors
            param_data_gpu = None
            ratio_data_gpu = None
        
        # Wait for remaining broadcasts
        for handle in handles:
            handle.wait()
        
        # Convert seed tensor to list
        if self.rank != 0:
            seeds = seed_tensor.tolist()
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        
        return param_data_cpu, ratio_data_cpu, seeds, x_ids, y



    
    
    def _gather_losses(self, loss):
        """
        Gather losses from all ranks.
        
        Args:
            loss: Loss tensor on current rank
            
        Returns:
            List of loss tensors from all ranks
        """
        # Make sure loss is a tensor
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor([loss], device=self.device)
        
        # Make sure loss is a tensor of size 1
        if loss.numel() != 1:
            loss = loss.view(1)
        
        # Gather all losses
        gathered_losses = [torch.zeros(1, device=self.device) for _ in range(self.world_size)]
        
        # Use all_gather instead of gather to ensure all processes participate
        dist.all_gather(gathered_losses, loss)
        
        return gathered_losses

    
    def _reduce_probes(self, probe):
        """
        Gather and average probes across perturbation ranks.
        """
        # Handle case where probe is None
        if probe is None:
            probe_size = torch.tensor([0], dtype=torch.long, device=self.device)
        else:
            probe_size = torch.tensor([probe.numel()], dtype=torch.long, device=self.device)
        
        # Broadcast probe size to all ranks
        dist.all_reduce(probe_size, op=dist.ReduceOp.MAX)
        max_size = probe_size.item()
        
        # Create empty probe for ranks without one
        if probe is None or probe.numel() == 0:
            probe = torch.zeros(max_size, device=self.device)
        
        # Zero out probe for non-participating ranks
        if not (1 <= self.rank <= self.num_perturbations):
            probe.zero_()
        
        # Create buffer for reduction result on rank 0
        result = probe.clone() if self.rank == 0 else None
        
        # Sum all probes to rank 0
        dist.reduce(probe, 0, op=dist.ReduceOp.SUM)
        
        # On rank 0, scale by number of perturbations
        if self.rank == 0 and self.num_perturbations > 0:
            probe.div_(self.num_perturbations)
            result_cpu = probe.cpu()
        else:
            result_cpu = torch.zeros(max_size, device='cpu')
        
        # Clean up
        probe = None
        torch.cuda.empty_cache()
        
        return result_cpu
        
    
    def _broadcast_updated_model(self, updated_params, updated_ratio):
        """
        Broadcast updated model parameters and adam ratio from rank 0 to all other ranks.
        
        Args:
            updated_params: Updated model parameters (valid on rank 0)
            updated_ratio: Updated adam ratio (valid on rank 0)
            
        Returns:
            Tuple of broadcasted parameters and ratio on all ranks
        """
        # Broadcast the sizes first
        if self.rank == 0:
            param_size = torch.tensor(updated_params.numel(), dtype=torch.long, device=self.device)
            ratio_size = torch.tensor(updated_ratio.numel(), dtype=torch.long, device=self.device)
        else:
            param_size = torch.tensor(0, dtype=torch.long, device=self.device)
            ratio_size = torch.tensor(0, dtype=torch.long, device=self.device)
        
        dist.broadcast(param_size, 0)
        dist.broadcast(ratio_size, 0)
        
        # For NCCL, we need to move CPU tensors to GPU for broadcasting
        if self.rank == 0:
            # Create GPU copies for broadcasting
            updated_params_gpu = updated_params.to(self.device)
            updated_ratio_gpu = updated_ratio.to(self.device)
        else:
            # Allocate tensors for receiving data on GPU
            updated_params_gpu = torch.zeros(param_size.item(), device=self.device)
            updated_ratio_gpu = torch.zeros(ratio_size.item(), device=self.device)
        
        # Broadcast data on GPU
        dist.broadcast(updated_params_gpu, 0)
        dist.broadcast(updated_ratio_gpu, 0)
        
        # Move back to CPU for the optimizer state
        if self.rank != 0:
            updated_params = updated_params_gpu.cpu()
            updated_ratio = updated_ratio_gpu.cpu()
        
        # Clean up GPU tensors
        updated_params_gpu = None
        updated_ratio_gpu = None
        torch.cuda.empty_cache()
        
        return updated_params, updated_ratio


try:
    from datasets import load_dataset
except ImportError:
    print("HuggingFace datasets library not found. OWT task will use dummy data.")

def train_distributed():
    parser = argparse.ArgumentParser()
    # Add distributed argument (automatically added by torch.distributed.launch)
    parser.add_argument("--local-rank", type=int, default=0, 
                      help="Local rank passed by torch.distributed.launch")
    
    parser.add_argument("--arch", type=str, default="ntm", 
                      choices=["ntm", "dnc", "tra", "tdnc", "tntm", "simplelstm", "mamba"])
    parser.add_argument("--task", type=str, default="copy",
                      choices=["copy", "repeat_copy", "associative_recall", "add", "sub", "mul", "div", "fib", "factorial", "owt"])
    parser.add_argument("--input_sample_length", type=int, default=150,
                      help="Base length for generating tasks. We'll do a simple curriculum on some tasks.")
    parser.add_argument("--max_seq_len", type=int, default=150,
                      help="For generation.")

    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--macro_batch_size", type=int, default=1)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--max_num", type=int, default=110,
                      help="This is the max number in the domain to use in training for arithmetic tasks.")

    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--memory_size", type=int, default=128)
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--tie_epsilon_to_lr_ratio", type=float, default=-1)
    parser.add_argument("--epsilon", type=float, default=1e-2, help="MeZO eps.")

    # In our version, we only support mezo_adaptive_sampling_fast_one_way
    parser.add_argument("--mezo_flavor", type=str, default="mezo_adaptive_sampling_fast_one_way", 
                      choices=["mezo_adaptive_sampling_fast_one_way"])

    parser.add_argument("--fixed_size_perturbation", action="store_true")
    
    parser.add_argument("--cosine_lr", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=100)

    parser.add_argument("--grad_norm", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="If >0, grad norm clipped.")

    parser.add_argument("--pad_bias", type=float, default=0.0, help="Initial logit bias for <PAD> in final layer.")
    parser.add_argument("--log_interval", type=int, default=300)
    parser.add_argument("--wandb_proj", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--minimum_starting_point_context_length", type=int, default=10, 
                      help="min seq length fed into the model to start the curriculum learning")
    parser.add_argument("--num_perturbations", type=int, default=1,
                      help="Number of perturbations used in the MeZO sampling.")
    parser.add_argument("--eps_schedule_multiplier", type=float, default=2,
                      help="Multiplier for epsilon schedule if use_same_eps_for_all_perturbations is False.")
    parser.add_argument("--use_same_probe_for_all_perturbations", action="store_true",
                      help="If set, all perturbations use the same random probe (z_data).")
    parser.add_argument("--use_same_eps_for_all_perturbations", action="store_true",
                      help="If set, all perturbations use the same epsilon value.")
    parser.add_argument("--aggregation_method", type=str, default="average",
                      choices=["average", "max", "weighted_average"],
                      help="Method to aggregate finite-difference updates.")
    parser.add_argument("--schedule_loss_window", type=int, default=100)
    parser.add_argument("--ema_coine_lr_alpha", type=float, default=0.01)
    parser.add_argument("--variance_reduction", type=float, default=1.)
    parser.add_argument("--reset_solver_after_plateau", type=int, default=-1)
    parser.add_argument("--num_global_perturbations", type=int, default=0)
    parser.add_argument("--probe_dropout_rate", type=float, default=0.0)
    parser.add_argument("--overfit_to_one_batch_flag", action="store_true",
                      help="Overfit to just one batch for testing.")
    parser.add_argument("--cosine_windowing", action="store_true",
                      help="Pause cosine schedule if train_loss is decreasing.")
    
    args = parser.parse_args()

    # Get number of available GPUs
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No CUDA devices available")
    
    # Get local rank from environment variable
    local_rank = args.local_rank
    
    # Check if local_rank is valid
    if local_rank >= available_gpus:
        print(f"Warning: Rank {local_rank} is greater than available GPUs ({available_gpus})")
        print(f"This process will exit gracefully to avoid errors")
        # Exit gracefully - do not call dist.init_process_group()
        return
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    
    # Initialize distributed environment
    if dist.is_available() and not dist.is_initialized():
        try:
            dist.init_process_group(backend="nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            device = f'cuda:{local_rank}'
        except Exception as e:
            print(f"Error initializing distributed environment: {e}")
            print("Falling back to single GPU mode")
            rank = 0
            world_size = 1
            device = f'cuda:{local_rank}'
    else:
        # Non-distributed mode
        rank = 0
        world_size = 1
        device = f'cuda:{local_rank}'

    if rank == 0:
        torch._C._jit_set_bailout_depth(1)  # Reduce JIT bailouts
        torch.jit.optimized_execution(True)
        print(f"Running with {world_size} processes on {available_gpus} GPUs")

    # Initialize logging - only on rank 0
    if rank == 0 and args.wandb_proj is not None:
        import wandb
        wandb.init(project=args.wandb_proj, name=args.wandb_run_name)
        wandb.config.update(vars(args))
    
    verbose = False 
    alpha = args.ema_coine_lr_alpha  # smoothing factor for the EMA
    ema_loss = None  # will hold the running EMA of the loss
    ema_history = []  # store EMA values for comparison later    

    total_samples_per_iter = args.micro_batch_size * args.macro_batch_size

    # Build vocab
    vocab_list, char_to_id, id_to_char = get_char_vocab()
    vocab_size = len(vocab_list)

    # Check available VRAM
    if rank == 0:
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
            args.mezo_flavor,
            True
        )
        total_estimated_vram_gb = vram_stats["total_estimated_gb"]
        print(f"Estimated VRAM usage: {total_estimated_vram_gb:.2f} GB")
    else:
        total_estimated_vram_gb = 0  # Only rank 0 calculates this
    
    # Create embedding model
    torch.cuda.reset_peak_memory_stats(device)
    embed = nn.Embedding(vocab_size, args.input_size).to(device)
    nn.init.orthogonal_(embed.weight)
    
    # Create model - only on rank 0, will be distributed to other ranks
    if rank == 0:
        if args.arch == "ntm":
            model = NTM(args.input_size, vocab_size, args.hidden_size, args.memory_size, args.head_size, args.num_heads, embed).to(device)
        elif args.arch == "dnc":
            model = DNC(args.input_size, vocab_size, args.hidden_size, args.memory_size, args.head_size, args.num_heads, embed).to(device)
        elif args.arch == "tdnc":
            model = TransformerMemoryDNC(args.input_size, vocab_size, args.hidden_size, args.memory_size, args.head_size, args.num_heads, embed).to(device)
        elif args.arch == "mamba":
            model = Mamba(args.input_size, vocab_size, args.hidden_size, args.memory_size, args.head_size, args.num_heads, embed).to(device)
        elif args.arch == "tntm":
            model = TransformerMemoryNTM(args.input_size, vocab_size, args.hidden_size, args.memory_size, args.head_size, args.num_heads, embed).to(device)    
        elif args.arch == "simplelstm":
            model = SimpleLSTM(args.input_size, vocab_size, args.hidden_size, embed).to(device)  
        else: # "tra"
            model = Transformer(args.input_size, vocab_size, args.hidden_size, args.memory_size, args.head_size, args.num_heads, embed).to(device) 
    else:
        # Other ranks will receive model parameters in MezoDistributedSampling
        model = None
    
    # Setup optimizer - only on rank 0
    if rank == 0:
        params = list(model.parameters()) + list(embed.parameters())
        optimizer = optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
        
        # Setup scheduler
        if args.cosine_lr:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iters, eta_min=args.learning_rate/1000)    
        else: 
            # default is LR plateau
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=100,
                verbose=True,
                min_lr=1e-8
            )
        
        # Warmup scheduler
        warmup_scheduler = WarmupScheduler(
            optimizer=optimizer,
            warmup_steps=args.warmup_steps,
            base_lr=1e-9,
            final_lr=args.learning_rate
        )
    else:
        optimizer = None
        scheduler = None
        warmup_scheduler = None
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Initialize MezoDistributedSampling - will handle distributed model setup
    mezo_wrapper = MezoDistributedSampling(
        model=model,
        num_perturbations=args.num_perturbations,
        probe_dropout_rate=args.probe_dropout_rate,
        verbose=True  # Only verbose on rank 0
    )
    
    # Prepare model for fast inference
    dummy_data = torch.zeros(args.micro_batch_size,
        args.minimum_starting_point_context_length,
        args.input_size,
        device=device
    )
    
    if rank == 0:
        model.eval()
        with torch.no_grad():
            prepare_model_for_fast_inference(model, dummy_data)
        
        print(f"CUDA memory usage: {torch.cuda.max_memory_allocated(device) / (1024 ** 3):.2f} GiB")
        print("Memory stats:")
        print(torch.cuda.memory_summary(device))
        print("Done preparing model")
    
    # Ensure all processes are synchronized before starting training
    if dist.is_initialized():
        dist.barrier()
    
    # Load dataset if needed
    ds = None
    if args.task == "owt":
        if rank == 0:
            print("Loading OpenWebText dataset...")
            try:
                iterr = 0
                while True:
                    try:
                        ds = load_dataset(
                            "haritzpuerto/the_pile_00_OpenWebText2",
                            split="train",
                            download_mode="reuse_dataset_if_exists"
                        )
                        break
                    except Exception as e:
                        print(f"Hugging face issue... {e}")
                        time.sleep(5)
                        iterr += 1
                        if iterr > 100:
                            raise Exception("HUGGING FACE ISSUES AGAIN!")
                print("Got the OWT dataset!")
            except Exception as e:
                print(f"Failed to load OpenWebText dataset: {e}")
                # Fallback - create a small dummy dataset for testing
                class DummyDataset:
                    def __init__(self):
                        self.texts = [
                            "This is a sample text for OpenWebText.",
                            "Another sample text to use for training.",
                            "A third sample text for demonstration purposes.",
                            "Yet another text sample to use in place of OWT."
                        ]
                    def __len__(self):
                        return len(self.texts)
                    def __getitem__(self, idx):
                        return {"text": self.texts[idx % len(self.texts)]}
                
                ds = DummyDataset()
                print("Created dummy OWT dataset for testing")
    
    # Training loop variables
    global_step = 0
    train_start_time = time.time()
    consecutive_succ = 0
    gen_efficiency_gen_loss = 0
    gen_efficiency_val_loss = 0
    
    minimum_starting_point_context_length = args.minimum_starting_point_context_length
    current_context_len = minimum_starting_point_context_length
    current_max_num = 15  # used only in arithmetic functions
    
    # Check curriculum settings
    assert current_max_num <= args.max_num, "Must have max_num > 15"
    assert current_context_len <= args.input_sample_length, f"Must have input_sample_length > 1: current_context_len:{current_context_len} and input_sample_length:{args.input_sample_length}"
    
    # Allow overfitting to one batch for testing
    overfit_to_one_batch_flag = args.overfit_to_one_batch_flag
    if overfit_to_one_batch_flag and rank == 0:
        print("Overfitting to one batch for testing")
        this_sample_context_length = 1 + np.random.randint( 
                                            minimum_starting_point_context_length, 
                                            max(1, current_context_len+1)
                        )
        this_sample_max_num = 1 + np.random.randint(0, max(1, current_max_num))
        
        x_strs, y_strs = generate_task_data(
            total_samples_per_iter, 
            args.task,
            this_sample_context_length,
            this_sample_max_num,
            train=True,
            ds=ds
        )
    
    # Track warmup steps
    warmup_counter = 0
    
    # Main training loop
    iteration = 0
    while True:#iteration < args.max_iters:
        iteration += 1
        iter_start_time = time.time()
        
        # Generate or reuse batch data - only on rank 0
        if rank == 0:
            if not overfit_to_one_batch_flag:
                this_sample_context_length = 1 + np.random.randint( 
                                                minimum_starting_point_context_length, 
                                                max(1, current_context_len+1)
                                )
                this_sample_max_num = 1 + np.random.randint(0, max(1, current_max_num))
                
                x_strs, y_strs = generate_task_data(
                    total_samples_per_iter, 
                    args.task,
                    this_sample_context_length,
                    this_sample_max_num,
                    train=True,
                    ds=ds
                )
        
        # Initialize loss sum for macro batches
        micro_loss_sum = 0.0
        
        # Process micro batches
        for micro_i in range(args.macro_batch_size):
            # Prepare batch data - only needed on rank 0, will be broadcast to other ranks
            if rank == 0:
                start_idx = micro_i * args.micro_batch_size
                end_idx = start_idx + args.micro_batch_size
                cur_x = x_strs[start_idx:end_idx]
                cur_y = y_strs[start_idx:end_idx]
                
                x_ids = str_to_tensor(cur_x, char_to_id).to(device)
                y_ids = str_to_tensor(cur_y, char_to_id).to(device)
                
                # Update epsilon if tied to learning rate
                if args.tie_epsilon_to_lr_ratio > -1:
                    args.epsilon = args.tie_epsilon_to_lr_ratio * optimizer.param_groups[0]['lr']
            else:
                # Other ranks will receive these in mezo_distributed_sampling
                x_ids = None
                y_ids = None
            
            # Call our distributed Mezo implementation
            loss_val = mezo_wrapper.mezo_distributed_sampling(
                x_ids=x_ids,
                y=y_ids,
                criterion=criterion,
                optimizer=optimizer,
                epsilon=args.epsilon
            )
            
            # Sum up losses for macro batches - only on rank 0
            if rank == 0:
                micro_loss_sum += loss_val
        
        # Apply warmup and LR scheduling - only on rank 0
        if rank == 0:
            train_loss_mezo = micro_loss_sum / args.macro_batch_size
            
            # Apply warmup
            if warmup_counter < args.warmup_steps:
                warmup_scheduler.step()
                warmup_counter += 1
            elif scheduler is not None:
                if args.cosine_lr:
                    if args.cosine_windowing:
                        # Update the exponential moving average (EMA) for the training loss
                        if ema_loss is None:
                            ema_loss = train_loss_mezo
                        else:
                            ema_loss = alpha * train_loss_mezo + (1 - alpha) * ema_loss
                        
                        # Append the current EMA to the history
                        ema_history.append(ema_loss)
                        
                        # Only start comparing once we have enough history
                        if len(ema_history) >= args.schedule_loss_window:
                            # The past EMA is taken from window_size epochs ago
                            ema_past = ema_history[-args.schedule_loss_window]
                            
                            print(f"  Current EMA: {ema_loss:.4f}, Past EMA (from {args.schedule_loss_window} epochs ago): {ema_past:.4f}")
                            
                            # Check for improvement and manage history
                            if len(ema_history) >= args.reset_solver_after_plateau * args.schedule_loss_window + 1:
                                del ema_history[0]  # remove oldest to manage size
                            
                            if ema_loss < ema_past - 1e-8:
                                # Improvement detected
                                print("  Improved EMA (current < past). Resetting patience.")
                            else:
                                scheduler.step()
                                print(f"  Not improving so scheduler stepped!")
                                
                                if len(ema_history) >= args.reset_solver_after_plateau * args.schedule_loss_window:
                                    ema_way_back_past = ema_history[0]
                                    if args.reset_solver_after_plateau > 0 and ema_loss < ema_way_back_past - 1e-8:
                                        # Reset everything if we're lost
                                        print("----------")
                                        print("  Resetting iteration, optim and schedulers because i am lost!")
                                        print("----------")
                                        
                                        warmup_counter = 0  # reset the warmup fresh
                                        
                                        optimizer = optim.Adam(params, 
                                                            lr=args.learning_rate, 
                                                            weight_decay=args.weight_decay,  
                                                            betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
                                        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                                        T_max=args.max_iters, 
                                                                                        eta_min=args.learning_rate/1000)
                                        warmup_scheduler = WarmupScheduler(
                                            optimizer=optimizer,
                                            warmup_steps=args.warmup_steps,
                                            base_lr=1e-9,
                                            final_lr=args.learning_rate
                                        )
                    else:
                        scheduler.step()
                else:
                    scheduler.step(train_loss_mezo)
            
            # Logging - only on rank 0
            iter_end_time = time.time()
            iteration_time = iter_end_time - iter_start_time
            total_elapsed = iter_end_time - train_start_time
            
            # Get current learning rate
            lr_current = optimizer.param_groups[0]["lr"]
            
            # Log memory usage
            vram_inferred = torch.cuda.max_memory_allocated(device)/1024**3
            
            # Log basic info every iteration
            msg = (f"Iter={iteration}, train_loss={train_loss_mezo:.3f}, "
                  f"LR={lr_current:.6f}, eps={args.epsilon:.6f}, vram_inferred={vram_inferred:.6f} GB, iter_time={iteration_time:.2f}s, total_time={total_elapsed/60:.2f}m, "
                  f"context_len={current_context_len}, max_num={current_max_num}, gen_eff_token={gen_efficiency_gen_loss}, gen_eff_sample={gen_efficiency_val_loss}")
            
            print(msg)
            sys.stdout.flush()
            
            # Check for divergence
            if train_loss_mezo > 100:
                print(f"Ending training - train_loss_mezo diverging: {train_loss_mezo}")
                break
            
            # Validation loop - run periodically
            if iteration % args.log_interval == 0:
                # Evaluate model on last micro-batch
                with torch.inference_mode():
                    # Convert tensors for evaluation
                    x_ids = str_to_tensor(cur_x, char_to_id).to(device)
                    y_ids = str_to_tensor(cur_y, char_to_id).to(device)
                    
                    # Generate sequences to check accuracy
                    generated_strs, generated_strs_with_padding, gen_ids_batch, probs_batch, train_loss, train_acc, train_acc_sample = generate_sequence_batched(
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
                    
                    # Print some samples
                    counter = 0
                    for b in range(len(generated_strs)):
                        print("="*30)
                        print(f" [Sample {b}] Input: {cur_x[b]}")
                        print(f" [Sample {b}] Target: {cur_y[b]}")
                        print(f" [Sample {b}] Generated (w/ spec toks): {generated_strs_with_padding[b]}")
                        print(f" [Sample {b}] Generated: {generated_strs[b]}")
                        print(f" [Sample {b}] Token IDs: {gen_ids_batch[b]}")
                        print(f" [Sample {b}] Probabilities: {probs_batch[b]}")
                        print("="*30)
                        counter += 1
                        if counter > 4:
                            break
                    
                    print(f"Generation loss: {train_loss}, Generation accuracy: {train_acc}, Generation sample accuracy: {train_acc_sample}")
                    print("="*30)
                    print("="*30)
                    
                    # Generate validation samples
                    this_sample_context_length_for_val = 1 + np.random.randint(
                        minimum_starting_point_context_length,
                        max(1, current_context_len+1)
                    )
                    
                    this_sample_max_num_for_val = 1 + np.random.randint(0, max(args.max_num, args.max_num))
                    
                    # Generate a validation batch
                    val_samples = 5
                    vx, vy = generate_task_data(
                        val_samples,
                        args.task,
                        this_sample_context_length_for_val,
                        this_sample_max_num_for_val,
                        train=False,
                        ds=ds
                    )
                    
                    # Convert to tensors
                    vx_ids = str_to_tensor(vx, char_to_id).to(device)
                    vy_ids = str_to_tensor(vy, char_to_id).to(device)
                    
                    # Compute teacher-forced validation loss
                    vx_emb = vx_ids
                    val_loss = teacher_forcing_loss_emb(model, vx_emb, vy_ids, criterion)
                    
                    # Do an auto-regressive generation pass
                    generated_strs, generated_strs_with_padding, gen_ids_batch, probs_batch, val_gen_loss, val_gen_acc, val_gen_acc_sample = generate_sequence_batched(
                        model=model,
                        x_ids=vx_ids,
                        embed=embed,
                        char_to_id=char_to_id,
                        id_to_char=id_to_char,
                        max_seq_len=args.max_seq_len,
                        device=device,
                        criterion=criterion,
                        y_ids=vy_ids
                    )
                
                # Print some validation samples
                sample_indices = random.sample(range(len(generated_strs)), min(3, len(generated_strs)))
                print("\n[DEBUG] Random Val Samples:")
                for idx in sample_indices:
                    print(f"  [Val idx={idx}]")
                    print(f"    Input:  '{vx[idx]}'")
                    print(f"    Target: '{vy[idx]}'")
                    print(f"    Generated(w/ spec toks): '{generated_strs_with_padding[idx]}'")
                    print(f"    Generated: '{generated_strs[idx]}'")
                    print(f"    Token IDs: {gen_ids_batch[idx]}")
                    print(f"    Probabilities: {probs_batch[idx]}")
                
                # Calculate efficiency metrics
                gen_efficiency_gen_loss = val_gen_loss * 100.0 / (total_estimated_vram_gb * (total_elapsed / 3600.0))
                gen_efficiency_val_loss = val_loss.item() * 100.0 / (total_estimated_vram_gb * (total_elapsed / 3600.0))
                
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
                
                # Update curriculum based on training accuracy
                new_ctx, new_mn, consecutive_succ = maybe_update_curriculum(train_acc, current_context_len, current_max_num, consecutive_succ)
                if current_context_len != new_ctx:
                    print(f"!!!!! UPDATING CURRICULUM FROM {current_context_len} to {new_ctx}")
                current_context_len = new_ctx
                current_max_num = new_mn
                
                sys.stdout.flush()
                
                # Log to wandb if enabled
                if args.wandb_proj is not None:
                    # Calculate weight decay loss term
                    weight_decay_loss = 0.0
                    for param in model.parameters():
                        if param.requires_grad:
                            weight_decay_loss += (args.weight_decay / 2) * torch.sum(param ** 2)
                    
                    # Prepare log data
                    log_data = {
                        "train_loss": train_loss_mezo,
                        "train_gen_loss": train_loss,
                        "train_acc": train_acc,
                        "lr": lr_current,
                        "iter_time_s": iteration_time,
                        "total_time_hours": total_elapsed / 3600.0,
                        "curr_context_len": current_context_len,
                        "curr_max_num": current_max_num,
                        "val_loss": val_loss.item(),
                        "val_gen_loss": val_gen_loss,
                        "val_gen_acc": val_gen_acc,
                        "total_estimated_vram_gb": total_estimated_vram_gb,
                        "GPU(GB)-Hours": total_estimated_vram_gb * total_elapsed / 3600.0,
                        "gen_efficiency_gen_loss": gen_efficiency_gen_loss,
                        "gen_efficiency_val_loss": gen_efficiency_val_loss,
                        "weight_decay_loss": weight_decay_loss.item(),
                        "vram_inferred": vram_inferred
                    }
                    
                    print("="*30)
                    print("VAL STATS")
                    print(log_data)
                    
                    try:
                        import wandb
                        wandb.log(log_data, step=iteration)
                    except Exception as e:
                        print(f"logging failed at iteration {iteration}. Error: {str(e)}")
    
    # Clean up - destroy process group
    if dist.is_initialized():
        dist.destroy_process_group()
    
    if rank == 0:
        print("Finished.")
        if args.wandb_proj is not None:
            try:
                import wandb
                wandb.finish()
            except:
                pass

if __name__ == "__main__":
    train_distributed()
