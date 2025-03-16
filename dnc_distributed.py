#!/usr/bin/env python3
# python -m torch.distributed.launch --nproc_per_node=10 dnc_distributed.py 
import os, argparse, time
import torch
import torch.nn as nn
import torch.distributed as dist
from ntm_with_modern_training_runs import DNC, teacher_forcing_loss_emb_parallel
import wandb
import math

CHUNK_SIZE = 1048576

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
        # dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        tensor.div_(group_size)
    
    # Wait for everyone
    # dist.barrier()
    
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
    def __init__(self, model, learning_rate=0.001, probe_dropout_rate=0.99, epsilon=0.001,
                 beta1=0.9, beta2=0.999, verbose=True):
        self.learning_rate = learning_rate
        self.probe_dropout_rate = probe_dropout_rate
        self.epsilon = epsilon  # initial; later tied 1:1 with lr
        self.beta1 = beta1
        self.beta2 = beta2
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
                
            
            

        # Adam state is maintained only on Rank 0.
        # For Rank 0, we will later receive meta (number of parameters and shapes) from Rank 1.
        # if dist.get_rank() == 0:
        #     self.adam_initialized = False
        # else:
        #     self.adam_initialized = True


    
    def distributed_step(self, x_ids, y, criterion, iteration, warmup_iters=100):
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = self.param_list[0].device if self.param_list is not None else torch.device(f"cuda:{rank}")
        
        current_lr = self.learning_rate 
        self.epsilon = current_lr
        
        # -------------------- Meta Stage: Parameter Meta Communication --------------------
        # if self.param_list is None:
        #     print(f"UPDATING PARAM LIST: {rank}")
        #     if rank == 0:
        #         # Receive number of parameters from Rank 1.
        #         num_params_tensor = torch.empty(1, device=device, dtype=torch.int)
        #         dist.recv(num_params_tensor, src=1)
        #         num_params = int(num_params_tensor.item())
        #         param_list = []
        #         for i in range(num_params):
        #             shape_tensor = torch.empty(10, device=device, dtype=torch.int)
        #             dist.recv(shape_tensor, src=1)
        #             shape = tuple(shape_tensor[shape_tensor != 0].tolist())
        #             dummy = torch.empty(shape, device=device)
        #             param_list.append(dummy)
        #         self.param_list = param_list
        #         self.d = sum(p.numel() for p in self.param_list)
        #         # Initialize Adam state on Rank 0.
        #         self.adam_m = [torch.zeros_like(p) for p in self.param_list]
        #         self.adam_v = [torch.zeros_like(p) for p in self.param_list]
        #         self.adam_initialized = True
        #     else:
                
        #         if rank == 1:
        #             # Only rank 1 needs to send meta to Rank 0.
        #             num_params = len(self.param_list)
        #             num_params_tensor = torch.tensor([num_params], device=device, dtype=torch.int)
        #             dist.send(num_params_tensor, dst=0)
        #             for p in self.param_list:
        #                 shape_tensor = torch.zeros(10, device=device, dtype=torch.int)
        #                 shape_vals = list(p.shape)
        #                 shape_tensor[:len(shape_vals)] = torch.tensor(shape_vals, device=device, dtype=torch.int)
        #                 dist.send(shape_tensor, dst=0)
                    
        #             # Initialize adam_ratio_list with zeros on rank 1
        #             self.adam_ratio_list = [torch.zeros_like(p, dtype=p.dtype, device=p.device) for p in self.param_list]
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
        if rank >= 2:
            # Create empty tensors to receive the x and y data
            shape_tensor = torch.empty(2, device=device, dtype=torch.int)
            dist.broadcast(shape_tensor, src=1, group=self.group_except_zero)
            batch_size, seq_len = shape_tensor.tolist()
            
            # Create empty tensors with the right shapes
            x_ids = torch.zeros((batch_size, seq_len), device=device, dtype=torch.long)
            y = torch.zeros((batch_size, seq_len), device=device, dtype=torch.long)
                
            # Receive the actual data
            dist.broadcast(x_ids, src=1, group=self.group_except_zero)
            dist.broadcast(y, src=1, group=self.group_except_zero)
            
            if self.verbose:
                log_msg("Stage 0c", rank, f"Received x_ids and y from rank 1: shapes {x_ids.shape}, {y.shape}")
        elif rank == 1:
            # Send shapes first
            shape_tensor = torch.tensor([x_ids.shape[0], x_ids.shape[1]], device=device, dtype=torch.int)
            dist.broadcast(shape_tensor, src=1, group=self.group_except_zero)
            
            # Send actual data
            dist.broadcast(x_ids, src=1, group=self.group_except_zero)
            dist.broadcast(y, src=1, group=self.group_except_zero)
            
            if self.verbose:
                log_msg("Stage 0c", rank, f"Broadcast x_ids and y to dirty ranks: shapes {x_ids.shape}, {y.shape}")
        dist.barrier()
        if self.verbose:
            log_end("Stage 0c", rank)


        # -------------------- Stage 3: Dirty Ranks Compute Their Dirty theta_t --------------------
        # We break Stage 3 into four mini-stages.
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


        # Stage 3b: Dirty ranks receive eps * adam_ratio from Rank 1 and add it to probe.
        if self.verbose:
            log_start("Stage 3b", rank)
        if rank == 1:
            # On Rank 1, broadcast one parameter at a time
            for idx, p in enumerate(self.adam_ratio_list):
                temp = (p * self.epsilon).to(dtype=p.dtype, device=p.device)
                broadcast_in_group(temp, src_rank=1, group=self.group_except_zero)
        elif rank >= 2:
            for idx, p in enumerate(self.probe_list):
                # Create a receiving buffer with the same shape
                temp = torch.zeros_like(p, dtype=p.dtype, device=p.device)
                broadcast_in_group(temp, src_rank=1, group=self.group_except_zero)
                # add to our probe
                p.add_(temp)
            
        # Single barrier at the end for all ranks
        if self.verbose:
            log_end("Stage 3b", rank)
        dist.barrier()
        

        # Stage 3c: Dirty ranks apply a random dropout mask to their probe.
        if self.verbose:
            log_start("Stage 3c", rank)
        if rank >= 2:
            for probe in self.probe_list:
                if self.probe_dropout_rate > 0:
                    mask = (torch.rand_like(probe) > self.probe_dropout_rate).float()
                    probe.mul_(mask)
                    del mask
        dist.barrier()
        if self.verbose:
            log_end("Stage 3c", rank)

        # Stage 3d: Dirty theta_t = clean theta_t + modified probe.
        if self.verbose:
            log_start("Stage 3d", rank)
        if rank >= 2:
            # On dirty ranks, add received value to local probe
            for param, probe in zip(self.param_list, probe_list):
                param.add_(probe)
                
        dist.barrier()
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
                # temp = torch.empty((), device=device)
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
                grad_est_dict[r] = grad_est/(self.epsilon + 1e-8) # take eps out of it 
            for r, ge in grad_est_dict.items():
                dist.send(ge, dst=r)
            self.grad_est_dict = grad_est_dict
            if self.verbose:
                log_msg("Stage 6", rank, f"Computed grad_est per dirty rank: {grad_est_dict}")
        elif rank >= 2:
            grad_est = torch.tensor(0.0, device=device) # torch.empty((), device=device)
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
                # probe.mul_(self.grad_est)
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
                probe.zero_()
                reduced_probe = reduce_mean_in_group(probe.clone(), dst_rank=1, group=self.group_except_zero, world_size=world_size)
                self.adam_ratio_list[i] = reduced_probe  # Store the reduced value
    
            self.avg_probe_list = self.adam_ratio_list # just rename it so we do not confuse it.
            
            if self.verbose:
                log_msg("Stage 9", rank, "Averaged scaled probes from dirty ranks.")
        dist.barrier()
        if self.verbose:
            log_end("Stage 9", rank)

        # -------------------- Stage 10: Rank 1 Streams Averaged Probe to Rank 0; Rank 0 Updates Adam State --------------------
      
        if self.verbose:
            log_start("Stage 10a", rank)
        if rank == 1:
            for idx, p in enumerate(self.avg_probe_list):
                dist.send(p, dst=0)
            if self.verbose:
                log_msg("Stage 10a", rank, "Streamed averaged probe to Rank 0.")
        elif rank == 0: # recieve it from rank 0 and in place update
            for idx, p in enumerate(self.adam_m):
                # view_m = p
                # view_v = self.adam_v[idx]
                temp = torch.zeros_like(p, device=device)
                dist.recv(temp, src=1)
                p.mul_(self.beta1).add_(temp, alpha=1 - self.beta1)
                self.adam_v[idx].mul_(self.beta2).addcmul_(temp, temp, value=1 - self.beta2)
                # view_m.mul_(self.beta1)
                # view_m.add_(temp, alpha=1 - self.beta1)
                # view_v.mul_(self.beta2)
                # view_v.addcmul_(temp, temp, value=1 - self.beta2)
                
            if self.verbose:
                log_msg("Stage 10a", rank, "Updated adam_v and adam_m")
        if self.verbose:
            log_end("Stage 10a", rank)
        
        dist.barrier()
        
        # -------------------- Stage 10b: Stream adam_ratio from rank 0 to rank 1 --------------------
        if self.verbose:
            log_start("Stage 10b", rank)
        if rank == 0:
            for idx, p in enumerate(self.adam_m):
                view_m = p
                view_v = self.adam_v[idx]
                
                # Adam Bias Correction: correct the bias in the adam estimates
                m_hat = view_m / (1 - self.beta1 ** (iteration+1) )
                v_hat = view_v / (1 - self.beta2 ** (iteration+1) )
                
                ratio_chunk = m_hat / (v_hat.sqrt() + 1e-8)
                # ratio_chunk = view_m / (view_v.sqrt() + 1e-11)
                
                dist.send(ratio_chunk, dst=1)
            if self.verbose:
                log_msg("Stage 10b", rank, "Streamed adam_ratio chunks to Rank 1.")
        elif rank == 1:
            
            for idx, p in enumerate(self.adam_ratio_list):
                temp = torch.zeros_like(p, device=device)
                dist.recv(temp, src=0)
                p.copy_(temp)
            if self.verbose:
                log_msg("Stage 10b", rank, "Reconstructed adam_ratio from received chunks.")
        dist.barrier()
        if self.verbose:
            log_end("Stage 10b", rank)

        # -------------------- Stage 11: On Rank 1, Update theta_t with New adam_ratio --------------------
        if self.verbose:
            log_start("Stage 11", rank)
        if rank == 1:
            # list_inplace_add list_inplace_sub
            list_inplace_sub(self.param_list, self.adam_ratio_list, alpha=current_lr )
            if self.verbose:
                log_msg("Stage 11", rank, f"Updated theta_t with learning rate {current_lr:.6f} in place.")
            log_msg(" ", rank, f"gathered_losses {gathered_losses}.")
            log_msg(" ", rank, f"grad_est_dict {grad_est_dict}.")
            
        dist.barrier()
        if self.verbose:
            log_end("Stage 11", rank)

        
        if rank == 1:
            return loss, current_lr
        else:
            return None



# =============================================================================
# Main Routine
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed training.")
    parser.add_argument("--mode", type=str, choices=["test", "train"], default="train", help="Run mode: test or train.")
    parser.add_argument("--max_iters", type=int, default=1e10, help="Maximum iterations for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Base learning rate (and eps, tied 1:1).")
    parser.add_argument("--epsilon", type=float, default=0.001, help="Perturbation scale epsilon (tied to learning rate).")
    parser.add_argument("--probe_dropout_rate", type=float, default=0.99, help="Dropout rate for probe vector.")
    parser.add_argument("--wandb_proj", type=str, default="DNC-SINGLE-BATCH-MEMORIZE", help="WandB project name (optional)")
    parser.add_argument("--wandb_run", type=str, default="test1", help="WandB run name (optional)")
    
    args = parser.parse_args()
    
    


    # TEMP OVERRIDE FOR NOW SO WE CAN DEBUG
    args.wandb_proj = None
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

    # Only Rank 1 creates the full model and input data.
    

    log_msg("Trying first dist.barrier(), if hanging here, no infiniband likely on node, need to turn off p2p",rank,"if so, run export NCCL_P2P_DISABLE=1")
    dist.barrier()
    
    
    model_scale=200
    vocab_size = 150
    hidden_size = 100 * model_scale
    memory_size = 100 * model_scale
    head_size = 100 * model_scale
    num_heads = 10
    input_size = 100 * model_scale
    batch_size = 1
    seq_len = 100
    warmup_iters = 100

    log_start("INIT MODEL", rank)

    if rank != 0:
        embed = nn.Embedding(vocab_size, input_size).to(device)
        model = DNC(input_size=input_size, output_size=vocab_size, hidden_size=hidden_size,
                    memory_size=memory_size, head_size=head_size, num_heads=num_heads, embed=embed).to(device)
        x_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long) # PLACEHOLDER
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device) # PLACEHOLDER
        if rank == 1:
            num_params = sum(p.numel() for p in model.parameters())
            num_layers = len(list(model.children()))
            print(f"[Init] Model has {num_params} parameters across {num_layers} layers.")


    elif rank == 0:
        
        embed = nn.Embedding(vocab_size, input_size).to(device)
        model = DNC(input_size=input_size, output_size=vocab_size, hidden_size=hidden_size,
                    memory_size=memory_size, head_size=head_size, num_heads=num_heads, embed=embed).to(device)
        x_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long) # TODO NOT NEEDED!
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device) # TODO NOT NEEDED!
    dist.barrier()
    log_end("INIT MODEL", rank)
    
    criterion = nn.CrossEntropyLoss().to(device)

    distributed_adaptive = MezoAdaptiveSamplingParallel(
        model=model,
        learning_rate=args.learning_rate,
        probe_dropout_rate=args.probe_dropout_rate,
        epsilon=args.epsilon,
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
        cosine_wavelength = 1000  # Length of each cosine cycle
        lr_decay = 0.99
        current_lr = base_lr
        schedule_iteration = 0
        
        # Track previous loss
        prev_loss = float('inf')
        
    
    if rank == 1 and args.wandb_proj is not None and wandb is not None:
        wandb.init(project=args.wandb_proj, name=args.wandb_run)
        print("[Rank 1] Initialized wandb logging.")

    
    with torch.inference_mode():
        
        for i in range(int(args.max_iters) ):
            # TODO: Sample x_ids,y from dataset
            output = distributed_adaptive.distributed_step(x_ids, y, criterion, iteration=i, warmup_iters=warmup_iters)
            
            if rank == 1:
                
                loss_val, _ = output
                
                # UPDATE THE LEARNING RATE WITH OUR FAST/SLOW EMA COSINE LR SCHEDULE
                if i < warmup_iters:
                    # Linear warmup
                    current_lr = base_lr * (i / warmup_iters)
                    
                else:
                    # Update EMAs at every iteration
                    if loss_ema_fast is None:
                        # First iteration - initialize both EMAs to the current loss
                        loss_ema_fast = loss_val
                        loss_ema_slow = loss_val
                    else:
                        # Update both EMAs with their respective decay rates
                        loss_ema_fast = ema_alpha_fast * loss_ema_fast + (1 - ema_alpha_fast) * loss_val
                        loss_ema_slow = ema_alpha_slow * loss_ema_slow + (1 - ema_alpha_slow) * loss_val
    

                    
                    # Compare fast vs slow EMA to detect if we are not improving, step schedule
                    if loss_ema_fast > (loss_ema_slow + 1e-5) and i > (warmup_iters + 1000):
                        # Loss is decreasing or stable, follow the schedule
                        schedule_iteration+=1

                        if schedule_iteration%100 == 0:
                            # Calculate position within current cycle
                            cycle_iteration = (schedule_iteration // 100) % cosine_wavelength
                            
                            # Calculate cosine annealing for this cycle
                            progress = cycle_iteration / cosine_wavelength
                            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                            current_lr = min_lr + (base_lr - min_lr) * cosine_factor
                        
                        
                
                # Update the learning rate in the optimizer
                distributed_adaptive.learning_rate = current_lr
                            
                # Log information
                print("="*50)
                print(f"[Train] Iteration {i}, loss = {loss_val}, loss_ema_fast = {loss_ema_fast}, loss_ema_slow = {loss_ema_slow}, lr = {current_lr}")
                
                # Only Rank 1 logs metrics to wandb every 100 iterations.
                if args.wandb_proj is not None and wandb is not None and (i % 100 == 0):
                    # Compute a dummy weight decay loss (if applicable)
                    weight_decay_loss = 0.0
                    for param in model.parameters():
                        if param.requires_grad:
                            weight_decay_loss += (1e-2 / 2) * torch.sum(param ** 2)  # using 1e-2 as dummy weight_decay
        
                    log_data = {
                        "train_loss": loss_val, 
                        "loss_ema_fast":loss_ema_fast,
                        "loss_ema_slow":loss_ema_slow,
                        "lr": current_lr,
                        "weight_decay_loss": weight_decay_loss.item(),
                    }
                    time.sleep(1)
                    
                    try:
                        wandb.log(log_data, step=i)
                    except Exception as e:
                        print(e)

            dist.barrier()
            
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
