#!/usr/bin/env python3: python -m torch.distributed.launch --nproc_per_node=10 dnc_distributed.py 
import os, argparse, time
import torch
import torch.nn as nn
import torch.distributed as dist
from ntm_with_modern_training_runs import DNC, teacher_forcing_loss_emb_parallel
import datetime
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


        # we loop around this meta_perturbations times to apply meta_perturbations * (world_size-2) total perturbations
        for meta_pert in range(self.meta_perturbations):

            if self.verbose:
                log_msg(f"Perturbation {meta_pert+1} of {self.meta_perturbations}", rank)

            
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


            # Stage 3b: Dirty ranks receive eps * adam_ratio from Rank 0 and add it to probe. 
            if self.verbose:
                log_start("Stage 3b", rank)
                
            if rank == 0:
                for idx, p in enumerate(self.adam_m):
                    view_m = p
                    view_v = self.adam_v[idx]
                    
                    # Adam Bias Correction: correct the bias in the adam estimates
                    m_hat = view_m / (1 - self.beta1 ** (iteration+1) )
                    v_hat = view_v / (1 - self.beta2 ** (iteration+1) )
                    
                    ratio_chunk = m_hat / (v_hat.sqrt() + 1e-8)
                    # ratio_chunk = view_m / (view_v.sqrt() + 1e-11)

                    dist.broadcast(ratio_chunk, src=0)
                    
                    # dist.send(ratio_chunk, dst=1)
            if rank == 1:
                # On Rank 1, broadcast one parameter at a time
                for idx, p in enumerate(self.adam_ratio_list):

                    temp = torch.zeros_like(p, device=device)
                    # drop it on the floor. we dont need it or want it yet. Just too lazy to make a group with all but rank 1. 
                    dist.broadcast(temp, src=0)
  

            elif rank >= 2:
                for idx, p in enumerate(self.probe_list):
                    temp = torch.zeros_like(p, dtype=p.dtype, device=p.device)
                    dist.broadcast(temp, src=0)
                    # add to our probe
                    p.add_(temp * self.epsilon)
                
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
                    if meta_pert == 0:
                        probe.zero_() # so we remove the adam_ratio data from probe, only for the first time around, otherwise we add it in
                        
                    reduced_probe = reduce_mean_in_group(probe.clone(), dst_rank=1, group=self.group_except_zero, world_size=world_size)
                    self.adam_ratio_list[i] = reduced_probe  # Store the reduced value
        
                self.avg_probe_list = self.adam_ratio_list # just rename it so we do not confuse it.
                
                if self.verbose:
                    log_msg("Stage 9", rank, "Averaged scaled probes from dirty ranks.")
            dist.barrier()
            if self.verbose:
                log_end("Stage 9", rank)
            
            # -------------------- Stage 9b: Prep for another meta-perturbation --------------------
            if self.meta_perturbations-meta_pert-1 > 0:
                # we are going to go back around, so we need to:
                # 1) remove the grad_est scale from the probes on ranks >= 2
                # 2) subtract the probe from the dirty thetas on ranks >= 2
                if self.verbose:
                    log_start("Stage 9b", rank)
                if rank >= 2:
                    # On dirty ranks, divide grad_est, and sub probe from params
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
            return loss
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
    parser.add_argument("--wandb_proj", type=str, default="DNC-SINGLE-DISTRIBUTED", help="WandB project name (optional)")
    parser.add_argument("--wandb_run", type=str, default="test1", help="WandB run name (optional)")
        
    
    
    
    # New CLI arguments for model configuration
    parser.add_argument("--model_scale", type=int, default=1, help="Scaling factor for model dimensions.")
    parser.add_argument("--vocab_size", type=int, default=150, help="Vocabulary size.")
    parser.add_argument("--num_heads", type=int, default=1, help="# dnc heads.")
    parser.add_argument("--memory_size", type=int, default=1, help="memory_size.")
    parser.add_argument("--hidden_size", type=int, default=1, help="hidden_size.")
    parser.add_argument("--input_size", type=int, default=1, help="Input size.")
    parser.add_argument("--head_size", type=int, default=1, help="head_size .")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length.")
    parser.add_argument("--warmup_iters", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--meta_perturbations", type=int, default=1, help="Number of Perturbations for all ranks per step.")

    args = parser.parse_args()


    # Derived values based on model_scale
    args.hidden_size = 100 * args.model_scale
    args.memory_size = 100 * args.model_scale
    args.head_size = 100 * args.model_scale
    args.input_size = 100 * args.model_scale
    args.num_heads = 1  # Static, not dependent on model_scale
    


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
    

    log_start("INIT MODEL", rank)

    
    embed = nn.Embedding(args.vocab_size, args.input_size).to(device)
    model = DNC(input_size=args.input_size, output_size=args.vocab_size, hidden_size=args.hidden_size, memory_size=args.memory_size, head_size=args.head_size, num_heads=args.num_heads, embed=embed).to(device)
    
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
        cosine_wavelength = 1000  # Length of each cosine cycle
        lr_decay = 0.99
        current_lr = base_lr
        schedule_iteration = 0
        
        # Track previous loss
        prev_loss = float('inf')
        
    
    if rank == 1 and args.wandb_proj is not None and wandb is not None:
        wandb.init(project=args.wandb_proj, name=args.wandb_run)
        print("[Rank 1] Initialized wandb logging.")

    start_time = datetime.datetime.now()
    with torch.inference_mode():
        
        for i in range(int(args.max_iters) ):
            # TODO: Sample x_ids,y from dataset
            loss_val = distributed_adaptive.distributed_step(x_ids, y, criterion, iteration=i, warmup_iters=args.warmup_iters)
            
            if rank == 1:
                                
                # UPDATE THE LEARNING RATE WITH OUR FAST/SLOW EMA COSINE LR SCHEDULE
                if i < args.warmup_iters:
                    # Linear warmup
                    current_lr = base_lr * (i / args.warmup_iters)
                    
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
                    
                    try:
                        wandb.log(log_data, step=i)
                    except Exception as e:
                        print(e)

            dist.barrier()
            if rank==1:
                if loss_val < 0.1:

                    end_time = datetime.datetime.now()
                    print("="*50)
                    print("="*50)
                    print("="*50)
                    log_msg("FINISHED TRAINING", rank, f"in {i} iterations acheived {loss_val} loss in {end_time - start_time} seconds.")
                    print(f"[Init] Model has {num_params} parameters across {num_layers} layers.")
            
                    print("="*50)
                    print("="*50)
                    print("="*50)
                    time.sleep(200)
                    break

                    
                
            
        dist.destroy_process_group()
        

if __name__ == "__main__":
    main()
