#!/usr/bin/env python3
"""
Neural Turing Machine / DNC / Transformer with MeZO, all using ASCII char-based tasks
(copy, repeat_copy, associative_recall, add, sub, mul, div, fib, factorial).
We use warmup & cosine LR logic, wandb logging, mezo single step and layerwise, etc.
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

    base_list = digits + letters
    final_list = []
    for ch in base_list:
        if ch not in special:
            final_list.append(ch)

    vocab_list = special + final_list  # ensures <PAD>=0, <bos>=1, <eos>=2, etc.
    char_to_id = {}
    id_to_char = {}
    for i, ch in enumerate(vocab_list):
        char_to_id[ch] = i
        id_to_char[i] = ch
    return vocab_list, char_to_id, id_to_char

##############################################################################
# Convert strings -> fixed [B, max_seq_len], pad or truncate
##############################################################################
def str_to_tensor(batch_strs, char_to_id, max_seq_len):
    B = len(batch_strs)
    out = torch.zeros(B, max_seq_len, dtype=torch.long)
    for i, s in enumerate(batch_strs):
        for j, ch in enumerate(s):
            if j >= max_seq_len:
                break
            out[i, j] = char_to_id.get(ch, 0)
    return out

##############################################################################
# Shift-by-one logic for tasks
##############################################################################
def shift_by_one_pairs(x_str, y_str):
    return f"<bos>{x_str}", f"{y_str}<eos>"

##############################################################################
# Task Generators
##############################################################################
def generate_copy_task_str(num_samples, context_length, train=True):
    # if not train => context_length *=5 (but we may override with curriculum too)
    import random
    letters = string.ascii_uppercase
    in_list, out_list = [], []
    for _ in range(num_samples):
        data_str = "".join(random.choice(letters) for _ in range(context_length))
        xinp, xtgt = shift_by_one_pairs(data_str, data_str)
        in_list.append(xinp)
        out_list.append(xtgt)
    return in_list, out_list

def generate_repeat_copy_task_str(num_samples, context_length, repeat_min=1, repeat_max=3, train=True):
    import random
    letters = string.ascii_uppercase
    in_list, out_list = [], []
    for _ in range(num_samples):
        data_str = "".join(random.choice(letters) for __ in range(context_length))
        c_val = random.randint(repeat_min, repeat_max)
        repeated = data_str*c_val
        xinp, xtgt = shift_by_one_pairs(data_str+str(c_val)+"|", repeated)
        in_list.append(xinp)
        out_list.append(xtgt)
    return in_list, out_list

def generate_associative_recall_task_str(num_samples, item_len=3, num_items=3, train=True):
    import random
    letters = string.ascii_uppercase
    in_list, out_list = [], []
    for _ in range(num_samples):
        items = ["".join(random.choice(letters) for __ in range(item_len)) for __ in range(num_items)]
        q_idx = random.randint(0, num_items-2)
        query_item = items[q_idx]
        ans_item = items[q_idx+1]
        flat_items= "".join(items)
        xinp, xtgt = shift_by_one_pairs(flat_items+"|"+query_item, ans_item)
        in_list.append(xinp)
        out_list.append(xtgt)
    return in_list, out_list

def generate_arithmetic_task_str(num_samples, context_length, task_type="add", 
                                 max_num=10, train=True):
    """
    We'll interpret 'max_num' as the upper domain
    For train, domain is [0..max_num], for val maybe [max_num+1.. max_num+5]? 
    But you can do your custom approach. 
    We'll do shift_by_one => input= <bos> expr, target= answer<eos>
    """
    import random
    in_list, out_list = [], []
    lo= 0
    hi= max_num
    for _ in range(num_samples):
        a = random.randint(lo, hi)
        b = random.randint(lo, hi)
        if task_type=='sub':
            if b>a:
                a,b= b,a
            res = a-b
            op='-'
        elif task_type=='mul':
            res= a*b
            op='*'
        elif task_type=='div':
            if b>a:
                a,b= b,a
            if b==0:
                b=1
            res= a//b
            op='/'
        else: # add
            res= a+b
            op='+'
        expr_in= f"{a}{op}{b}="
        out_str= f"{res}"
        xinp, xtgt= shift_by_one_pairs(expr_in, out_str)
        in_list.append(xinp)
        out_list.append(xtgt)
    return in_list, out_list

def generate_fibonacci_task_str(num_samples, context_length, max_n=10, train=True):
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

def generate_factorial_task_str(num_samples, context_length, max_n=6, train=True):
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
    def __init__(self, input_size, output_size, hidden_size, memory_size, head_size, num_heads):
        super(NTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.memory_size = memory_size
        self.head_size = head_size

        self.controller = nn.LSTM(input_size + num_heads * head_size, hidden_size, batch_first=True)
        self.fc_head = nn.Linear(hidden_size, num_heads * (head_size + memory_size + 3))
        self.fc_out = nn.Linear(hidden_size + num_heads * head_size, output_size)

    def _addressing(self, memory, head_params, prev_weights):
        B, N, _ = memory.size()
        read_keys = head_params[:, :, :self.head_size]
        read_weights = torch.einsum('bnk,bmk->bnm', read_keys, memory)/(self.head_size**0.5)
        read_weights = F.softmax(read_weights, dim=-1)
        read_content = torch.einsum('bnm,bmh->bnh', read_weights, memory)

        write_keys = head_params[:, :, self.head_size:2*self.head_size]
        write_strength = torch.sigmoid(head_params[:, :, 2*self.head_size:2*self.head_size+1])

        write_weights = torch.einsum('bnk,bmk->bnm', write_keys, memory)/(self.head_size**0.5)
        write_weights = F.softmax(write_weights, dim=-1)
        write_content = write_strength * write_keys
        delta = torch.einsum('bnm,bnh->bmh', write_weights, write_content)
        memory = memory + delta
        return memory, read_content, read_weights, write_weights

    def forward(self, x_emb, hidden=None, memory=None):
        B, L, E = x_emb.size()
        if hidden is None:
            h0= x_emb.new_zeros(1,B,self.hidden_size)
            c0= x_emb.new_zeros(1,B,self.hidden_size)
            hidden= (h0,c0)
        if memory is None:
            memory= x_emb.new_zeros(B, self.memory_size, self.head_size)

        outputs= []
        read_contents= x_emb.new_zeros(B, self.num_heads, self.head_size)
        for t in range(L):
            inp_t= torch.cat([x_emb[:, t, :], read_contents.view(B, -1)], dim=-1).unsqueeze(1)
            out_ctrl, hidden= self.controller(inp_t, hidden)
            h= out_ctrl.squeeze(1)
            head_params= self.fc_head(h).view(B, self.num_heads, self.head_size+ self.memory_size+3)
            memory, read_contents,_,_ = self._addressing(memory, head_params, None)
            out= torch.cat([h, read_contents.view(B, -1)], dim=-1)
            out= self.fc_out(out)
            outputs.append(out.unsqueeze(1))
        outputs= torch.cat(outputs, dim=1)
        return outputs, memory, hidden


class DNC(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, memory_size, head_size, num_heads=1):
        super(DNC, self).__init__()
        assert num_heads==1
        self.input_size= input_size
        self.output_size= output_size
        self.hidden_size= hidden_size
        self.memory_size= memory_size
        self.head_size= head_size
        self.num_reads= num_heads

        self.controller= nn.LSTM(input_size+ self.num_reads*self.head_size, hidden_size, batch_first=True)
        self.interface_size= (self.head_size*2 + 2 + self.head_size+1 + 3)
        self.fc_interface= nn.Linear(hidden_size, self.interface_size)
        self.fc_output= nn.Linear(hidden_size+ self.num_reads*self.head_size, output_size)

        self.reset_memory()

    def reset_memory(self, batch_size=1, device="cpu"):
        self.memory= None
        self.usage= None
        self.precedence= None
        self.link= None
        self.read_weights= None
        self.write_weights= None

    def _init_memory_if_needed(self, batch_size, device):
        if self.memory is None or self.memory.size(0)!= batch_size:
            self.memory= torch.zeros(batch_size,self.memory_size,self.head_size, device=device)
            self.usage= torch.zeros(batch_size,self.memory_size, device=device)
            self.precedence= torch.zeros(batch_size,self.memory_size, device=device)
            self.link= torch.zeros(batch_size,self.memory_size,self.memory_size, device=device)
            self.read_weights= torch.zeros(batch_size,self.memory_size, device=device)
            self.write_weights= torch.zeros(batch_size,self.memory_size, device=device)

    def forward(self, x_emb, hidden=None):
        B, L, E= x_emb.size()
        device= x_emb.device
        self._init_memory_if_needed(B, device)
        if hidden is None:
            h0= x_emb.new_zeros(1,B,self.hidden_size)
            c0= x_emb.new_zeros(1,B,self.hidden_size)
            hidden= (h0,c0)

        read_vec= x_emb.new_zeros(B, self.head_size)
        outs=[]
        for t in range(L):
            inp_t= torch.cat([x_emb[:, t, :], read_vec], dim=-1).unsqueeze(1)
            out_ctrl, hidden= self.controller(inp_t, hidden)
            h= out_ctrl.squeeze(1)
            interface= self.fc_interface(h)
            offset=0
            erase_vec= torch.sigmoid(interface[..., offset: offset+ self.head_size])
            offset+= self.head_size
            write_vec= interface[..., offset: offset+ self.head_size]
            offset+= self.head_size
            write_gate= torch.sigmoid(interface[..., offset: offset+1]).squeeze(-1)
            offset+=1
            alloc_gate= torch.sigmoid(interface[..., offset: offset+1]).squeeze(-1)
            offset+=1

            alloc_w= self._get_allocation_weights()
            w_gate= write_gate.unsqueeze(-1)
            a_gate= alloc_gate.unsqueeze(-1)
            write_w= w_gate*a_gate* alloc_w
            self.write_weights= write_w

            erase_mat= erase_vec.unsqueeze(1)
            self.memory= self.memory* (1- torch.bmm(write_w.unsqueeze(-1), erase_mat))
            add_mat= write_vec.unsqueeze(1)
            self.memory= self.memory+ torch.bmm(write_w.unsqueeze(-1), add_mat)
            self._update_usage()
            self._update_temporal_link(write_w)

            read_key= interface[..., offset: offset+ self.head_size]
            offset+= self.head_size
            def softplus(z): return torch.log1p(torch.exp(z))
            read_strength= softplus(interface[..., offset: offset+1]).squeeze(-1)
            offset+=1
            read_mode= interface[..., -3:]
            read_mode= F.softmax(read_mode, dim=-1)

            cw= self._content_addressing(read_key, read_strength)
            bw= torch.bmm(self.link.transpose(1,2), self.read_weights.unsqueeze(-1)).squeeze(-1)
            fw= torch.bmm(self.link, self.read_weights.unsqueeze(-1)).squeeze(-1)
            read_w= read_mode[...,0:1]*bw + read_mode[...,1:2]*fw + read_mode[...,2:3]*cw
            read_w= read_w+ 1e-8
            read_w= read_w / read_w.sum(dim=-1, keepdim=True)
            self.read_weights= read_w
            read_vec= torch.bmm(read_w.unsqueeze(1), self.memory).squeeze(1)

            out= torch.cat([h, read_vec], dim=-1)
            out= self.fc_output(out)
            outs.append(out.unsqueeze(1))

        outs= torch.cat(outs, dim=1)
        self.memory= self.memory.detach()
        self.usage= self.usage.detach()
        self.precedence= self.precedence.detach()
        self.link= self.link.detach()
        self.read_weights= self.read_weights.detach()
        self.write_weights= self.write_weights.detach()
        hidden= (hidden[0].detach(), hidden[1].detach())
        return outs, (self.memory, self.usage, self.link, self.precedence), hidden

    def _update_usage(self):
        self.usage= self.usage+ (1- self.usage)* self.write_weights

    def _get_allocation_weights(self):
        usage_sorted, idx_sorted= torch.sort(self.usage, dim=-1)
        alloc_w= torch.zeros_like(self.usage)
        cprod= torch.cumprod(usage_sorted, dim=-1)
        cprod= F.pad(cprod[:, :-1], (1,0), value=1.0)
        alloc_in_order= (1- usage_sorted)* cprod
        alloc_w.scatter_(1, idx_sorted, alloc_in_order)
        return alloc_w

    def _update_temporal_link(self, write_w):
        ww_ij= write_w.unsqueeze(-1)+ write_w.unsqueeze(1)
        self.link= (1- ww_ij)* self.link
        self.link+= torch.bmm(self.precedence.unsqueeze(-1), write_w.unsqueeze(1))
        diag= torch.eye(self.memory_size, device= write_w.device).unsqueeze(0)
        self.link= self.link*(1- diag)
        self.precedence= (1- write_w.sum(dim=-1, keepdim=True))* self.precedence
        self.precedence= self.precedence+ write_w

    def _content_addressing(self, key, strength):
        dot= torch.einsum("bkw,bnw->bn", key.unsqueeze(1), self.memory)
        key_norm= torch.norm(key,2,dim=-1,keepdim=True)+ 1e-8
        mem_norm= torch.norm(self.memory,2,dim=-1)+ 1e-8
        dot= dot/(key_norm*mem_norm)
        dot= dot* strength.unsqueeze(-1)
        return F.softmax(dot, dim=-1)


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
    def __init__(self, input_size, output_size, hidden_size):
        super(TransformerNTM, self).__init__()
        self.transformer= TransformerController(d_model=input_size, nhead=4,
                                                num_layers=2, dim_feedforward=4*input_size)
        self.fc_out= nn.Linear(input_size, output_size)

    def forward(self, x_emb, hidden=None, memory=None):
        trans_out= self.transformer(x_emb)
        out= self.fc_out(trans_out)
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


##############################################################################
# MeZO single
##############################################################################
def mezo_char_single(model, x, y, criterion, epsilon=1e-3):
    """
    We do 2 forward passes for entire model in a single random direction approach,
    but instead of directly updating param.data, we set param.grad => letting momentum handle it.
    """
    all_params = list(model.parameters())
    for p in all_params:
        if p.grad is not None:
            p.grad.zero_()

    orig = [p.data.clone() for p in all_params]
    directions = [torch.randn_like(p) if p.requires_grad else None for p in all_params]

    with torch.no_grad():
        # +epsilon
        for p, d in zip(all_params, directions):
            if p.requires_grad and d is not None:
                p.data.add_(epsilon*d.sign())

        out, _, _ = model(x)
        B, L, V = out.size()
        loss_plus = criterion(out.view(B*L, V), y.view(B*L))

        # -2 epsilon
        for p, d in zip(all_params, directions):
            if p.requires_grad and d is not None:
                p.data.sub_(2.0* epsilon*d.sign())

        out_m, _, _= model(x)
        Bm,Lm,Vm= out_m.size()
        loss_minus= criterion(out_m.view(Bm*Lm,Vm), y.view(Bm*Lm))

        # restore
        for p, od in zip(all_params, orig):
            p.data.copy_(od)

    # This is the total gradient => param.grad
    grad_est= (loss_plus- loss_minus)/(2* epsilon)

    # set param.grad => letting momentum handle it
    for p,d in zip(all_params,directions):
        if p.requires_grad and d is not None:
            if p.grad is None:
                p.grad = (grad_est* d.sign())
            else:
                p.grad.add_( grad_est* d.sign())

    return 0.5*(loss_plus.item()+ loss_minus.item())


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
                p.data.add_(epsilon * directions[i].sign())

            out_p,_,_= model(x)
            Bp,Lp,Vp= out_p.size()
            loss_plus= criterion(out_p.view(Bp*Lp, Vp), y.view(Bp*Lp))

            # -2 eps
            for i, (_, p) in enumerate(param_list):
                p.data.sub_(2.0* epsilon* directions[i].sign())

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
                    p.grad= grad_est* directions[i].sign()
                else:
                    p.grad.add_( grad_est* directions[i].sign())

            avg_loss= 0.5*(loss_plus.item()+ loss_minus.item())
            total_loss+= avg_loss

    if layer_count>0:
        total_loss/= float(layer_count)
    return total_loss


##############################################################################
# Main
##############################################################################
def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="ntm", choices=["ntm","dnc","tra"])
    parser.add_argument("--task", type=str, default="copy",
                        choices=["copy","repeat_copy","associative_recall","add","sub","mul","div","fib","factorial"])
    parser.add_argument("--context_length", type=int, default=10,
                        help="Base length for generating tasks. We'll do a simple curriculum on some tasks.")
    parser.add_argument("--max_seq_len", type=int, default=50,
                        help="We pad/truncate all inputs/targets to this length.")

    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--macro_batch_size", type=int, default=1)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--max_num", type=int, default=10,
                        help="We'll do a simple curriculum for arithmetic if task in [add,sub,mul,div].")

    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--memory_size", type=int, default=128)
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=1)

    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam","mezo"])
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epsilon", type=float, default=1e-3, help="MeZO eps.")
    parser.add_argument("--mezo", action="store_true")
    parser.add_argument("--mezo_layerwise", action="store_true",
                        help="Enable layerwise mezo using grouped params to reduce overhead.")
    parser.add_argument("--cosine_lr", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=0)

    parser.add_argument("--grad_clip", type=float, default=0.0, help="If >0, grad norm clipped.")
    parser.add_argument("--pad_bias", type=float, default=0.0, help="Initial logit bias for <PAD> in final layer.")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--wandb_proj", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args= parser.parse_args()

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
    if args.arch=="ntm":
        model= NTM(args.input_size, vocab_size, args.hidden_size,
                   args.memory_size, args.head_size, args.num_heads).to(device)
    elif args.arch=="dnc":
        model= DNC(args.input_size, vocab_size, args.hidden_size,
                   args.memory_size, args.head_size, args.num_heads).to(device)
    else:
        model= TransformerNTM(args.input_size, vocab_size, args.hidden_size).to(device)

    # add pad_bias if needed
    if args.pad_bias != 0.0:
        with torch.no_grad():
            if isinstance(model, NTM):
                model.fc_out.bias[0]+= args.pad_bias
            elif isinstance(model, DNC):
                model.fc_output.bias[0]+= args.pad_bias
            else:
                model.fc_out.bias[0]+= args.pad_bias

    # build optimizer
    params= list(model.parameters())+ list(embed.parameters())
    if args.optimizer=="adam":
        optimizer= optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        # mezo uses the same optimizer for momentum, we'll do param.grad => momentum => param.data
        optimizer= optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    scheduler= None
    if args.cosine_lr:
        scheduler= optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iters, eta_min=1e-6)

    
    criterion= nn.CrossEntropyLoss(ignore_index=0)
    global_step=0

    # track time
    train_start_time= time.time()

    # ++++++++ CURRICULUM LOGIC ++++++++
    # We'll do a naive approach:
    # For "copy" => start with context_length=2, each time we see train_acc>0.95 for a consecutive # of times => +1
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
                print(f"[CURRICULUM] copy: increasing context_length => {new_ct}")
                return new_ct, current_maxnum
            elif task in ["add","sub","mul","div"]:
                new_mn= current_maxnum+5
                print(f"[CURRICULUM] {task}: increasing max_num => {new_mn}")
                return current_context, new_mn
        return current_context, current_maxnum

    # to start the curriculum
    # if copy => start with context_length=2 (we ignore user param if they want, or we do min(2, user param) for train)
    # if add => start with max_num=5 if user param is bigger
    if args.task=="copy":
        current_context_len= min(args.context_length,2)
    else:
        current_context_len= args.context_length

    if args.task in ["add","sub","mul","div"]:
        current_max_num= min(args.max_num, 5)
    else:
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

    def train_micro_batch(x_emb, y_ids):
        """
        x_emb => [micro_batch_size, max_seq_len, embed_dim]
        y_ids => [micro_batch_size, max_seq_len]
        returns => float loss
        """
        if args.mezo:
            if args.mezo_layerwise:
                loss_val= mezo_char_layerwise(model, x_emb, y_ids, criterion, epsilon=args.epsilon)
            else:
                loss_val= mezo_char_single(model, x_emb, y_ids, criterion, epsilon=args.epsilon)
        else:
            model.train()
            optimizer.zero_grad()
            out, _, _= model(x_emb)
            B,L,V= out.size()
            loss= criterion(out.view(B*L, V), y_ids.view(B*L))
            loss.backward()
            loss_val= loss.item()
        return loss_val

    for iteration in range(1, args.max_iters+1):
        iter_start_time= time.time()

        # generate data
        # we do a curriculum for train
        x_strs, y_strs= generate_task_data(total_samples_per_iter, args.task,
                                           current_context_len, current_max_num,
                                           train=True)

        # zero grads
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        for p in embed.parameters():
            if p.grad is not None:
                p.grad.zero_()

        micro_loss_sum= 0.0

        # micro/macro approach
        for micro_i in range(args.macro_batch_size):
            start_idx= micro_i* args.micro_batch_size
            end_idx= start_idx+ args.micro_batch_size
            cur_x= x_strs[start_idx:end_idx]
            cur_y= y_strs[start_idx:end_idx]
            x_ids= str_to_tensor(cur_x, char_to_id, args.max_seq_len).to(device)
            y_ids= str_to_tensor(cur_y, char_to_id, args.max_seq_len).to(device)
            x_emb= embed(x_ids)
            loss_val= train_micro_batch(x_emb, y_ids)
            micro_loss_sum+= loss_val

        # do momentum-based step
        if not args.mezo:
            if args.grad_clip>0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        else:
            # mezo => we've set param.grad => let's do momentum
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(float(args.macro_batch_size))
            for p in embed.parameters():
                if p.grad is not None:
                    p.grad.div_(float(args.macro_batch_size))

            if args.grad_clip>0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        train_loss_val= micro_loss_sum/ args.macro_batch_size

        # warmup
        if scheduler is not None and iteration> args.warmup_steps:
            scheduler.step()

        # compute train accuracy on last micro-batch
        with torch.no_grad():
            x_ids= str_to_tensor(cur_x, char_to_id, args.max_seq_len).to(device)
            y_ids= str_to_tensor(cur_y, char_to_id, args.max_seq_len).to(device)
            x_emb= embed(x_ids)
            out, _, _= model(x_emb)
            B,L,V= out.size()
            preds= torch.argmax(out, dim=-1)
            mask= (y_ids!=0)
            correct= ((preds== y_ids)& mask).sum().item()
            total= mask.sum().item()
            train_acc= correct / total if total>0 else 1.0

        # possibly update curriculum
        if args.task in ["copy","add","sub","mul","div"]:
            new_ctx, new_mn= maybe_update_curriculum(train_acc, current_context_len, current_max_num, args.task)
            current_context_len= new_ctx
            current_max_num= new_mn

        iter_end_time= time.time()
        iteration_time= iter_end_time- iter_start_time
        total_elapsed= iter_end_time- train_start_time

        # log every iteration
        lr_current= optimizer.param_groups[0]["lr"]
        msg= (f"Iter={iteration}, train_loss={train_loss_val:.3f}, train_acc={train_acc:.3f}, "
              f"LR={lr_current:.6f}, iter_time={iteration_time:.2f}s, total_time={total_elapsed/60:.2f}m, "
              f"context_len={current_context_len}, max_num={current_max_num}")
        print(msg)
        sys.stdout.flush()

        # wandb
        if args.wandb_proj is not None:
            wandb.log({
                "train_loss": train_loss_val,
                "train_acc": train_acc,
                "lr": lr_current,
                "iter_time_s": iteration_time,
                "total_time_min": total_elapsed/60.0,
                "curr_context_len": current_context_len,
                "curr_max_num": current_max_num
            }, step=iteration)

        # validation every log_interval
        if iteration % args.log_interval == 0:
            with torch.no_grad():
                val_samples = total_samples_per_iter
                # for val, we do a bigger domain if not train
                # for copy => 5x
                # for add => domain = current_max_num+5
                # if you prefer a different approach, do so
                if args.task=="copy":
                    vx, vy= generate_copy_task_str(val_samples, current_context_len*5, train=False)
                elif args.task=="repeat_copy":
                    vx, vy= generate_repeat_copy_task_str(val_samples, current_context_len*5, train=False)
                elif args.task=="associative_recall":
                    vx, vy= generate_associative_recall_task_str(val_samples, item_len=3,
                                                                 num_items=current_context_len*5, train=False)
                elif args.task in ["add","sub","mul","div"]:
                    vx, vy= generate_arithmetic_task_str(val_samples, context_length=current_context_len,
                                                         task_type=args.task,
                                                         max_num=current_max_num+5, train=False)
                elif args.task=="fib":
                    vx, vy= generate_fibonacci_task_str(val_samples, current_context_len,
                                                        max_n=current_max_num+5, train=False)
                elif args.task=="factorial":
                    vx, vy= generate_factorial_task_str(val_samples, current_context_len,
                                                        max_n=current_max_num+5, train=False)
                else:
                    raise ValueError(f"Unknown task: {args.task} for val")

                vx_ids= str_to_tensor(vx, char_to_id, args.max_seq_len).to(device)
                vy_ids= str_to_tensor(vy, char_to_id, args.max_seq_len).to(device)
                vx_emb= embed(vx_ids)
                model.eval()
                val_out, _, _= model(vx_emb)
                B2,L2,V2= val_out.size()
                val_loss= criterion(val_out.view(B2*L2,V2), vy_ids.view(B2*L2))

                val_preds= torch.argmax(val_out, dim=-1)
                mask= (vy_ids!=0)
                val_correct= ((val_preds== vy_ids)& mask).sum().item()
                val_total= mask.sum().item()
                val_acc= val_correct/ val_total if val_total>0 else 1.0

                sample_indices= random.sample(range(B2), min(3,B2))
                print("\n[DEBUG] Random Val Samples:")
                for idx in sample_indices:
                    input_str= vx[idx]
                    preds_i= val_preds[idx].cpu().tolist()

                    pred_str_chars= []
                    for token_id in preds_i:
                        if token_id==0:
                            continue
                        if token_id==2:
                            pred_str_chars.append("<eos>")
                            break
                        pred_str_chars.append(id_to_char.get(token_id,'?'))
                    pred_str= "".join(pred_str_chars)

                    # pad version
                    pad_pred_chars= []
                    for token_id in preds_i:
                        if token_id==0:
                            pad_pred_chars.append("[PAD]")
                        elif token_id==2:
                            pad_pred_chars.append("<eos>")
                            break
                        else:
                            pad_pred_chars.append(id_to_char.get(token_id,'?'))
                    pad_pred_str= "".join(pad_pred_chars)

                    target_str= vy[idx]
                    print(f"  [Val idx={idx}]")
                    print(f"    Input:  '{input_str}'")
                    print(f"    Target: '{target_str}'")
                    print(f"    Pred:   '{pred_str}'")
                    print(f"    Pred with padding: '{pad_pred_str}'")
                print("[END DEBUG]\n")

            msg_val= (f"[VAL] Iter={iteration}, val_loss={val_loss.item():.3f}, val_acc={val_acc:.3f}")
            print(msg_val)
            sys.stdout.flush()

            if args.wandb_proj is not None:
                wandb.log({
                    "val_loss": val_loss.item(),
                    "val_acc": val_acc
                }, step=iteration)

    print("Finished.")


if __name__=="__main__":
    main()
