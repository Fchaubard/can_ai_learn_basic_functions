#!/usr/bin/env python3
"""
Neural Turing Machine / DNC / Transformer, all using ASCII char-based tasks
(copy, repeat_copy, associative_recall, add, sub, mul, div, fib, factorial).
We keep your warmup & cosine LR logic, wandb logging, mezo steps, etc.
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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb

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
# ASCII Vocab + Tokenizer
##############################################################################
def get_char_vocab():
    """
    We'll define a small ASCII set:
    index 0 => <PAD>
    digits 0..9
    uppercase A..Z
    operators + - * / =
    space ' '
    marker '|'
    """
    special = ['+', '-', '*', '/', '=', ' ', '|']
    digits = list(string.digits)        # '0'..'9'
    letters = list(string.ascii_uppercase)
    vocab_list = ['<PAD>'] + digits + letters + special

    char_to_id = {}
    id_to_char = {}
    for i, ch in enumerate(vocab_list):
        char_to_id[ch] = i
        id_to_char[i] = ch
    return vocab_list, char_to_id, id_to_char

def str_to_tensor(batch_strs, char_to_id):
    """
    Convert list of strings -> [B, L] LongTensor, pad with 0 => <PAD>.
    """
    max_len = max(len(s) for s in batch_strs)
    B = len(batch_strs)
    out = torch.zeros(B, max_len, dtype=torch.long)
    for i, s in enumerate(batch_strs):
        for j, ch in enumerate(s):
            out[i,j] = char_to_id.get(ch, 0)
    return out

##############################################################################
# String-based Task Generators
##############################################################################
def generate_copy_task_str(batch_size, seq_len, train=True):
    import random
    import string
    if not train:
        seq_len *= 5
    letters = string.ascii_uppercase
    x_list, y_list = [], []
    for _ in range(batch_size):
        data_str = "".join(random.choice(letters) for _ in range(seq_len))
        xfull = data_str + '|' + (' ' * seq_len)
        yfull = (' ' * (seq_len+1)) + data_str
        x_list.append(xfull)
        y_list.append(yfull)
    return x_list, y_list

def generate_repeat_copy_task_str(batch_size, seq_len, repeat_min=1, repeat_max=3, train=True):
    import random
    if not train:
        seq_len*=5
    letters= string.ascii_uppercase
    x_list,y_list=[],[]
    for _ in range(batch_size):
        data_str= "".join(random.choice(letters) for __ in range(seq_len))
        c_val= random.randint(repeat_min, repeat_max)
        c_ch= str(c_val)
        pad_len= seq_len* repeat_max
        xfull= data_str+ c_ch+ '|'+ (' '*pad_len)
        front_len= len(data_str)+1+1
        repeated= data_str*c_val
        leftover= (repeat_max-c_val)*seq_len
        yfull= (' '* front_len)+ repeated+ (' '* leftover)
        x_list.append(xfull)
        y_list.append(yfull)
    return x_list,y_list

def generate_associative_recall_task_str(batch_size, item_len=3, num_items=3, train=True):
    import random
    import string
    if not train:
        num_items*=5
    letters= string.ascii_uppercase
    x_list,y_list=[],[]
    for _ in range(batch_size):
        items= ["".join(random.choice(letters) for __ in range(item_len)) for __ in range(num_items)]
        q_idx= random.randint(0, num_items-2)
        query_item= items[q_idx]
        ans_item= items[q_idx+1]
        flat_items= "".join(items)
        xfull= flat_items+ '|'+ query_item
        xfull+= ' '* item_len
        front_len= len(flat_items)+1+ len(query_item)
        yfull= (' '* front_len)+ ans_item
        x_list.append(xfull)
        y_list.append(yfull)
    return x_list,y_list

def generate_arithmetic_task_str(batch_size, task_type="add", max_num=100, train=True):
    import random
    if train:
        lo, hi=1, max_num
    else:
        lo, hi= max_num+1, max_num+10
    x_list,y_list=[],[]
    for _ in range(batch_size):
        a= random.randint(lo,hi)
        b= random.randint(lo,hi)
        if task_type=='sub':
            if b>a:
                a,b= b,a
            res= a-b
            op='-'
        elif task_type=='mul':
            res= a*b
            op='*'
        elif task_type=='div':
            if b>a:
                a,b=b,a
            if b==0:
                b=1
            res= a//b
            op='/'
        else:  # add
            res= a+b
            op='+'
        expr_in= f"{a}{op}{b}="
        out_str= f"{res}"
        L= max(len(expr_in), len(out_str))
        xfull= expr_in + (' '*(L-len(expr_in)))
        yfull= out_str + (' '*(L-len(out_str)))
        x_list.append(xfull)
        y_list.append(yfull)
    return x_list,y_list

def generate_fibonacci_task_str(batch_size, max_n=10, train=True):
    import random
    if train:
        lo, hi=1, max_n
    else:
        lo, hi= max_n+1, max_n+10
    fib_cache=[0,1]
    for i in range(2, hi+11):
        fib_cache.append(fib_cache[-1]+ fib_cache[-2])
    x_list,y_list=[],[]
    for _ in range(batch_size):
        n_val= random.randint(lo,hi)
        fib_n= fib_cache[n_val]
        expr_in= f"{n_val}="
        out_str= f"{fib_n}"
        L= max(len(expr_in), len(out_str))
        xfull= expr_in+ (' '*(L-len(expr_in)))
        yfull= out_str+ (' '*(L-len(out_str)))
        x_list.append(xfull)
        y_list.append(yfull)
    return x_list,y_list

def generate_factorial_task_str(batch_size, max_n=6, train=True):
    import random
    if train:
        lo, hi=1, max_n
    else:
        lo, hi= max_n+1, max_n+10
    fact_cache=[1]
    for i in range(1, hi+11):
        fact_cache.append(fact_cache[-1]* i)
    x_list,y_list=[],[]
    for _ in range(batch_size):
        n_val= random.randint(lo,hi)
        fact_n= fact_cache[n_val]
        expr_in= f"{n_val}="
        out_str= f"{fact_n}"
        L= max(len(expr_in), len(out_str))
        xfull= expr_in+ (' '*(L-len(expr_in)))
        yfull= out_str+ (' '*(L-len(out_str)))
        x_list.append(xfull)
        y_list.append(yfull)
    return x_list,y_list


##############################################################################
# NTM / DNC / Transformer
##############################################################################
class NTM(nn.Module):
    """
    We'll interpret input_size as the embedding dimension for char tokens.
    The final output => vocab_size for cross-entropy.
    """
    def __init__(self, input_size, output_size, hidden_size, memory_size, head_size, num_heads):
        super(NTM, self).__init__()
        self.hidden_size= hidden_size
        self.num_heads= num_heads
        self.memory_size= memory_size
        self.head_size= head_size
        
        self.controller= nn.LSTM(input_size + num_heads* head_size, hidden_size, batch_first=True)
        self.fc_head= nn.Linear(hidden_size, num_heads* (head_size+ memory_size+ 3))
        self.fc_out= nn.Linear(hidden_size+ num_heads* head_size, output_size)

    def _addressing(self, memory, head_params, prev_weights):
        batch_size, mem_size, _= memory.size()
        read_keys= head_params[:,:, :self.head_size]
        read_weights= torch.einsum('bnk,bmk->bnm', read_keys, memory)/(self.head_size**0.5)
        read_weights= F.softmax(read_weights, dim=-1)
        read_content= torch.einsum('bnm,bmh->bnh', read_weights, memory)

        write_keys= head_params[:,:, self.head_size: 2*self.head_size]
        write_strength= torch.sigmoid(head_params[:,:,2*self.head_size:2*self.head_size+1])

        write_weights= torch.einsum('bnk,bmk->bnm', write_keys, memory)/(self.head_size**0.5)
        write_weights= F.softmax(write_weights, dim=-1)
        write_content= write_strength* write_keys
        delta= torch.einsum('bnm,bnh->bmh', write_weights, write_content)
        memory= memory+ delta
        return memory, read_content, read_weights, write_weights

    def forward(self, x_emb, hidden=None, memory=None):
        """
        x_emb => [B,L, embed_dim]
        """
        batch_size, seq_len, embed_dim= x_emb.size()
        if hidden is None:
            h0= x_emb.new_zeros(1,batch_size,self.hidden_size)
            c0= x_emb.new_zeros(1,batch_size,self.hidden_size)
            hidden= (h0,c0)
        if memory is None:
            memory= x_emb.new_zeros(batch_size, self.memory_size, self.head_size)
        
        outputs= []
        read_contents= x_emb.new_zeros(batch_size, self.num_heads, self.head_size)
        for t in range(seq_len):
            inp_t= torch.cat([x_emb[:,t,:], read_contents.view(batch_size,-1)], dim=-1).unsqueeze(1)
            out_ctrl, hidden= self.controller(inp_t, hidden)
            h= out_ctrl.squeeze(1)
            head_params= self.fc_head(h).view(batch_size, self.num_heads, self.head_size+ self.memory_size+ 3)
            memory, read_contents, _,_= self._addressing(memory, head_params, None)
            out= torch.cat([h, read_contents.view(batch_size,-1)], dim=-1)
            out= self.fc_out(out)
            outputs.append(out.unsqueeze(1))
        outputs= torch.cat(outputs, dim=1)  # [B,L, output_size]
        return outputs, memory, hidden


class DNC(nn.Module):
    """
    input_size => embed_dim for chars
    output_size => vocab_size
    """
    def __init__(self, input_size, output_size, hidden_size, memory_size, head_size, num_heads=1):
        super(DNC,self).__init__()
        assert num_heads==1
        self.input_size= input_size
        self.output_size= output_size
        self.hidden_size= hidden_size
        self.memory_size= memory_size
        self.head_size= head_size
        self.num_reads= num_heads

        self.controller= nn.LSTM(input_size+ (self.num_reads*self.head_size), hidden_size, batch_first=True)
        self.interface_size= ( self.head_size*2 + 2 + self.head_size+1 +3 )
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
        batch_size, seq_len, embed_dim= x_emb.size()
        device= x_emb.device
        self._init_memory_if_needed(batch_size, device)
        if hidden is None:
            h0= x_emb.new_zeros(1,batch_size,self.hidden_size)
            c0= x_emb.new_zeros(1,batch_size,self.hidden_size)
            hidden= (h0,c0)

        read_vec= x_emb.new_zeros(batch_size, self.head_size)
        outs=[]
        for t in range(seq_len):
            inp_t= torch.cat([x_emb[:,t,:], read_vec], dim=-1).unsqueeze(1)
            out_ctrl, hidden= self.controller(inp_t, hidden)
            h= out_ctrl.squeeze(1)
            interface= self.fc_interface(h)
            # parse
            offset=0
            erase_vec= torch.sigmoid(interface[..., offset: offset+self.head_size])
            offset+= self.head_size
            write_vec= interface[..., offset: offset+self.head_size]
            offset+= self.head_size
            write_gate= torch.sigmoid(interface[..., offset: offset+1]).squeeze(-1)
            offset+=1
            alloc_gate= torch.sigmoid(interface[..., offset: offset+1]).squeeze(-1)
            offset+=1

            # allocation
            alloc_w= self._get_allocation_weights()
            w_gate= write_gate.unsqueeze(-1)
            a_gate= alloc_gate.unsqueeze(-1)
            write_w= w_gate*a_gate* alloc_w
            self.write_weights= write_w

            # erase
            erase_mat= erase_vec.unsqueeze(1)
            self.memory= self.memory*(1- torch.bmm(write_w.unsqueeze(-1), erase_mat))
            # add
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
            read_w= read_w/ read_w.sum(dim=-1,keepdim=True)
            self.read_weights= read_w
            read_vec= torch.bmm(read_w.unsqueeze(1), self.memory).squeeze(1)

            out= torch.cat([h, read_vec], dim=-1)
            out= self.fc_output(out)
            outs.append(out.unsqueeze(1))

        outs= torch.cat(outs, dim=1)
        # detach memory
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
        cprod= F.pad(cprod[:,:-1], (1,0), value=1.0)
        alloc_in_order= (1- usage_sorted)* cprod
        alloc_w.scatter_(1, idx_sorted, alloc_in_order)
        return alloc_w

    def _update_temporal_link(self, write_w):
        ww_ij= write_w.unsqueeze(-1)+ write_w.unsqueeze(1)
        self.link= (1- ww_ij)* self.link
        self.link+= torch.bmm(self.precedence.unsqueeze(-1), write_w.unsqueeze(1))
        diag= torch.eye(self.memory_size, device= write_w.device).unsqueeze(0)
        self.link= self.link* (1- diag)
        self.precedence= (1- write_w.sum(dim=-1,keepdim=True))* self.precedence
        self.precedence= self.precedence+ write_w

    def _content_addressing(self, key, strength):
        dot= torch.einsum("bkw,bnw->bn", key.unsqueeze(1), self.memory)
        key_norm= torch.norm(key,2,dim=-1,keepdim=True)+1e-8
        mem_norm= torch.norm(self.memory,2,dim=-1)+ 1e-8
        dot= dot/(key_norm* mem_norm)
        dot= dot* strength.unsqueeze(-1)
        return F.softmax(dot, dim=-1)


class TransformerController(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerController,self).__init__()
        encoder_layer= nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder= nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x):
        return self.encoder(x)

class TransformerNTM(nn.Module):
    """
    We'll interpret input_size as the embed dim => pass x_emb => [B,L,input_size]
    """
    def __init__(self, input_size, output_size, hidden_size):
        super(TransformerNTM, self).__init__()
        # We'll just do a Transformer with d_model=input_size
        self.transformer= TransformerController(d_model=input_size, nhead=4, num_layers=2, dim_feedforward=4*input_size)
        self.fc_out= nn.Linear(input_size, output_size)

    def forward(self, x_emb, hidden=None, memory=None):
        trans_out= self.transformer(x_emb) # [B,L,input_size]
        out= self.fc_out(trans_out)       # => [B,L, output_size]
        return out, None, None

##############################################################################
# MeZO single / layerwise for char-based
##############################################################################
def mezo_char_single(model, x, y, criterion, epsilon=1e-3):
    all_params= list(model.parameters())
    for p in all_params:
        if p.grad is not None:
            p.grad.zero_()
    orig= [p.data.clone() for p in all_params]
    directions= [torch.randn_like(p) if p.requires_grad else None for p in all_params]

    with torch.no_grad():
        # +eps
        for p,d in zip(all_params, directions):
            if p.requires_grad and d is not None:
                p.data.add_(epsilon*d.sign())
        out_p,_,_= model(x) if isinstance(model,(NTM,DNC)) else model(x)
        B,L,V= out_p.size()
        loss_plus= criterion(out_p.view(B*L,V), y.view(B*L))

        # -2eps
        for p,d in zip(all_params,directions):
            if p.requires_grad and d is not None:
                p.data.sub_(2*epsilon*d.sign())
        out_m,_,_= model(x) if isinstance(model,(NTM,DNC)) else model(x)
        loss_minus= criterion(out_m.view(B*L,V), y.view(B*L))

        for p, od in zip(all_params, orig):
            p.data.copy_(od)

    grad_est= (loss_plus- loss_minus)/(2*epsilon)
    for p,d in zip(all_params,directions):
        if p.requires_grad and d is not None:
            if p.grad is None:
                p.grad= grad_est* d.sign()
            else:
                p.grad.add_( grad_est* d.sign())
    return 0.5*(loss_plus.item()+ loss_minus.item())

def mezo_char_layerwise(model, x, y, criterion, epsilon=1e-3):
    all_params= list(model.parameters())
    for p in all_params:
        if p.grad is not None:
            p.grad.zero_()
    total_loss= 0.0
    param_count=0
    with torch.no_grad():
        for param in all_params:
            if not param.requires_grad:
                continue
            param_count+=1
            orig= param.data.clone()
            direction= torch.randn_like(param)
            param.data.add_(epsilon* direction.sign())
            out_p,_,_= model(x) if isinstance(model,(NTM,DNC)) else model(x)
            B,L,V= out_p.size()
            loss_plus= criterion(out_p.view(B*L,V), y.view(B*L))

            param.data.copy_(orig- 2*epsilon* direction.sign())
            out_m,_,_= model(x) if isinstance(model,(NTM,DNC)) else model(x)
            loss_minus= criterion(out_m.view(B*L,V), y.view(B*L))

            grad_est= (loss_plus- loss_minus)/(2*epsilon)
            param.data.copy_(orig)
            if param.grad is None:
                param.grad= grad_est* direction.sign()
            else:
                param.grad.add_( grad_est* direction.sign())

            total_loss+= 0.5*(loss_plus.item()+ loss_minus.item())
    if param_count>0:
        total_loss/= param_count
    return total_loss

##############################################################################
# Main
##############################################################################
def main():
    parser= argparse.ArgumentParser(description="NTM/DNC/Transformer with ASCII char-based tasks, warmup, Cosine LR, MeZO, wandb.")
    parser.add_argument("--arch", type=str, default="ntm",
                        choices=["ntm","dnc","tra"],
                        help="Which architecture to use: NTM, DNC, or Transformer-based model.")
    parser.add_argument("--task", type=str, default="copy",
                        choices=["copy","repeat_copy","associative_recall","add","sub","mul","div","fib","factorial"])
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--max_num", type=int, default=100)

    parser.add_argument("--input_size", type=int, default=32,
                        help="Embedding dimension for characters (the NTM/DNC/Transformer sees [B,L,input_size]).")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--memory_size", type=int, default=128)
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=1)

    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam","mezo"])
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epsilon", type=float, default=1e-3,
                        help="MeZO perturbation.")
    parser.add_argument("--mezo", action="store_true",
                        help="If true, use mezo gradient instead of backprop.")
    parser.add_argument("--mezo_layerwise", action="store_true",
                        help="Layerwise mezo or single-step mezo.")
    parser.add_argument("--cosine_lr", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=0)

    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--wandb_proj", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args= parser.parse_args()

    if torch.cuda.is_available():
        gpu_index= pick_gpu_with_most_free_mem()
        device= torch.device(f"cuda:{gpu_index}")
        print(f"[INFO] Using GPU: {gpu_index}")
    else:
        device= torch.device("cpu")
        print("[INFO] Using CPU")

    # Build the vocab
    vocab_list, char_to_id, id_to_char= get_char_vocab()
    vocab_size= len(vocab_list)

    # We'll do an embedding => [B,L] -> [B,L,input_size]
    embed= nn.Embedding(vocab_size, args.input_size, padding_idx=0).to(device)

    # Build architecture
    if args.arch=="ntm":
        model= NTM(input_size=args.input_size,
                   output_size=vocab_size,
                   hidden_size=args.hidden_size,
                   memory_size=args.memory_size,
                   head_size=args.head_size,
                   num_heads=args.num_heads).to(device)
    elif args.arch=="dnc":
        model= DNC(input_size=args.input_size,
                   output_size=vocab_size,
                   hidden_size=args.hidden_size,
                   memory_size=args.memory_size,
                   head_size=args.head_size,
                   num_heads=args.num_heads).to(device)
    else: # "tra"
        model= TransformerNTM(input_size=args.input_size,
                              output_size=vocab_size,
                              hidden_size=args.hidden_size).to(device)

    # Build optimizer
    # We must optimize both model + embedding
    params= list(model.parameters())+ list(embed.parameters())
    if args.optimizer=="adam":
        optimizer= optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer= optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    # Possibly define a Cosine LR
    scheduler= None
    if args.cosine_lr:
        scheduler= optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=args.max_iters,
                                                        eta_min=1e-6)

    # W&B init
    if args.wandb_proj is not None:
        wandb.init(project=args.wandb_proj, name=args.wandb_run_name)
        wandb.config.update(args)

    criterion= nn.CrossEntropyLoss(ignore_index=0)
    global_step=0

    for iteration in range(1, args.max_iters+1):
        # generate train data
        if args.task=="copy":
            x_strs, y_strs= generate_copy_task_str(args.batch_size, args.seq_len, train=True)
        elif args.task=="repeat_copy":
            x_strs, y_strs= generate_repeat_copy_task_str(args.batch_size, args.seq_len, train=True)
        elif args.task=="associative_recall":
            x_strs, y_strs= generate_associative_recall_task_str(args.batch_size, item_len=3, num_items=args.seq_len, train=True)
        elif args.task in ["add","sub","mul","div"]:
            x_strs, y_strs= generate_arithmetic_task_str(args.batch_size, task_type=args.task, max_num=args.max_num, train=True)
        elif args.task=="fib":
            x_strs, y_strs= generate_fibonacci_task_str(args.batch_size, max_n=args.max_num, train=True)
        elif args.task=="factorial":
            x_strs, y_strs= generate_factorial_task_str(args.batch_size, max_n=args.max_num, train=True)
        else:
            raise ValueError(f"Unknown task {args.task}.")

        x_ids= str_to_tensor(x_strs, char_to_id).to(device)   # [B,L]
        y_ids= str_to_tensor(y_strs, char_to_id).to(device)   # [B,L]
        x_emb= embed(x_ids)  # => [B,L,input_size]

        # Warmup
        if args.optimizer=="adam" and args.warmup_steps>0 and iteration<= args.warmup_steps:
            frac= iteration/ float(args.warmup_steps)
            new_lr= args.learning_rate* frac
            for pg in optimizer.param_groups:
                pg["lr"]= new_lr



        if args.mezo:
            if args.mezo_layerwise:
                loss_val= mezo_char_layerwise(model, x_emb, y_ids, criterion, epsilon=args.epsilon)
            else:
                loss_val= mezo_char_single(model, x_emb, y_ids, criterion, epsilon=args.epsilon)
            
            # apply optimizer step
            optimizer.step()
        
            # after mezo, do a forward pass to get `out` for train accuracy
            model.eval()
            with torch.no_grad():
                if args.arch in ["ntm","dnc"]:
                    out, _, _ = model(x_emb)
                else:
                    out = model(x_emb)  # for TransformerNTM
                    out = out[0] 
                    # if it returns (out,_,_) then adapt accordingly
            out = out.detach()
            
        else:
            # standard backprop
            model.train()
            optimizer.zero_grad()
            out,_,_= model(x_emb) if args.arch in ["ntm","dnc"] else model(x_emb)
            B,L,V= out.size()
            loss= criterion(out.view(B*L,V), y_ids.view(B*L))
            loss.backward()
            optimizer.step()
            loss_val= loss.item()

        # step scheduler if after warmup
        if scheduler is not None and iteration> args.warmup_steps:
            scheduler.step()

        global_step+=1
        ##############################################################################
        # UPDATED VALIDATION/LOGGING SECTION
        ##############################################################################
        if (global_step % args.log_interval) == 0:
            # --------------------------------------------------------
            # 1) Compute train accuracy on the current training batch
            #    (We assume "out" is the last forward pass on the train batch,
            #     and y_ids is the ground-truth token IDs.)
            # --------------------------------------------------------
            # out shape: [B, L, vocab_size]
            # y_ids shape: [B, L]
            with torch.no_grad():
                train_preds = torch.argmax(out, dim=-1)  # [B, L]
                # We'll ignore <PAD>=0
                train_mask = (y_ids != 0)  # boolean
                train_correct = ((train_preds == y_ids) & train_mask).sum().item()
                train_total = train_mask.sum().item()
                train_acc = (train_correct / train_total) if train_total > 0 else 1.0
        
            # --------------------------------------------------------
            # 2) Build a small validation batch
            # --------------------------------------------------------
            with torch.no_grad():
                if args.task == "copy":
                    vx_strs, vy_strs = generate_copy_task_str(args.batch_size, args.seq_len, train=False)
                elif args.task == "repeat_copy":
                    vx_strs, vy_strs = generate_repeat_copy_task_str(args.batch_size, args.seq_len, train=False)
                elif args.task == "associative_recall":
                    vx_strs, vy_strs = generate_associative_recall_task_str(args.batch_size, item_len=3, num_items=args.seq_len, train=False)
                elif args.task in ["add", "sub", "mul", "div"]:
                    vx_strs, vy_strs = generate_arithmetic_task_str(args.batch_size, task_type=args.task, max_num=args.max_num, train=False)
                elif args.task == "fib":
                    vx_strs, vy_strs = generate_fibonacci_task_str(args.batch_size, max_n=args.max_num, train=False)
                elif args.task == "factorial":
                    vx_strs, vy_strs = generate_factorial_task_str(args.batch_size, max_n=args.max_num, train=False)
                else:
                    raise ValueError(f"Unknown task {args.task} for validation")
        
                vx_ids = str_to_tensor(vx_strs, char_to_id).to(device)  # [B, L]
                vy_ids = str_to_tensor(vy_strs, char_to_id).to(device)  # [B, L]
                vx_emb = embed(vx_ids)                                  # [B, L, embed_dim]
        
                # forward pass for validation
                model.eval()
                val_out, _, _ = model(vx_emb) if args.arch in ["ntm", "dnc"] else model(vx_emb)
                B, L, V = val_out.size()
                val_loss = criterion(val_out.view(B * L, V), vy_ids.view(B * L))
        
                # 3) Compute validation accuracy
                val_preds = torch.argmax(val_out, dim=-1)  # [B, L]
                val_mask = (vy_ids != 0)
                val_correct = ((val_preds == vy_ids) & val_mask).sum().item()
                val_total = val_mask.sum().item()
                val_acc = (val_correct / val_total) if val_total > 0 else 1.0
        
                # 4) Randomly print a few examples from val (input/prediction/target)
                sample_indices = random.sample(range(B), min(3, B))  # up to 3 samples
                print("\n[DEBUG] Random Val Samples:")
                for idx in sample_indices:
                    # decode input, predicted, target strings
                    input_str = vx_strs[idx]
                    # convert predicted IDs -> string
                    pred_str = []
                    for token_id in val_preds[idx].cpu().tolist():
                        if token_id == 0:
                            break  # <PAD>
                        pred_str.append(id_to_char.get(token_id, '?'))
                    pred_str = "".join(pred_str)
        
                    target_str = vy_strs[idx]
                    print(f"  [Val idx={idx}]")
                    print(f"    Input:  '{input_str}'")
                    print(f"    Target: '{target_str}'")
                    print(f"    Pred:   '{pred_str}'")
                print("[END DEBUG]\n")
        
            # --------------------------------------------------------
            # 5) Print/log final message with train/val stats
            # --------------------------------------------------------
            lr_current = optimizer.param_groups[0]["lr"]
            msg = (f"Iter={global_step}, "
                   f"train_loss={loss_val:.3f}, train_acc={train_acc:.3f}, "
                   f"val_loss={val_loss.item():.3f}, val_acc={val_acc:.3f}, "
                   f"LR={lr_current:.6f}")
            print(msg)
            sys.stdout.flush()
        
            # 6) W&B logging if desired
            if args.wandb_proj is not None:
                wandb.log({
                    "train_loss": loss_val,
                    "train_acc": train_acc,
                    "val_loss": val_loss.item(),
                    "val_acc": val_acc,
                    "lr": lr_current
                }, step=global_step)
    
    print("Finished training!")


if __name__=="__main__":
    main()
