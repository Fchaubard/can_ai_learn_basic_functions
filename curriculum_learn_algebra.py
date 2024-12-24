
# curriculum_learn_algebra
import argparse
import torch
import wandb
import random
import torch.nn.functional as F
import numpy as np
import os
import math
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

from torch.utils.data import DataLoader, Dataset

os.environ["WANDB_API_KEY"] = ""

# --------------------------------------------------------------------------
#  Filter gradients function for GAF
# --------------------------------------------------------------------------
def filter_gradients(G1, G2, cos_distance_thresh=1):
    G1_flat = torch.cat([g.view(-1) for g in G1])
    G2_flat = torch.cat([g.view(-1) for g in G2])
    cos_sim = F.cosine_similarity(G1_flat, G2_flat, dim=0)
    cos_distance = 1 - cos_sim
    if cos_distance > cos_distance_thresh:
        return None, cos_distance.item()
    return [(g1 + g2) / 2 for g1, g2 in zip(G1, G2)], cos_distance.item()

# --------------------------------------------------------------------------
#  Utility: float comparison with up to 2 decimal places
# --------------------------------------------------------------------------
def floats_are_close(val_pred, val_gt, decimals=2):
    """
    Returns True if val_pred is accurate within 'decimals' decimal places of val_gt.
    """
    # If either is NaN or not finite, return False
    if (not math.isfinite(val_pred)) or (not math.isfinite(val_gt)):
        return False
    
    # Compare within 10^(-decimals)
    return abs(val_pred - val_gt) < 0.5 * 10**(-decimals)

# --------------------------------------------------------------------------
#  Utility: convert a float to scientific notation if over 5 sig figs
# --------------------------------------------------------------------------
def convert_to_sci_if_needed(val_str):
    """
    If val_str is numeric and length is >5 sig figs (including decimal point),
    convert to scientific notation with the same number of sig figs.
    """
    # Attempt to parse
    try:
        val_float = float(val_str)
    except:
        return val_str  # if can't parse, return as is

    # Convert to string in normal form
    normal_str = f"{val_float:.10g}"  # limit to 10 digits total
    # Check if we want to switch to scientific
    # '10g' can automatically produce scientific if needed, but let's do an explicit check:
    # Count significant digits in normal_str ignoring minus sign and decimal point
    digits = [c for c in normal_str if c.isdigit()]
    if len(digits) > 5:
        # Force scientific with 5 sig figs
        sci_str = f"{val_float:.5g}"
        return sci_str
    else:
        return normal_str

# --------------------------------------------------------------------------
#  Compute per-sample accuracy (with up to 2 decimal correctness for decimals)
#  Also track separate accuracy by task type.
# --------------------------------------------------------------------------
def compute_per_sample_accuracy(input_ids, labels, logits, tokenizer, tasks_tracked):
    """
    tasks_tracked is a dict to accumulate correctness for each task type, e.g.:
       {
         "add": [correct_count, total_count],
         "subtract": [...],
         "multiply": [...],
         "divide": [...],
         "factorial": [...],
         "fibonacci": [...],
         "algebra": [...]
       }

    Returns:
        overall_acc (float),
        incorrect_samples (list of dicts),
        tasks_tracked (updated).
    """
    batch_size = input_ids.size(0)
    correct = 0
    total = 0
    incorrect_samples = []

    # Convert special tokens to ID
    equal_token_id = tokenizer.convert_tokens_to_ids('=')

    for i in range(batch_size):
        input_id = input_ids[i]
        label = labels[i]
        logit = logits[i]

        # Identify the task from the input (heuristic based on presence of certain tokens)
        # We'll do a simple parse of the string representation:
        input_text_full = tokenizer.decode(input_id, skip_special_tokens=True)
        # Decide the task type
        # (You could do something more elaborate, but let's keep it simple)
        if "!" in input_text_full:
            task_type = "factorial"
        elif "fib" in input_text_full.lower():
            task_type = "fibonacci"
        elif "x" in input_text_full.lower() and ("=" in input_text_full):
            task_type = "algebra"
        elif "+" in input_text_full:
            task_type = "add"
        elif "-" in input_text_full:
            task_type = "subtract"
        elif "*" in input_text_full:
            task_type = "multiply"
        elif "/" in input_text_full:
            task_type = "divide"
        else:
            # fallback
            task_type = "unknown"

        if task_type not in tasks_tracked:
            tasks_tracked[task_type] = [0,0]

        # Find the position of '=' token
        equal_pos = (input_id == equal_token_id).nonzero(as_tuple=True)

        if len(equal_pos[0]) > 0:
            equal_pos = equal_pos[0].item()
            # Ground truth tokens after '='
            ground_truth_tokens = label[equal_pos + 1:]
            ground_truth_tokens = ground_truth_tokens[ground_truth_tokens != -100]

            # Predicted tokens after '='
            predicted_logits = logit[equal_pos:]
            predicted_tokens = predicted_logits.argmax(dim=-1)
            predicted_tokens = predicted_tokens[:len(ground_truth_tokens)]

            ground_truth_text = tokenizer.decode(ground_truth_tokens, skip_special_tokens=True).strip()
            predicted_text = tokenizer.decode(predicted_tokens, skip_special_tokens=True).strip()

            # Remove spaces
            ground_truth_text = ground_truth_text.replace(' ', '')
            predicted_text = predicted_text.replace(' ', '')

            # If we allowed "thinking tokens" `_` in generation, strip them out:
            ground_truth_text = ground_truth_text.replace('_','')
            predicted_text = predicted_text.replace('_','')

            # Attempt to parse as float
            # But for factorial/fibonacci (often integer), or algebra solutions, we still do float-based compare
            # with a 2-decimal tolerance
            try:
                ground_truth_val = float(ground_truth_text)
                predicted_val = float(predicted_text)
                # We consider it correct if they're close within 2 decimals
                if floats_are_close(predicted_val, ground_truth_val, decimals=2):
                    tasks_tracked[task_type][0] += 1
                    correct += 1
                else:
                    incorrect_samples.append({
                        'input': input_text_full,
                        'prediction': predicted_text,
                        'ground_truth': ground_truth_text,
                    })
                tasks_tracked[task_type][1] += 1
                total += 1
            except:
                # Could not parse, mark as incorrect
                tasks_tracked[task_type][1] += 1
                total += 1
                incorrect_samples.append({
                    'input': input_text_full,
                    'prediction': predicted_text,
                    'ground_truth': ground_truth_text,
                })

    overall_acc = correct / total if total > 0 else 0.0
    return overall_acc, incorrect_samples, tasks_tracked

# --------------------------------------------------------------------------
#  Utility: compute MSE on parsed answers after '=' (still might be helpful)
# --------------------------------------------------------------------------
def compute_mse_on_parsed_answers(input_ids, labels, logits, tokenizer):
    batch_size = input_ids.size(0)
    total_mse = 0.0
    count = 0
    equal_token_id = tokenizer.convert_tokens_to_ids('=')
    for i in range(batch_size):
        input_id = input_ids[i]
        label = labels[i]
        logit = logits[i]
        equal_pos = (input_id == equal_token_id).nonzero(as_tuple=True)

        if len(equal_pos[0]) > 0:
            equal_pos = equal_pos[0].item()
            ground_truth_tokens = label[equal_pos + 1:]
            ground_truth_tokens = ground_truth_tokens[ground_truth_tokens != -100]
            predicted_logits = logit[equal_pos:]
            predicted_tokens = predicted_logits.argmax(dim=-1)
            predicted_tokens = predicted_tokens[:len(ground_truth_tokens)]

            ground_truth_text = tokenizer.decode(ground_truth_tokens, skip_special_tokens=True).strip()
            predicted_text = tokenizer.decode(predicted_tokens, skip_special_tokens=True).strip()

            # Remove spaces and underscores
            ground_truth_text = ground_truth_text.replace(' ', '').replace('_','')
            predicted_text = predicted_text.replace(' ', '').replace('_','')

            try:
                ground_truth_val = float(ground_truth_text)
                predicted_val = float(predicted_text)
            except:
                continue

            mse = (predicted_val - ground_truth_val)**2
            total_mse += mse
            count += 1

    if count > 0:
        return total_mse / count
    else:
        return None

# --------------------------------------------------------------------------
#  Identify GPU with best free memory
# --------------------------------------------------------------------------
def get_best_gpu():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise ValueError("No GPUs available.")

    max_free_mem = 0
    best_gpu = None
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        if free_mem > max_free_mem or (free_mem == max_free_mem and (best_gpu is None or i < best_gpu)):
            max_free_mem = free_mem
            best_gpu = i

    if best_gpu is None:
        best_gpu = 0
    return best_gpu

# --------------------------------------------------------------------------
#  Collate function to handle padding/masking
# --------------------------------------------------------------------------
def collate_fn(batch, tokenizer):
    inputs = tokenizer(batch, return_tensors="pt", padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    labels = input_ids.clone()
    equal_token_id = tokenizer.convert_tokens_to_ids('=')
    for i, input_id in enumerate(input_ids):
        equal_pos = (input_id == equal_token_id).nonzero(as_tuple=True)
        if len(equal_pos[0]) > 0:
            eq_pos = equal_pos[0].item()
            # Mask everything before and including '='
            labels[i, :eq_pos+1] = -100
        else:
            labels[i, :] = -100
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

# --------------------------------------------------------------------------
#  Curriculum: We'll define 3 "stages". We'll produce tasks accordingly
# --------------------------------------------------------------------------
def generate_4function_str(a, b, op, allow_thinking, space_tokens):
    """
    Generate a sample string for x op y = z
    Insert 10 underscores if allow_thinking is True.
    For multiplication or large results, convert to sci if needed.
    """
    if op == '+':
        ans = a + b
        prompt = f"{a} + {b} ="
    elif op == '-':
        ans = a - b
        prompt = f"{a} - {b} ="
    elif op == '*':
        ans = a * b
        prompt = f"{a} * {b} ="
    elif op == '/':
        # avoid divide by zero
        if b == 0:
            b = 1
        ans = a / b
        prompt = f"{a} / {b} ="
    # Convert to str, possibly in scientific notation if large
    ans_str = convert_to_sci_if_needed(str(ans))
    # Insert underscores if allow_thinking
    if allow_thinking:
        # e.g. " = _ _ _ ... _  10 times ... <answer>"
        # We'll just do the final string as:
        # "X op Y = _ _ _ _ _ _ _ _ _ _ {answer}"
        # then if space_tokens => separate with spaces
        underscores = "_ " * 10 if space_tokens else "_"*10
        full_str = f"{prompt} {underscores} {ans_str}"
    else:
        full_str = f"{prompt} {ans_str}"

    if space_tokens:
        # Insert spacing so that each char is split
        # e.g. '12 / 3 = 4' -> '1 2   /   3   =   4'
        spaced = ' '.join(list(full_str))
        return spaced
    else:
        return full_str

def generate_factorial_str(a, allow_thinking, space_tokens):
    # factorial
    # We'll do something like: "a ! = answer"
    try:
        # compute factorial
        val = 1
        for i in range(1, a+1):
            val *= i
    except:
        val = 1
    prompt = f"{a} ! ="
    ans_str = convert_to_sci_if_needed(str(val))
    if allow_thinking:
        underscores = "_ " * 10 if space_tokens else "_"*10
        full_str = f"{prompt} {underscores} {ans_str}"
    else:
        full_str = f"{prompt} {ans_str}"
    if space_tokens:
        return ' '.join(list(full_str))
    return full_str

def fib(n):
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b

def generate_fibonacci_str(a, allow_thinking, space_tokens):
    # "fib(a) = answer"
    val = fib(a)
    prompt = f"fib ( {a} ) ="
    ans_str = convert_to_sci_if_needed(str(val))
    if allow_thinking:
        underscores = "_ " * 10 if space_tokens else "_"*10
        full_str = f"{prompt} {underscores} {ans_str}"
    else:
        full_str = f"{prompt} {ans_str}"
    if space_tokens:
        return ' '.join(list(full_str))
    return full_str

def generate_algebra_str(allow_thinking, space_tokens, domain_start, domain_end):
    """
    Randomly pick a format:
      y = m x + b
      y = m ( x + b )
    Then pick random ints for m, x, b in [domain_start, domain_end].
    Then ask for "solve for x".
    We'll produce the prompt with the actual y, then we want the model to find x.
    E.g. "Solve x: y= 5 x + 10 =>  ???"
    or "Solve x: y = 2 ( x + 3 ) => ???"
    We'll do just linear (not polynomials), so we can solve easily.

    We'll do a random solution in the range so we can get the correct x in that domain.
    """
    m = random.randint(domain_start, domain_end)
    x = random.randint(domain_start, domain_end)
    b = random.randint(domain_start, domain_end)

    # pick a pattern
    pattern = random.choice(["y = m x + b", "y = m ( x + b )"])

    # For each pattern, compute y accordingly
    if pattern == "y = m x + b":
        y = m*x + b
        # prompt
        prompt = f"Solve x : y = {m} x + {b} =>"
        # the correct x is 'x'
        ans_str = convert_to_sci_if_needed(str(x))

    else:  # y = m ( x + b )
        y = m * (x + b)
        prompt = f"Solve x : y = {m} ( x + {b} ) =>"
        ans_str = convert_to_sci_if_needed(str(x))

    if allow_thinking:
        underscores = "_ " * 10 if space_tokens else "_"*10
        full_str = f"{prompt} {underscores} {ans_str}"
    else:
        full_str = f"{prompt} {ans_str}"

    if space_tokens:
        return ' '.join(list(full_str))

    return full_str

# --------------------------------------------------------------------------
#  A single dataset class that can produce tasks for the *current stage*
# --------------------------------------------------------------------------
class MultiTaskDataset(Dataset):
    """
    stage: 1 => 4 functions only
           2 => 4 functions + factorial + fibonacci
           3 => 4 functions + factorial + fibonacci + algebra
    domain_start, domain_end => random picks from [domain_start, domain_end]
    size => how many samples
    """
    def __init__(self, stage, domain_start, domain_end, size=10000, space_tokens=False, allow_thinking=False):
        super().__init__()
        self.data = []
        self.stage = stage

        ops = ['+', '-', '*', '/']  # for stage 1, we have these
        # We'll unify them in a single list if stage=2, add factorial/fib. If stage=3, add algebra, etc.

        for _ in range(size):
            a = random.randint(domain_start, domain_end)
            b = random.randint(domain_start, domain_end)

            # Weighted random choice among tasks that are valid for current stage
            # For stage 1 => we only pick among ops
            # For stage 2 => pick among ops + factorial + fib
            # For stage 3 => pick among ops + factorial + fib + algebra
            if self.stage == 1:
                # pick from 4 ops
                op = random.choice(ops)
                sample_str = generate_4function_str(a, b, op, allow_thinking, space_tokens)

            elif self.stage == 2:
                # pick from [4 ops, factorial, fibonacci]
                task = random.choice(['op','op','op','op','factorial','fibonacci'])
                if task == 'op':
                    op = random.choice(ops)
                    sample_str = generate_4function_str(a, b, op, allow_thinking, space_tokens)
                elif task == 'factorial':
                    # do factorial of a
                    sample_str = generate_factorial_str(a, allow_thinking, space_tokens)
                else:
                    # fibonacci
                    sample_str = generate_fibonacci_str(a, allow_thinking, space_tokens)

            else:
                # stage == 3 => pick from [4 ops, factorial, fibonacci, algebra]
                task = random.choice(['op','op','op','op','factorial','fibonacci','algebra'])
                if task == 'op':
                    op = random.choice(ops)
                    sample_str = generate_4function_str(a, b, op, allow_thinking, space_tokens)
                elif task == 'factorial':
                    sample_str = generate_factorial_str(a, allow_thinking, space_tokens)
                elif task == 'fibonacci':
                    sample_str = generate_fibonacci_str(a, allow_thinking, space_tokens)
                else:
                    sample_str = generate_algebra_str(allow_thinking, space_tokens, domain_start, domain_end)

            self.data.append(sample_str)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --------------------------------------------------------------------------
#  MeZO parameter perturb
# --------------------------------------------------------------------------
def perturb_parameters(model, epsilon, seed):
    torch.manual_seed(seed)
    for param in model.parameters():
        z = torch.randn_like(param.data)
        param.data.add_(epsilon * z)

# --------------------------------------------------------------------------
#  Compute KLSparsity loss
# --------------------------------------------------------------------------
def compute_klsparsity_loss(model, pi):
    kl_loss = 0.0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, 'activations'):
                activations = module.activations
                p = torch.mean(torch.sigmoid(activations), dim=0)
                kl = pi * torch.log(pi / (p + 1e-10)) + (1 - pi) * torch.log((1 - pi) / (1 - p + 1e-10))
                kl_loss += kl.sum()
    return kl_loss

# --------------------------------------------------------------------------
#  Hook to store activations for KLSparsity
# --------------------------------------------------------------------------
def store_activations(module, input, output):
    module.activations = output

# --------------------------------------------------------------------------
#  The main train function
# --------------------------------------------------------------------------
def train(args):
    global tokenizer

    # Initialize wandb
    wandb.init(project="Extended-Math-Curriculum", config=vars(args))

    # Decide device
    if torch.cuda.is_available():
        device_id = get_best_gpu()
        device = torch.device(f'cuda:{device_id}')
        print(f"Using device {device_id}")
    else:
        device = torch.device('cpu')

    # Load tokenizer + model
    # You can expand the "ASCII characters" in the minimal tokenizer if you wish,
    # or just use the standard pythia tokenizer. For demonstration, let's keep pythia's default,
    # but you could also implement your custom ASCII-based WordLevel if needed.
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-410m",
        bos_token="<bos>",
        eos_token="<eos>",
        sep_token="<sep>",
        pad_token="<pad>",
    )
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if args.raw_weights:
        print("initing weights back to raw")
        model.init_weights()

    # Minimal tokenizer approach if requested
    if args.limited_tokens:
        print("initing the tokenizer to minimal set (but now extended with ASCII) ...")
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        # ASCII range: 32 to 126
        ascii_chars = [chr(i) for i in range(32, 127)]
        special_tokens = ["<bos>", "<eos>", "<sep>", "<pad>", "<unk>"]
        all_tokens = special_tokens + ascii_chars
        vocab = {tok: i for i, tok in enumerate(all_tokens)}

        tokenizer_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer_obj.pre_tokenizer = Whitespace()

        new_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj,
            bos_token="<bos>",
            eos_token="<eos>",
            sep_token="<sep>",
            pad_token="<pad>",
            unk_token="<unk>"
        )
        tokenizer = new_tokenizer
        model.resize_token_embeddings(len(tokenizer))

    # Register hooks for KLSparsity
    if args.klsparsity:
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.register_forward_hook(store_activations)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_iters,
    )

    # Curriculum control
    # We'll define stage=1 => 4 ops, stage=2 => 4 ops + factorial + fib, stage=3 => + algebra
    current_stage = 1
    max_stage = 3

    train_domain1 = 0
    train_domain2 = 10

    # We'll track how many times in a row we get >99% on validation for each stage
    passing_increments_count = 0
    # We'll only proceed to the next stage after 10 increments in a row with >99% val accuracy
    # Then for stage=2, we watch again for factorial+fib, etc.
    # Then once we pass stage=2, we move to stage=3.

    # Because we want to continue indefinitely (or up to a max iteration), let's do a while loop:
    while current_stage <= max_stage:
        print(f"\n=== Training Stage {current_stage} ===\n")
        # Build train & val dataset for the current domain
        train_dataset = MultiTaskDataset(
            stage=current_stage,
            domain_start=train_domain1,
            domain_end=train_domain2,
            size=5000,
            space_tokens=args.limited_tokens,
            allow_thinking=args.allow_thinking
        )
        val_dataset = MultiTaskDataset(
            stage=current_stage,
            domain_start=train_domain2+1,
            domain_end=train_domain2+10,
            size=1000,
            space_tokens=args.limited_tokens,
            allow_thinking=args.allow_thinking
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=args.micro_batch_size, shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, tokenizer),
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.micro_batch_size, shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer),
        )

        # We'll do a local training loop until we either get good train accuracy or exhaust max_iters
        patience_counter = 0
        train_iters = 0
        train_correct_tokens = 0
        train_total_tokens = 0
        train_loss_total = 0.0

        # We'll do an indefinite training loop, break when we pass our "training_at_100_patience" or max_iters
        while True:
            model.train()
            for batch in train_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                if args.gaf:
                    # GAF approach
                    batch_size = input_ids.size(0)
                    if batch_size < 2:
                        continue
                    indices = torch.randperm(batch_size)
                    mid_point = batch_size // 2
                    batch1_indices = indices[:mid_point]
                    batch2_indices = indices[mid_point:]

                    # First micro-batch
                    input_ids1 = input_ids[batch1_indices]
                    attention_mask1 = attention_mask[batch1_indices]
                    labels1 = labels[batch1_indices]
                    outputs1 = model(input_ids=input_ids1, attention_mask=attention_mask1, labels=labels1)
                    loss1 = outputs1.loss
                    optimizer.zero_grad()
                    loss1.backward()
                    G1 = [p.grad.clone() for p in model.parameters()]
                    optimizer.zero_grad()

                    # Second micro-batch
                    input_ids2 = input_ids[batch2_indices]
                    attention_mask2 = attention_mask[batch2_indices]
                    labels2 = labels[batch2_indices]
                    outputs2 = model(input_ids=input_ids2, attention_mask=attention_mask2, labels=labels2)
                    loss2 = outputs2.loss
                    optimizer.zero_grad()
                    loss2.backward()
                    G2 = [p.grad.clone() for p in model.parameters()]
                    optimizer.zero_grad()

                    filtered_grad, cosine_distance = filter_gradients(G1, G2, args.gaf_tau)
                    loss_value = (loss1.item() + loss2.item())/2
                    base_loss = loss_value
                    kl_loss = 0
                    outputs = outputs1
                    labels_for_acc = labels1
                    input_ids_for_acc = input_ids1

                    if filtered_grad is not None:
                        with torch.no_grad():
                            for param, grad in zip(model.parameters(), filtered_grad):
                                param.grad = grad
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    else:
                        print(f"Skipping batch update: cos_dist={cosine_distance}")

                elif args.mezo:
                    # MeZO
                    seed_ = random.randint(0, int(1e6))
                    perturb_parameters(model, args.mezo_epsilon, seed_)
                    outputs_pos = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    base_loss_pos = outputs_pos.loss.item()
                    if args.klsparsity:
                        kl_loss_pos = compute_klsparsity_loss(model, args.klsparsity_pi).item()
                    else:
                        kl_loss_pos = 0.0
                    total_loss_pos = base_loss_pos + args.klsparsity_lambda * kl_loss_pos

                    # Negative
                    perturb_parameters(model, -2 * args.mezo_epsilon, seed_)
                    outputs_neg = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    base_loss_neg = outputs_neg.loss.item()
                    if args.klsparsity:
                        kl_loss_neg = compute_klsparsity_loss(model, args.klsparsity_pi).item()
                    else:
                        kl_loss_neg = 0.0
                    total_loss_neg = base_loss_neg + args.klsparsity_lambda * kl_loss_neg

                    # Reset
                    perturb_parameters(model, args.mezo_epsilon, seed_)

                    # Pseudo-grad
                    grad_scalar = (total_loss_pos - total_loss_neg)/(2 * args.mezo_epsilon)
                    torch.manual_seed(seed_)
                    for param in model.parameters():
                        if param.requires_grad:
                            z = torch.randn_like(param.data)
                            if param.grad is not None:
                                param.grad.zero_()
                            else:
                                param.grad = torch.zeros_like(param.data)
                            param.grad.add_(grad_scalar * z)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    base_loss = (base_loss_pos + base_loss_neg)/2
                    kl_loss = (kl_loss_pos + kl_loss_neg)/2 if args.klsparsity else 0.0
                    loss_value = (total_loss_pos + total_loss_neg)/2
                    outputs = outputs_pos
                    labels_for_acc = labels
                    input_ids_for_acc = input_ids

                else:
                    # Standard
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    base_loss = outputs.loss
                    if args.klsparsity:
                        kl_loss = compute_klsparsity_loss(model, args.klsparsity_pi)
                    else:
                        kl_loss = 0.0

                    total_loss = base_loss + args.klsparsity_lambda * kl_loss
                    optimizer.zero_grad()
                    total_loss.backward()
                    # Log-normal gradient noise
                    if args.log_normal_gradient_noise:
                        with torch.no_grad():
                            for param in model.parameters():
                                if param.grad is not None:
                                    noise = torch.empty_like(param.grad).normal_(
                                        mean=args.log_normal_mu, std=args.log_normal_sigma
                                    ).exp_()
                                    param.grad.mul_(noise)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    loss_value = total_loss.item()
                    base_loss = base_loss.item()
                    kl_loss = kl_loss.item() if args.klsparsity else 0.0
                    labels_for_acc = labels
                    input_ids_for_acc = input_ids

                # Weighted decay cap at 10
                if optimizer.param_groups[0]['weight_decay'] > 10.0:
                    optimizer.param_groups[0]['weight_decay'] = 10.0

                # Per-token accuracy
                with torch.no_grad():
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels_for_acc[..., 1:].contiguous()
                    loss_mask = shift_labels != -100
                    pred_tokens = shift_logits.argmax(dim=-1)
                    c = (pred_tokens == shift_labels) & loss_mask
                    num_correct = c.sum().item()
                    num_total = loss_mask.sum().item()
                    train_correct_tokens += num_correct
                    train_total_tokens += num_total

                # Weight decay loss (for logging)
                weight_decay_loss = 0.0
                for group in optimizer.param_groups:
                    for param in group['params']:
                        if param.requires_grad:
                            weight_decay_loss += torch.sum(param.data**2)
                weight_decay_loss *= optimizer.param_groups[0]['weight_decay']

                train_loss_total += (loss_value + weight_decay_loss)
                train_iters += 1

                # MSE
                with torch.no_grad():
                    mse_ = compute_mse_on_parsed_answers(input_ids_for_acc, labels_for_acc, logits, tokenizer)

                # Log to wandb
                message = {
                    'train_loss': loss_value,
                    'base_loss': base_loss,
                    'kl_loss': kl_loss,
                    'weight_decay_loss': weight_decay_loss.item(),
                    'lr': scheduler.get_last_lr()[0],
                    'weight_decay': optimizer.param_groups[0]['weight_decay'],
                    'train_iters': train_iters,
                    'train_token_acc': (train_correct_tokens / train_total_tokens) if train_total_tokens>0 else 0.0,
                }
                if mse_ is not None:
                    message['train_mse'] = mse_
                wandb.log(message)
                print(message)

                # Check if we've reached 100% train token accuracy or near it
                if (train_correct_tokens / train_total_tokens) >= 0.98:
                    patience_counter += 1
                else:
                    patience_counter = 0

                if patience_counter >= args.training_at_100_patience or train_iters >= args.max_iters:
                    break
            if patience_counter >= args.training_at_100_patience or train_iters >= args.max_iters:
                break

        # --------------------------
        # Validation
        # --------------------------
        model.eval()
        val_tasks_tracked = {
            "add": [0,0], "subtract": [0,0], "multiply": [0,0],
            "divide": [0,0], "factorial": [0,0], "fibonacci": [0,0],
            "algebra": [0,0]
        }
        val_correct_overall = 0
        val_total_overall = 0
        val_incorrect_samples = []
        val_loss_total = 0.0
        val_base_loss_total = 0.0
        val_kl_loss_total = 0.0
        val_weight_decay_loss = 0.0
        val_mse_sum = 0.0
        val_mse_count = 0

        with torch.no_grad():
            for val_batch in val_dataloader:
                input_ids = val_batch['input_ids'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                labels = val_batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                base_loss_val = outputs.loss
                if args.klsparsity:
                    kl_loss_val = compute_klsparsity_loss(model, args.klsparsity_pi)
                else:
                    kl_loss_val = 0.0
                total_loss_val = base_loss_val + args.klsparsity_lambda * kl_loss_val

                # weight decay
                wdl = 0.0
                for group in optimizer.param_groups:
                    for param in group['params']:
                        if param.requires_grad:
                            wdl += torch.sum(param.data**2)
                wdl *= optimizer.param_groups[0]['weight_decay']

                val_loss_total += total_loss_val.item()
                val_base_loss_total += base_loss_val.item()
                val_kl_loss_total += kl_loss_val.item() if args.klsparsity else 0.0
                val_weight_decay_loss += wdl

                logits = outputs.logits
                # Now compute per-sample accuracy
                batch_acc, batch_incorrect, val_tasks_tracked = compute_per_sample_accuracy(
                    input_ids, labels, logits, tokenizer, val_tasks_tracked
                )
                val_correct_overall += batch_acc * input_ids.size(0)
                val_total_overall += input_ids.size(0)
                val_incorrect_samples.extend(batch_incorrect)

                # MSE
                mse_ = compute_mse_on_parsed_answers(input_ids, labels, logits, tokenizer)
                if mse_ is not None:
                    val_mse_sum += mse_
                    val_mse_count += 1

        # Summaries
        val_acc_per_sample = val_correct_overall / val_total_overall if val_total_overall>0 else 0.0
        val_loss_avg = val_loss_total / len(val_dataloader)
        val_base_loss_avg = val_base_loss_total / len(val_dataloader)
        val_kl_loss_avg = val_kl_loss_total / len(val_dataloader)
        val_weight_decay_loss_avg = val_weight_decay_loss / len(val_dataloader)
        val_mse_avg = val_mse_sum / val_mse_count if val_mse_count>0 else None

        # Compute per-task accuracy:
        task_accuracies = {}
        for tkey, (corr, tot) in val_tasks_tracked.items():
            task_accuracies[tkey] = (corr/tot) if tot>0 else 0.0
        # The average among tasks we actually used:
        # We'll consider tasks with tot>0 as the ones used in this stage's validation
        used_accuracies = [task_accuracies[k] for k in val_tasks_tracked if val_tasks_tracked[k][1]>0]
        if len(used_accuracies) == 0:
            avg_acc_across_tasks = 0.0
        else:
            avg_acc_across_tasks = sum(used_accuracies)/len(used_accuracies)

        print(f"\n==== Validation Results for Stage {current_stage} ====")
        print(f"val_acc_per_sample (overall) = {val_acc_per_sample*100:.2f}%")
        print(f"avg_acc_across_tasks        = {avg_acc_across_tasks*100:.2f}%")
        print("Task Accuracies:")
        for tk, acc_ in task_accuracies.items():
            print(f"  {tk} => {acc_*100:.2f}%  (count={val_tasks_tracked[tk][1]})")
        print("\nIncorrect Samples (subset):")
        for sample in val_incorrect_samples[:10]:  # just show a few
            print(sample)
            print("-"*40)

        # wandb message
        log_msg = {
            "val_loss": val_loss_avg,
            "val_base_loss": val_base_loss_avg,
            "val_kl_loss": val_kl_loss_avg,
            "val_weight_decay_loss": val_weight_decay_loss_avg,
            "val_acc_overall": val_acc_per_sample,
            "val_acc_average_tasks": avg_acc_across_tasks,
            "val_mse": val_mse_avg,
            "train_domain2": train_domain2,
            "stage": current_stage,
        }
        for tk, acc_ in task_accuracies.items():
            log_msg[f"val_acc_{tk}"] = acc_

        wandb.log(log_msg)
        print(log_msg)

        # Now check if we pass the >99% threshold on average accuracy across tasks
        # If yes, we do the +10 to train_domain2. If we pass 10 times in a row, we move next stage.
        if avg_acc_across_tasks >= 0.99: #0.99
            passing_increments_count += 1
            # If we get 10 passes, move on
            if passing_increments_count >= 10:
                print(f"Stage {current_stage}: Reached 10 increments with >99% val accuracy. Moving to next stage.")
                current_stage += 1
                passing_increments_count = 0

                # Only break out of the entire loop if we haven't yet advanced beyond max_stage
                if current_stage > max_stage:
                    break

            # If still within stage, we skip training further and just do domain2 += 10
            train_domain2 += 10
            print(f"Validation pass => increment domain to {train_domain2}, passing_increments_count={passing_increments_count}")
            # Do NOT train further if we're passing => we go back to the top, re-generate dataset
            # continuing the while current_stage <= max_stage
        else:
            # If we fail, reset passing_increments_count, but also STILL do domain2 += 10
            # and keep training
            passing_increments_count = 0
            train_domain2 += 10
            print(f"Validation not pass => increment domain to {train_domain2}, passing_increments_count=0")
            # We'll just continue the while loop on the same stage

    print("\n=== Done with all stages! ===")

# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LLM on a multi-stage math curriculum")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="Micro batch size")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--training_at_100_patience", type=int, default=100, help="Patience for near-100% train token accuracy")
    parser.add_argument("--max_iters", type=int, default=100000, help="Max number of train iterations per domain increment")
    parser.add_argument("--weight_decay_schedule", action="store_true", help="Use weight decay schedule")
    parser.add_argument("--weight_decay_k", type=float, default=1.0, help="Weight decay multiplier k")
    parser.add_argument("--gaf", action="store_true", help="Use Gradient Agreement Filtering")
    parser.add_argument("--gaf_tau", type=float, default=1.0, help="GAF tau value")
    parser.add_argument("--mezo", action="store_true", help="Use Memory Efficient Zero Order optimization")
    parser.add_argument("--mezo_epsilon", type=float, default=0.001, help="MeZO epsilon value")
    parser.add_argument("--klsparsity", action="store_true", help="Use KLSparsity regularization")
    parser.add_argument("--klsparsity_pi", type=float, default=0.05, help="KLSparsity pi value")
    parser.add_argument("--klsparsity_lambda", type=float, default=0.1, help="KLSparsity lambda value")
    parser.add_argument("--log_normal_gradient_noise", action="store_true", help="Add log-normal gradient noise")
    parser.add_argument("--log_normal_mu", type=float, default=0.0, help="Log-normal mu")
    parser.add_argument("--log_normal_sigma", type=float, default=0.01, help="Log-normal sigma")
    parser.add_argument("--raw_weights", action="store_true", help="Initialize the model with raw weights")
    parser.add_argument("--limited_tokens", action="store_true", help="Use a minimal ASCII-based WordLevel tokenizer")
    parser.add_argument("--allow_thinking", action="store_true", help="Insert 10 underscore tokens after '=' to let the LLM 'think'")

    args = parser.parse_args()
    train(args)
