import argparse
import torch
import wandb
import random
import torch.nn.functional as F
import numpy as np
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

from torch.utils.data import DataLoader, Dataset

os.environ["WANDB_API_KEY"] = ""

# ------------------------------
# Utility / Shared Functions
# ------------------------------

def format_number(val: float) -> str:
    """
    Convert number to a string with at most 5 significant figures.
    For instance, 123456 -> "1.2346e+05"
    or 0.0000123456 -> "1.2346e-05"
    Otherwise just a normal float if it can be expressed <= 5 sig figs.
    """
    return f"{val:.5g}"

def floats_match_up_to_2_decimals(pred: float, gt: float) -> bool:
    """
    Return True if (pred - gt) < 0.01 in absolute value.
    """
    return abs(pred - gt) < 0.01

def parse_float_with_tolerance(s: str):
    """
    Attempt to parse a string to float. If it fails, return None.
    """
    try:
        return float(s)
    except:
        return None

def parse_and_compare_prediction(
    predicted_text: str,
    ground_truth_text: str,
    allow_decimal_tolerance: bool = True
):
    """
    1) Remove underscores/spaces from both predicted and ground truth
    2) Attempt float conversion
    3) If both integers -> exact match
    4) If either is float -> match up to 2 decimals
    Return (correct: bool, (pred_value, gt_value) or None)
    """
    # Remove spaces and underscores
    predicted_text = predicted_text.replace(" ", "").replace("_", "")
    ground_truth_text = ground_truth_text.replace(" ", "").replace("_", "")

    pred_val = parse_float_with_tolerance(predicted_text)
    gt_val = parse_float_with_tolerance(ground_truth_text)

    if pred_val is None or gt_val is None:
        return False, None

    # Check if both are "effectively integers"
    if float(pred_val).is_integer() and float(gt_val).is_integer():
        return (pred_val == gt_val), (pred_val, gt_val)
    else:
        # Compare up to 2 decimals
        if allow_decimal_tolerance:
            return floats_match_up_to_2_decimals(pred_val, gt_val), (pred_val, gt_val)
        else:
            return pred_val == gt_val, (pred_val, gt_val)

# ------------------------------
# Filter Gradients (GAF)
# ------------------------------

def filter_gradients(G1, G2, cos_distance_thresh=1):
    G1_flat = torch.cat([g.view(-1) for g in G1])
    G2_flat = torch.cat([g.view(-1) for g in G2])
    cos_sim = F.cosine_similarity(G1_flat, G2_flat, dim=0)
    cos_distance = 1 - cos_sim
    if cos_distance > cos_distance_thresh:
        return None, cos_distance.item()
    return [(g1 + g2) / 2 for g1, g2 in zip(G1, G2)], cos_distance.item()

# ------------------------------
# Curriculum and Data Generation
# ------------------------------

TASK_4FUNCTION = ["add", "sub", "mul", "div"]
TASK_EXOTIC = ["factorial", "fib"]
TASK_ALGEBRA = ["algebra"]

def generate_4function_example(space_tokens: bool = False, thinking_steps: int = 0):
    """
    Return a random problem from +, -, *, /
    We format the ground truth with possible scientific notation if needed.
    """
    op = random.choice(TASK_4FUNCTION)
    x = random.randint(1, 9999)
    y = random.randint(1, 9999)

    if op == "add":
        result = x + y
        op_symbol = "+"
    elif op == "sub":
        result = x - y
        op_symbol = "-"
    elif op == "mul":
        result = x * y
        op_symbol = "*"
    else:
        # division
        # Avoid dividing by zero
        if y == 0:
            y = 1
        result = x / y
        op_symbol = "/"

    # Possibly format result in scientific notation if big
    result_str = format_number(result)

    # Insert the "thinking" underscores if requested
    underscores = " ".join(["_"] * thinking_steps)
    # We'll add them right after "=" with one space
    # Final ground truth is after these underscores
    # e.g. "... = _ _ _ _ 42"
    example = f"{x} {op_symbol} {y} = {underscores} {result_str}" if space_tokens else f"{x}{op_symbol}{y}={underscores}{result_str}"

    return example, op  # We return both the text and the task type

def factorial(n: int) -> int:
    if n < 2:
        return 1
    return n * factorial(n - 1)

def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

def generate_exotic_example(space_tokens: bool = False, thinking_steps: int = 0):
    """
    Return either factorial or fibonacci randomly
    """
    op = random.choice(TASK_EXOTIC)
    # restrict n so it doesn't blow up too big
    if op == "factorial":
        x = random.randint(0, 12)  # factorial(12) = 479001600 is big but safe
        result = factorial(x)
        prompt_str = f"{x} !"
    else:
        x = random.randint(0, 20)  # fib(20) = 6765
        result = fib(x)
        prompt_str = f"fib ( {x} )"

    result_str = format_number(result)
    underscores = " ".join(["_"] * thinking_steps)

    if space_tokens:
        example = f"{prompt_str} = {underscores} {result_str}"
    else:
        # might remove spaces in the prompt for a quick style
        prompt_str_nospace = prompt_str.replace(" ", "")
        example = f"{prompt_str_nospace}={underscores}{result_str}"

    return example, op

def generate_algebra_example(space_tokens: bool = False, thinking_steps: int = 0):
    """
    Generate a random "solve for x" style problem.
    We'll create a random form, e.g. y = m*x + b, or y = m*(x+b), ...
    Then we pick random ints for m, b, x => compute y => the problem is "y= m*x+b, solve for x".
    The ground truth is x. 
    """
    # We can define a few forms:
    # 1) y = m * x + b
    # 2) y = m * (x + b)
    # 3) y = (x + b) / m
    # 4) y = (m + x) / b
    # 5) y = m * x - b
    forms = [
        "y = m * x + b",
        "y = m * ( x + b )",
        "y = ( x + b ) / m",
        "y = ( m + x ) / b",
        "y = m * x - b"
    ]
    form = random.choice(forms)

    m = random.randint(1, 15)
    b = random.randint(1, 15)
    x = random.randint(0, 15)

    # Compute y according to the form
    # We'll do a small eval approach carefully
    # we replace m, x, b, then evaluate for y
    # Then the question is: solve for x
    if form == "y = m * x + b":
        y = m * x + b
        eq_str = f"{y} = {m} * x + {b}"
    elif form == "y = m * ( x + b )":
        y = m * (x + b)
        eq_str = f"{y} = {m} * ( x + {b} )"
    elif form == "y = ( x + b ) / m":
        y = (x + b) / m
        eq_str = f"{y} = ( x + {b} ) / {m}"
    elif form == "y = ( m + x ) / b":
        # ensure b!=0
        if b == 0:
            b = 1
        y = (m + x) / b
        eq_str = f"{y} = ( {m} + x ) / {b}"
    else:
        # y = m*x - b
        y = m * x - b
        eq_str = f"{y} = {m} * x - {b}"

    # The ground truth we want is x
    # Possibly format y with .5g if it is large or small
    # same for m, b
    eq_str = eq_str.replace(str(y), format_number(y))
    eq_str = eq_str.replace(str(m), format_number(m))
    eq_str = eq_str.replace(str(b), format_number(b))

    # The question: "solve for x"
    # We'll place an "=" for the final so that the LLM can produce the numeric answer
    underscores = " ".join(["_"] * thinking_steps)
    # ground truth is x
    x_str = format_number(x)

    if space_tokens:
        example = f"{eq_str}, solve for x = {underscores} {x_str}"
    else:
        eq_nospace = eq_str.replace(" ", "")
        example = f"{eq_nospace},solveforx={underscores}{x_str}"

    return example, "algebra"

class CurriculumDataset(Dataset):
    """
    This dataset will create random samples based on the 'stage' of training:
    - stage="4function" => generate from +, -, *, /
    - stage="exotic" => factorial, fib
    - stage="algebra" => random algebra
    Each __getitem__ returns one example string + metadata (which sub-task it belongs to).
    """
    def __init__(self, stage, size=10000, space_tokens=False, thinking_steps=0):
        self.stage = stage
        self.space_tokens = space_tokens
        self.thinking_steps = thinking_steps
        self.size = size

        # Pre-generate
        self.data = []
        self.labels = []  # store sub-task type
        for _ in range(size):
            if self.stage == "4function":
                ex, subtask = generate_4function_example(
                    space_tokens=self.space_tokens,
                    thinking_steps=self.thinking_steps,
                )
            elif self.stage == "exotic":
                ex, subtask = generate_exotic_example(
                    space_tokens=self.space_tokens,
                    thinking_steps=self.thinking_steps,
                )
            else:
                ex, subtask = generate_algebra_example(
                    space_tokens=self.space_tokens,
                    thinking_steps=self.thinking_steps,
                )
            self.data.append(ex)
            self.labels.append(subtask)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "text": self.data[idx],
            "subtask": self.labels[idx]
        }

# ------------------------------
# Collate + Accuracy
# ------------------------------

def collate_fn(batch, tokenizer):
    texts = [b["text"] for b in batch]
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    labels = input_ids.clone()

    # Identify '='. Then we allow everything before '=' to remain -100 in labels
    equal_token_id = tokenizer.convert_tokens_to_ids('=')

    # We mask out everything *before and including* '=' from the label
    # so the model only "predicts" the text after '='
    for i, input_id in enumerate(input_ids):
        eq_pos = (input_id == equal_token_id).nonzero(as_tuple=True)
        if len(eq_pos[0]) > 0:
            eq_idx = eq_pos[0][0].item()
            # mask out everything up to eq_idx
            labels[i, :eq_idx + 1] = -100
        else:
            labels[i, :] = -100
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'subtasks': [b["subtask"] for b in batch]
    }

def compute_accuracy_per_subtask_and_overall(
    model,
    tokenizer,
    dataloader,
    device,
    tasks_of_interest,
    decimal_tolerance=True
):
    """
    Return a dict of accuracies for each subtask and an 'average' for them.
    We'll parse each sample after '=' to see if it matches ground truth.
    """
    model.eval()
    correct_counts = {t: 0 for t in tasks_of_interest}
    total_counts = {t: 0 for t in tasks_of_interest}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            subtasks = batch['subtasks']  # list of strings

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            logits = outputs.logits

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                subtask = subtasks[i]
                # Subtask might be one of 'add','sub','mul','div','factorial','fib','algebra'
                if subtask not in tasks_of_interest:
                    # skip if not relevant for the current stage
                    continue

                # find '=' in input_ids[i]
                eq_token_id = tokenizer.convert_tokens_to_ids('=')
                eq_pos = (input_ids[i] == eq_token_id).nonzero(as_tuple=True)

                if len(eq_pos[0]) > 0:
                    eq_idx = eq_pos[0][0].item()
                    # ground truth tokens
                    ground_truth_tokens = labels[i, eq_idx + 1:]
                    ground_truth_tokens = ground_truth_tokens[ground_truth_tokens != -100]

                    # predicted tokens
                    predicted_logits = logits[i, eq_idx+1 : eq_idx+1 + len(ground_truth_tokens), :]
                    predicted_tokens = predicted_logits.argmax(dim=-1)

                    ground_truth_text = tokenizer.decode(ground_truth_tokens, skip_special_tokens=True).strip()
                    predicted_text = tokenizer.decode(predicted_tokens, skip_special_tokens=True).strip()

                    # Compare
                    is_correct, _ = parse_and_compare_prediction(predicted_text, ground_truth_text, decimal_tolerance)
                    total_counts[subtask] += 1
                    if is_correct:
                        correct_counts[subtask] += 1

    # compute accuracies
    accuracies = {}
    avg_acc = 0.0
    count_t = 0
    for t in tasks_of_interest:
        if total_counts[t] > 0:
            acc = correct_counts[t] / total_counts[t]
            accuracies[t] = acc
            avg_acc += acc
            count_t += 1
        else:
            # in case we had none for that subtask
            accuracies[t] = 0

    if count_t > 0:
        accuracies["average"] = avg_acc / count_t
    else:
        accuracies["average"] = 0.0

    return accuracies

# ------------------------------
# MSE on parsed answers (optional)
# ------------------------------

def compute_mse_on_parsed_answers(input_ids, labels, logits, tokenizer, device):
    """
    Example function if you still want MSE logging (optional).
    """
    batch_size = input_ids.size(0)
    total_mse = 0.0
    count = 0
    eq_token_id = tokenizer.convert_tokens_to_ids('=')

    for i in range(batch_size):
        eq_pos = (input_ids[i] == eq_token_id).nonzero(as_tuple=True)
        if len(eq_pos[0]) > 0:
            eq_idx = eq_pos[0].item()
            gt_tokens = labels[i, eq_idx + 1:]
            gt_tokens = gt_tokens[gt_tokens != -100]
            predicted_logits = logits[i, eq_idx+1 : eq_idx+1 + len(gt_tokens), :]
            pred_tokens = predicted_logits.argmax(dim=-1)
            gt_text = tokenizer.decode(gt_tokens, skip_special_tokens=True).strip()
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            # remove underscores
            gt_text = gt_text.replace("_", "").replace(" ", "")
            pred_text = pred_text.replace("_", "").replace(" ", "")

            gt_val = parse_float_with_tolerance(gt_text)
            pred_val = parse_float_with_tolerance(pred_text)
            if gt_val is not None and pred_val is not None:
                total_mse += (gt_val - pred_val) ** 2
                count += 1

    if count > 0:
        return total_mse / count
    else:
        return None

# ------------------------------
# Perturb + KLSparsity + etc.
# ------------------------------

def perturb_parameters(model, epsilon, seed):
    torch.manual_seed(seed)
    for param in model.parameters():
        z = torch.randn_like(param.data)
        param.data.add_(epsilon * z)

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

def store_activations(module, input, output):
    module.activations = output

# ------------------------------
# GPU selection
# ------------------------------

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

# ------------------------------
# Main Training Loop
# ------------------------------

def train(args):
    # -------------------
    # Init wandb
    # -------------------
    wandb.init(project="LLM-Curriculum-Math", config=vars(args))

    # -------------------
    # Set device
    # -------------------
    if torch.cuda.is_available():
        device_id = get_best_gpu()
        device = torch.device(f'cuda:{device_id}')
        print(f"Using device {device_id}")
    else:
        device = torch.device('cpu')

    # -------------------
    # Load base tokenizer & model
    # -------------------
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

    # If user wants raw weights
    if args.raw_weights:
        print("Re-initializing model to raw weights")
        model.init_weights()

    # If user wants KLSparsity, attach activation hooks
    if args.klsparsity:
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.register_forward_hook(store_activations)

    # -------------------
    # Minimal tokenizer with all ASCII if requested
    # -------------------
    if args.limited_tokens:
        print("Using minimal tokenizer with all ASCII + special tokens")
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        # Create ASCII vocab: range(32..126) plus special tokens
        special_tokens = ["<bos>", "<eos>", "<sep>", "<pad>", "<unk>"]
        ascii_chars = [chr(i) for i in range(32, 127)]
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

    # -------------------
    # Set up optimizer + scheduler
    # -------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_iters,
    )

    # -------------------
    # Curriculum Settings
    # -------------------
    # We start in stage "4function", then "exotic", then "algebra"
    current_stage = "4function"
    # We'll track how many times we've incremented train_domain2, as per your older code logic.
    domain_increment_count = 0
    # We'll track how many times we've seen >99% average val accuracy in the current stage
    consecutive_high_accuracy_counts = 0

    # Because your original code used "train_domain2 += 10" as an iteration scheme,
    # we can do something similar to mimic that stepping. We'll simply do while loop
    # until we pass some stage threshold, then change stage, etc.
    # We'll define the criteria:
    #  Stage A -> B: once we achieve >99% val average accuracy for 4function
    #    for more than (4 + 10) increments, i.e. 14 increments in total
    #  Stage B -> C: once we achieve >99% val average accuracy for factorial+fib
    #    for some consecutive times, say 3 increments, or set your own logic.
    # (You can adjust the details as needed.)

    # We'll store the subtask sets for each stage
    stage_to_subtasks = {
        "4function": TASK_4FUNCTION,    # ["add", "sub", "mul", "div"]
        "exotic": TASK_EXOTIC,          # ["factorial", "fib"]
        "algebra": TASK_ALGEBRA         # ["algebra"]
    }

    # Some convenience thresholds:
    STAGE_A_THRESHOLD = 0.99
    STAGE_A_REQUIRED_INCREMENTS = 4 + 10  # =14
    STAGE_B_THRESHOLD = 0.99
    STAGE_B_REQUIRED_INCREMENTS = 3       # or however many you want

    # We'll do a big while True, break out when we finish stage "algebra"
    step_count = 0

    while True:
        # Create train and val datasets according to the current stage
        train_dataset = CurriculumDataset(
            stage=current_stage,
            size=20000,
            space_tokens=args.limited_tokens,  # reuse this flag to decide if we space out everything
            thinking_steps=args.thinking_steps
        )
        val_dataset = CurriculumDataset(
            stage=current_stage,
            size=2000,
            space_tokens=args.limited_tokens,
            thinking_steps=args.thinking_steps
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.micro_batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, tokenizer),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.micro_batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, tokenizer),
        )

        print(f"----- Starting Stage: {current_stage} (domain increment: {domain_increment_count}) -----")

        # --------------
        # Train Loop
        # --------------
        model.train()
        patience_counter = 0
        train_iters = 0
        train_correct = 0
        train_total = 0
        train_loss_total = 0.0

        for epoch in range(1):  # you can do multiple epochs if you want
            for batch in train_loader:
                step_count += 1
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Forward
                if args.gaf:
                    # GAF logic
                    batch_size = input_ids.size(0)
                    if batch_size < 2:
                        continue
                    indices = torch.randperm(batch_size)
                    mid = batch_size // 2
                    b1 = indices[:mid]
                    b2 = indices[mid:]

                    # Batch1
                    out1 = model(input_ids=b1, attention_mask=attention_mask[b1], labels=labels[b1])
                    loss1 = out1.loss
                    optimizer.zero_grad()
                    loss1.backward()
                    G1 = [p.grad.clone() for p in model.parameters()]
                    optimizer.zero_grad()

                    # Batch2
                    out2 = model(input_ids=b2, attention_mask=attention_mask[b2], labels=labels[b2])
                    loss2 = out2.loss
                    optimizer.zero_grad()
                    loss2.backward()
                    G2 = [p.grad.clone() for p in model.parameters()]
                    optimizer.zero_grad()

                    filtered_grad, cos_dist = filter_gradients(G1, G2, args.gaf_tau)
                    if filtered_grad is not None:
                        with torch.no_grad():
                            for param, g in zip(model.parameters(), filtered_grad):
                                param.grad = g
                        optimizer.step()
                        scheduler.step()
                    # log
                    loss_value = (loss1.item() + loss2.item()) / 2
                    outputs = out1
                elif args.mezo:
                    # MeZO logic
                    optimizer.zero_grad()
                    seed = random.randint(0, 10**6)
                    # + epsilon
                    perturb_parameters(model, args.mezo_epsilon, seed)
                    out_pos = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    base_loss_pos = out_pos.loss.item()
                    kl_loss_pos = compute_klsparsity_loss(model, args.klsparsity_pi).item() if args.klsparsity else 0.0
                    total_loss_pos = base_loss_pos + args.klsparsity_lambda * kl_loss_pos

                    # - 2*epsilon
                    perturb_parameters(model, -2*args.mezo_epsilon, seed)
                    out_neg = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    base_loss_neg = out_neg.loss.item()
                    kl_loss_neg = compute_klsparsity_loss(model, args.klsparsity_pi).item() if args.klsparsity else 0.0
                    total_loss_neg = base_loss_neg + args.klsparsity_lambda * kl_loss_neg

                    # revert
                    perturb_parameters(model, args.mezo_epsilon, seed)

                    # gradient
                    grad_scalar = (total_loss_pos - total_loss_neg) / (2 * args.mezo_epsilon)
                    torch.manual_seed(seed)
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

                    loss_value = (total_loss_pos + total_loss_neg) / 2
                    outputs = out_pos
                else:
                    # Normal
                    optimizer.zero_grad()
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    base_loss = outputs.loss
                    if args.klsparsity:
                        kl_loss = compute_klsparsity_loss(model, args.klsparsity_pi)
                    else:
                        kl_loss = 0.0
                    total_loss = base_loss + args.klsparsity_lambda * kl_loss

                    # Log-normal gradient noise if requested
                    if args.log_normal_gradient_noise:
                        total_loss.backward()
                        with torch.no_grad():
                            for param in model.parameters():
                                if param.grad is not None:
                                    noise = torch.empty_like(param.grad).normal_(
                                        mean=args.log_normal_mu,
                                        std=args.log_normal_sigma
                                    ).exp_()
                                    param.grad.mul_(noise)
                    else:
                        total_loss.backward()

                    optimizer.step()
                    scheduler.step()
                    loss_value = total_loss.item()

                # Training accuracy (token-level) just for a rough metric
                with torch.no_grad():
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_mask = (shift_labels != -100)
                    pred_tokens = shift_logits.argmax(dim=-1)
                    correct = ((pred_tokens == shift_labels) & loss_mask).sum().item()
                    total = loss_mask.sum().item()
                    train_correct += correct
                    train_total += total

                train_loss_total += loss_value
                train_iters += 1

                # Possibly log
                if train_iters % 100 == 0:
                    train_acc = train_correct / train_total if train_total > 0 else 0.0
                    wandb.log({
                        "stage": current_stage,
                        "train_loss": loss_value,
                        "lr": scheduler.get_last_lr()[0],
                        "weight_decay": optimizer.param_groups[0]['weight_decay'],
                        "token_acc": train_acc,
                        "iteration": train_iters,
                    })
                    print(f"[Iter {train_iters}] stage={current_stage}, train_acc={train_acc:.3f}, loss={loss_value:.3f}")

        # -----------------
        # Validation
        # -----------------
        model.eval()
        accuracies = compute_accuracy_per_subtask_and_overall(
            model=model,
            tokenizer=tokenizer,
            dataloader=val_loader,
            device=device,
            tasks_of_interest=stage_to_subtasks[current_stage],
            decimal_tolerance=True
        )
        val_average_acc = accuracies["average"]

        # MSE (optional)
        val_mse = 0.0
        # If you want:
        #   We'll do a quick loop again or just do one pass in the same loop, but let's do it separately:
        total_mse_samples = 0
        for val_batch in val_loader:
            input_ids = val_batch["input_ids"].to(device)
            attention_mask = val_batch["attention_mask"].to(device)
            labels = val_batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            batch_mse = compute_mse_on_parsed_answers(input_ids, labels, logits, tokenizer, device)
            if batch_mse is not None:
                val_mse += batch_mse
                total_mse_samples += 1
        if total_mse_samples > 0:
            val_mse /= total_mse_samples
        else:
            val_mse = None

        print(f"[Validation] Stage={current_stage}, subtask accuracies={accuracies}, val_mse={val_mse}")
        wandb_dict = {
            "stage": current_stage,
            "val_avg_acc": val_average_acc,
            "val_mse": val_mse if val_mse else 0.0
        }
        for t in stage_to_subtasks[current_stage]:
            wandb_dict[f"val_acc_{t}"] = accuracies[t]
        wandb.log(wandb_dict)

        # ---------------
        # Weight Decay Cap
        # ---------------
        if args.weight_decay_schedule:
            # Multiply
            new_wd = optimizer.param_groups[0]['weight_decay'] * args.weight_decay_k
            if new_wd > 10.0:   # cap at <= 10
                new_wd = 10.0
            optimizer.param_groups[0]['weight_decay'] = new_wd
            print(f"Updated weight decay => {new_wd}")

        domain_increment_count += 1

        # ---------------
        # Check if we proceed to next stage
        # ---------------
        if current_stage == "4function":
            # We want >99% average AND do that for at least (4+10)=14 increments
            if val_average_acc >= STAGE_A_THRESHOLD:
                consecutive_high_accuracy_counts += 1
            else:
                consecutive_high_accuracy_counts = 0

            if consecutive_high_accuracy_counts >= STAGE_A_REQUIRED_INCREMENTS:
                print("==> Achieved 4function >99% accuracy for enough increments! Moving to stage B (exotic).")
                current_stage = "exotic"
                consecutive_high_accuracy_counts = 0
                # reset domain increment? up to you
                # domain_increment_count = 0

        elif current_stage == "exotic":
            # Once we achieve >99% on factorial & fib for X increments => stage C
            if val_average_acc >= STAGE_B_THRESHOLD:
                consecutive_high_accuracy_counts += 1
            else:
                consecutive_high_accuracy_counts = 0

            if consecutive_high_accuracy_counts >= STAGE_B_REQUIRED_INCREMENTS:
                print("==> Achieved exotic >99% accuracy for enough increments! Moving to stage C (algebra).")
                current_stage = "algebra"
                consecutive_high_accuracy_counts = 0

        else:
            # Algebra
            # let's say once we achieve >99% for 2 increments, we can end
            if val_average_acc >= 0.99:
                consecutive_high_accuracy_counts += 1
            else:
                consecutive_high_accuracy_counts = 0

            if consecutive_high_accuracy_counts >= 2:
                print("==> Algebra stage >99% accuracy for 2 increments. Training complete!")
                break

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LLM on a math curriculum")

    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="Micro batch size")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--max_iters", type=int, default=100000, help="Maximum number of steps for the scheduler")
    parser.add_argument("--weight_decay_schedule", action="store_true", help="Use weight decay schedule (multiplier)")
    parser.add_argument("--weight_decay_k", type=float, default=1.0, help="Weight decay multiplier k (capped at <= 10)")
    
    # GAF
    parser.add_argument("--gaf", action="store_true", help="Use Gradient Agreement Filtering")
    parser.add_argument("--gaf_tau", type=float, default=1.0, help="GAF tau value")
    
    # MeZO
    parser.add_argument("--mezo", action="store_true", help="Use Memory Efficient Zero Order optimization")
    parser.add_argument("--mezo_epsilon", type=float, default=0.001, help="MeZO epsilon value")

    # KL-Sparsity
    parser.add_argument("--klsparsity", action="store_true", help="Use KLSparsity regularization")
    parser.add_argument("--klsparsity_pi", type=float, default=0.05, help="KLSparsity pi value")
    parser.add_argument("--klsparsity_lambda", type=float, default=0.1, help="KLSparsity lambda value")

    # Log-normal gradient noise
    parser.add_argument("--log_normal_gradient_noise", action="store_true", help="Add log-normal gradient noise")
    parser.add_argument("--log_normal_mu", type=float, default=0.0, help="Log-normal mu")
    parser.add_argument("--log_normal_sigma", type=float, default=0.01, help="Log-normal sigma")

    # Weight init
    parser.add_argument("--raw_weights", action="store_true", help="Initialize the model with raw weights")

    # Minimal tokenizer that includes all ASCII
    parser.add_argument("--limited_tokens", action="store_true", help="Use minimal WordLevel with ASCII only")

    # "Thinking" underscores
    parser.add_argument("--thinking_steps", type=int, default=0, help="Number of underscores to insert between '=' and the correct answer")

    args = parser.parse_args()
    train(args)
