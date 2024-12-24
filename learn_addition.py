# Learning Addition
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



# Filter gradients function for GAF
def filter_gradients(G1, G2, cos_distance_thresh=1):
    G1_flat = torch.cat([g.view(-1) for g in G1])
    G2_flat = torch.cat([g.view(-1) for g in G2])
    cos_sim = F.cosine_similarity(G1_flat, G2_flat, dim=0)
    cos_distance = 1 - cos_sim
    if cos_distance > cos_distance_thresh:
        return None, cos_distance.item()
    return [(g1 + g2) / 2 for g1, g2 in zip(G1, G2)], cos_distance.item()


def compute_per_sample_accuracy(input_ids, labels, logits):
    batch_size = input_ids.size(0)
    correct = 0
    total = 0
    incorrect_samples = []

    for i in range(batch_size):
        input_id = input_ids[i]
        label = labels[i]
        logit = logits[i]

        # Find the position of '=' token
        equal_token_id = tokenizer.convert_tokens_to_ids('=')
        equal_pos = (input_id == equal_token_id).nonzero(as_tuple=True)

        if len(equal_pos[0]) > 0:
            equal_pos = equal_pos[0].item()
            # Get ground truth tokens after '='
            ground_truth_tokens = label[equal_pos + 1:]
            ground_truth_tokens = ground_truth_tokens[ground_truth_tokens != -100]

            # Get predicted tokens after '='
            predicted_logits = logit[equal_pos:]
            predicted_tokens = predicted_logits.argmax(dim=-1)
            predicted_tokens = predicted_tokens[:len(ground_truth_tokens)]

            ground_truth_text = tokenizer.decode(ground_truth_tokens, skip_special_tokens=True).strip()
            predicted_text = tokenizer.decode(predicted_tokens, skip_special_tokens=True).strip()
            
            # Remove spaces if your tokenization creates them
            ground_truth_text = ground_truth_text.replace(' ', '')
            predicted_text = predicted_text.replace(' ', '')
            input_text = tokenizer.decode(input_id, skip_special_tokens=True)
                
            # Now parse
            # ground_truth_value = float(ground_truth_text)
            # predicted_value = float(predicted_text)
            try:
                # Now parse
                ground_truth_value = float(ground_truth_text)
                predicted_value = float(predicted_text)
            
            except:
                print(f"could not parse. ground_truth_text:{ground_truth_text} predicted_text:{predicted_text}")
                continue  # Skip if cannot parse
            # Compare the sequences
            if torch.equal(predicted_tokens, ground_truth_tokens):
                correct += 1
                if np.random.rand()>.99:
                    print({
                    'input': input_text,
                    'prediction': predicted_text,
                    'ground_truth': ground_truth_text,
                    })
            else:
                # Collect incorrect sample
                
                incorrect_samples.append({
                    'input': input_text,
                    'prediction': predicted_text,
                    'ground_truth': ground_truth_text,
                })
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, incorrect_samples


def get_best_gpu():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise ValueError("No GPUs available.")

    max_free_mem = 0
    best_gpu = None
    for i in range(num_gpus):
        # Set device and clear cache to get accurate readings
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()

        # Get free and total memory (in bytes)
        free_mem, total_mem = torch.cuda.mem_get_info(i)

        # Convert to MB for easier interpretation (optional)
        free_mem_mb = free_mem / (1024 ** 2)

        # Select GPU with the most free memory
        if free_mem > max_free_mem or (free_mem == max_free_mem and (best_gpu is None or i < best_gpu)):
            max_free_mem = free_mem
            best_gpu = i

    if best_gpu is None:
        best_gpu = 0  # Default to GPU 0 if none found

    return best_gpu
    
# Dataset for addition problems
class AdditionDataset(Dataset):
    def __init__(self, domain_start, domain_end, size=10000, space_tokens=False):
        self.data = []
        for _ in range(size):
            a = random.randint(domain_start, domain_end)
            b = random.randint(domain_start, domain_end)
            if space_tokens:
                a_str = ' '.join(list(str(a)))    # "12" -> "1 2"
                b_str = ' '.join(list(str(b)))    # "3" -> "3"
                res_str = ' '.join(list(str(a+b))) # "15" -> "1 5"
                
                # Now ensure that '+' and '=' are also separated by spaces:
                self.data.append(f"{a_str} + {b_str} = {res_str}")
            else:
                
                self.data.append(f"{a}+{b}={a+b}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Collate function to handle padding and masking
def collate_fn(batch):
    inputs = tokenizer(batch, return_tensors="pt", padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    labels = input_ids.clone()
    equal_token_id = tokenizer.convert_tokens_to_ids('=')
    for i, input_id in enumerate(input_ids):
        equal_pos = (input_id == equal_token_id).nonzero(as_tuple=True)[0]
        if len(equal_pos) > 0:
            equal_pos = equal_pos[0]
            labels[i, :equal_pos + 1] = -100
        else:
            labels[i, :] = -100  # Mask all tokens if '=' not found
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

# Function to perturb parameters for MeZO
def perturb_parameters(model, epsilon, seed):
    torch.manual_seed(seed)
    for param in model.parameters():
        z = torch.randn_like(param.data)
        param.data.add_(epsilon * z)

# Function to compute KLSparsity loss
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

# Hook to store activations for KLSparsity
def store_activations(module, input, output):
    module.activations = output

# Function to compute MSE on parsed answers after '='
def compute_mse_on_parsed_answers(input_ids, labels, logits):
    batch_size = input_ids.size(0)
    total_mse = 0.0
    count = 0
    for i in range(batch_size):
        input_id = input_ids[i]
        logit = logits[i]
        equal_token_id = tokenizer.convert_tokens_to_ids('=')
        equal_pos = (input_id == equal_token_id).nonzero(as_tuple=True)
        if len(equal_pos[0]) > 0:
            equal_pos = equal_pos[0].item()
            ground_truth_tokens = input_id[equal_pos + 1:]
            # ground_truth_tokens = label[equal_pos + 1:]
            ground_truth_tokens = ground_truth_tokens[ground_truth_tokens != -100]

            # Get predicted tokens after '='
            predicted_logits = logit[equal_pos:]
            predicted_tokens = predicted_logits.argmax(dim=-1)
            predicted_tokens = predicted_tokens[:len(ground_truth_tokens)]

            ground_truth_text = tokenizer.decode(ground_truth_tokens, skip_special_tokens=True).strip()
            predicted_text = tokenizer.decode(predicted_tokens, skip_special_tokens=True).strip()
            
            # Remove spaces if your tokenization creates them
            ground_truth_text = ground_truth_text.replace(' ', '')
            predicted_text = predicted_text.replace(' ', '')
            
            
            try:
                # Now parse
                ground_truth_value = float(ground_truth_text)
                predicted_value = float(predicted_text)
            
            except:
                # print(f"could not parse. ground_truth_text:{ground_truth_text} predicted_text:{predicted_text}")
                continue 
                
            mse = (predicted_value - ground_truth_value) ** 2
            total_mse += mse
            count += 1
    if count > 0:
        return total_mse / count
    else:
        return None

# def validate(model, tokenizer, val_dataloader, device, optimizer, args):
#     model.eval()
#     val_loss_total = 0.0
#     val_base_loss_total = 0.0
#     val_kl_loss_total = 0.0
#     val_weight_decay_loss = 0.0
#     val_correct = 0
#     val_total = 0
#     val_mse_total = 0.0
#     val_count = 0
#     val_incorrect_samples = []

#     # For generation, decide on parameters like max_new_tokens
#     # Assuming we know sums won't exceed 3 digits (safe margin)
#     max_new_tokens = 8

#     with torch.no_grad():
#         for val_batch in val_dataloader:
#             input_ids = val_batch['input_ids'].to(device)
#             attention_mask = val_batch['attention_mask'].to(device)
#             labels = val_batch['labels'].to(device)

#             # Compute losses for logging (same as before)
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             base_loss = outputs.loss
#             if args.klsparsity:
#                 kl_loss = compute_klsparsity_loss(model, args.klsparsity_pi)
#             else:
#                 kl_loss = 0.0
#             total_loss = base_loss + args.klsparsity_lambda * kl_loss

#             # Compute weight decay loss
#             weight_decay_loss = 0.0
#             for group in optimizer.param_groups:
#                 for param in group['params']:
#                     if param.requires_grad:
#                         weight_decay_loss += torch.sum(param.data ** 2)
#             weight_decay_loss *= optimizer.param_groups[0]['weight_decay']

#             val_loss_total += total_loss.item()
#             val_base_loss_total += base_loss.item()
#             val_kl_loss_total += kl_loss.item() if args.klsparsity else 0.0
#             val_weight_decay_loss += weight_decay_loss.item()

#             # Now perform generation for accuracy calculation
#             batch_size = input_ids.size(0)
#             batch_predictions = []
#             batch_ground_truths = []
#             batch_inputs = []

#             # Identify positions of '=' and slice input to generate from there
#             equal_token_id = tokenizer.convert_tokens_to_ids('=')

#             for i in range(batch_size):
#                 seq = input_ids[i]
#                 # Find '=' position
#                 equal_pos = (seq == equal_token_id).nonzero(as_tuple=True)
#                 if len(equal_pos[0]) == 0:
#                     # No '=' found, skip
#                     continue
#                 equal_pos = equal_pos[0].item()

#                 # The prefix is everything up to and including '='
#                 prefix = seq[:equal_pos+1].unsqueeze(0)  # shape: (1, seq_len)
#                 # Generate tokens after '='
#                 generated = model.generate(
#                     prefix,
#                     max_new_tokens=max_new_tokens,
#                     pad_token_id=tokenizer.pad_token_id,
#                     eos_token_id=tokenizer.eos_token_id,
#                     do_sample=False  # Greedy for this simple task
#                 )

#                 # Extract the newly generated tokens (after prefix)
#                 new_tokens = generated[0, equal_pos+1:]
#                 predicted_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
#                 predicted_text = predicted_text.replace(' ', '')

#                 # Ground truth tokens
#                 ground_truth_tokens = labels[i, equal_pos+1:]
#                 ground_truth_tokens = ground_truth_tokens[ground_truth_tokens != -100]
#                 ground_truth_text = tokenizer.decode(ground_truth_tokens, skip_special_tokens=True).strip()
#                 ground_truth_text = ground_truth_text.replace(' ', '')

#                 input_text = tokenizer.decode(seq, skip_special_tokens=True)

#                 # Store for accuracy computation
#                 batch_predictions.append(predicted_text)
#                 batch_ground_truths.append(ground_truth_text)
#                 batch_inputs.append(input_text)

#             # Compute per-sample accuracy and track incorrect samples
#             # We'll consider a sample "correct" if the predicted_text matches exactly with ground_truth_text
#             sample_correct = 0
#             sample_count = 0
#             local_incorrect = []
#             for inp, pred, gt in zip(batch_inputs, batch_predictions, batch_ground_truths):
#                 if gt and pred:
#                     try:
#                         predicted_value = float(pred)
#                         ground_truth_value = float(gt)
#                     except:
#                         # If can't parse as float, skip
#                         continue

#                     # Check exact token equality by re-tokenizing
#                     # Or just compare strings since we stripped spaces and both are digits
#                     if pred == gt:
#                         sample_correct += 1
#                     else:
#                         local_incorrect.append({
#                             'input': inp,
#                             'prediction': pred,
#                             'ground_truth': gt
#                         })
#                     sample_count += 1

#             if sample_count > 0:
#                 batch_accuracy = sample_correct / sample_count
#             else:
#                 batch_accuracy = 0.0

#             val_correct += sample_correct
#             val_total += sample_count
#             val_incorrect_samples.extend(local_incorrect)

#             # Compute MSE for those that parsed correctly
#             for pred, gt in zip(batch_predictions, batch_ground_truths):
#                 if pred and gt:
#                     try:
#                         p_val = float(pred)
#                         g_val = float(gt)
#                         mse = (p_val - g_val)**2
#                         val_mse_total += mse
#                         val_count += 1
#                     except:
#                         pass

#     val_acc_per_sample = val_correct / val_total if val_total > 0 else 0.0
#     val_loss_avg = val_loss_total / len(val_dataloader)
#     val_base_loss_avg = val_base_loss_total / len(val_dataloader)
#     val_kl_loss_avg = val_kl_loss_total / len(val_dataloader)
#     val_weight_decay_loss_avg = val_weight_decay_loss / len(val_dataloader)
#     val_mse_avg = val_mse_total / val_count if val_count > 0 else None

#     print(f"Validation Accuracy: {val_acc_per_sample*100:.2f}%")
#     print("Incorrect Samples:")
#     for sample in val_incorrect_samples:
#         print(f"Input: {tokenizer.tokenize(sample['input'])}")
#         print(f"Prediction: {tokenizer.tokenize(sample['prediction'])}")
#         print(f"Ground Truth: {tokenizer.tokenize(sample['ground_truth'])}")
#         print("-" * 40)

#     return {
#         'val_loss': val_loss_avg,
#         'val_base_loss': val_base_loss_avg,
#         'val_kl_loss': val_kl_loss_avg,
#         'val_weight_decay_loss': val_weight_decay_loss_avg,
#         'val_acc_per_sample': val_acc_per_sample,
#         'val_mse': val_mse_avg,
#     }
    
# Training function
def train(args):
    global tokenizer  # Needed for collate_fn
    # Initialize wandb
    wandb.init(project="Solving addition (2)", config=vars(args))

    # Set device
    if torch.cuda.is_available():
        device_id = get_best_gpu()
        device = torch.device(f'cuda:{device_id}')
        print(f"Using device {device_id}")
    else:
        device = torch.device('cpu')

    # Load tokenizer and model
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

    # Register hooks for KLSparsity
    if args.klsparsity:
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.register_forward_hook(store_activations)

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_iters,
    )

    # Initialize domains
    train_domain1 = 0
    train_domain2 = 10

    
    if args.raw_weights:
        print("initing weights back to raw")
        model.init_weights()  # Initialize with raw weights


     
    if args.limited_tokens:
        print("initing the tokenizer to the minimal set for addition")
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
    
        # Define your tokens
        special_tokens = ["<bos>", "<eos>", "<sep>", "<pad>", "<unk>"]
        tokens = [str(d) for d in range(10)] + ["+", "="]
    
        # Combine special and normal tokens into a single vocabulary
        all_tokens = special_tokens + tokens
    
        # Create a vocabulary dictionary
        vocab = {tok: i for i, tok in enumerate(all_tokens)}
    
        # Create a WordLevel tokenizer with the limited vocab and specify unk_token
        tokenizer_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        # Add a Whitespace pre-tokenizer to split on spaces
        tokenizer_obj.pre_tokenizer = Whitespace()
    
        # Initialize PreTrainedTokenizerFast with our tokenizer object and special tokens
        new_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj,
            bos_token="<bos>",
            eos_token="<eos>",
            sep_token="<sep>",
            pad_token="<pad>",
            unk_token="<unk>"
        )
    
        # Assign the new tokenizer globally
        tokenizer = new_tokenizer
    
        # Resize the model embeddings for the new vocabulary
        model.resize_token_embeddings(len(tokenizer))

    while True:
        # Create datasets
        train_dataset = AdditionDataset(train_domain1, train_domain2, space_tokens=args.limited_tokens)
        val_dataset = AdditionDataset(train_domain2 + 1, train_domain2 + 10, space_tokens=args.limited_tokens)

        print("train_dataset: ",train_dataset)
        print("val_dataset: ",val_dataset)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.micro_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.micro_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        patience_counter = 0
        train_iters = 0
        train_correct = 0
        train_total = 0
        train_loss_total = 0.0

        while True:
            model.train()
            for batch in train_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                if args.gaf:
                    # Split batch into two micro-batches
                    batch_size = input_ids.size(0)
                    if batch_size < 2:
                        continue  # Skip if not enough data
                    indices = torch.randperm(batch_size)
                    mid_point = batch_size // 2
                    batch1_indices = indices[:mid_point]
                    batch2_indices = indices[mid_point:]

                    # First micro-batch
                    input_ids1 = input_ids[batch1_indices]
                    attention_mask1 = attention_mask[batch1_indices]
                    labels1 = labels[batch1_indices]
                    outputs1 = model(
                        input_ids=input_ids1,
                        attention_mask=attention_mask1,
                        labels=labels1,
                    )
                    loss1 = outputs1.loss
                    optimizer.zero_grad()
                    loss1.backward()
                    G1 = [p.grad.clone() for p in model.parameters()]
                    optimizer.zero_grad()

                    # Second micro-batch
                    input_ids2 = input_ids[batch2_indices]
                    attention_mask2 = attention_mask[batch2_indices]
                    labels2 = labels[batch2_indices]
                    outputs2 = model(
                        input_ids=input_ids2,
                        attention_mask=attention_mask2,
                        labels=labels2,
                    )
                    loss2 = outputs2.loss
                    optimizer.zero_grad()
                    loss2.backward()
                    G2 = [p.grad.clone() for p in model.parameters()]
                    optimizer.zero_grad()

                    # Filter gradients
                    filtered_grad, cosine_distance = filter_gradients(
                        G1, G2, args.gaf_tau
                    )
                    
                    
                    # Log individual losses
                    loss_value = (loss1.item() + loss2.item()) / 2
                    base_loss = loss_value
                    kl_loss = 0
                    outputs = outputs1
                    labels = labels1
                    input_ids = input_ids1
                    
                    if filtered_grad is not None:
                        # Apply filtered gradients
                        with torch.no_grad():
                            for param, grad in zip(model.parameters(), filtered_grad):
                                param.grad = grad
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        print(f"cosine_distance={cosine_distance}")
                    else:
                        print(f"skipping batch update: cosine_distance={cosine_distance}")
                        
                    # Log cosine distance
                    print({'cosine_distance': cosine_distance})
                    wandb.log({'cosine_distance': cosine_distance})
                                    
                elif args.mezo:
                    # MeZO optimization
                    seed = random.randint(0, int(1e6))
                
                    # Perturb parameters positively
                    perturb_parameters(model, args.mezo_epsilon, seed)
                
                    # Compute positive loss
                    outputs_pos = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    base_loss_pos = outputs_pos.loss.item()
                
                    # Compute KLSparsity loss if enabled
                    if args.klsparsity:
                        kl_loss_pos = compute_klsparsity_loss(model, args.klsparsity_pi).item()
                    else:
                        kl_loss_pos = 0.0
                
                    total_loss_pos = base_loss_pos + args.klsparsity_lambda * kl_loss_pos
                
                    # Perturb parameters negatively
                    perturb_parameters(model, -2 * args.mezo_epsilon, seed)
                
                    # Compute negative loss
                    outputs_neg = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    base_loss_neg = outputs_neg.loss.item()
                
                    if args.klsparsity:
                        kl_loss_neg = compute_klsparsity_loss(model, args.klsparsity_pi).item()
                    else:
                        kl_loss_neg = 0.0
                
                    total_loss_neg = base_loss_neg + args.klsparsity_lambda * kl_loss_neg
                
                    # Reset parameters to original
                    perturb_parameters(model, args.mezo_epsilon, seed)
                
                    # Compute projected gradient
                    projected_grad_scalar = (total_loss_pos - total_loss_neg) / (2 * args.mezo_epsilon)
                
                    # Reset random number generator
                    torch.manual_seed(seed)
                
                    # Generate the same random directions z
                    for param in model.parameters():
                        if param.requires_grad:
                            z = torch.randn_like(param.data)
                            # Assign the pseudo-gradient
                            if param.grad is not None:
                                param.grad.zero_()
                            else:
                                param.grad = torch.zeros_like(param.data)
                            param.grad.add_(projected_grad_scalar * z)
                
                    # Perform optimizer step
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                    # Log individual losses
                    base_loss = (base_loss_pos + base_loss_neg) / 2
                    kl_loss = (kl_loss_pos + kl_loss_neg) / 2 if args.klsparsity else 0.0
                    loss_value = (total_loss_pos + total_loss_neg) / 2
                    outputs = outputs_pos


                else:
                    # Standard backpropagation
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    base_loss = outputs.loss

                    # Compute KLSparsity loss if enabled
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
                                    # Create noise directly on the correct device
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

                # Compute per-token accuracy
                with torch.no_grad():
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_mask = shift_labels != -100
                    pred_tokens = shift_logits.argmax(dim=-1)
                    correct = (pred_tokens == shift_labels) & loss_mask
                    num_correct = correct.sum().item()
                    num_total = loss_mask.sum().item()
                    train_correct += num_correct
                    train_total += num_total
                    
                    # MSE on parsed answers
                    mse = compute_mse_on_parsed_answers(input_ids, labels, logits)
                    if mse is not None:
                        wandb.log({'train_mse': mse})

                # Compute weight decay loss for logging
                weight_decay_loss = 0.0
                for group in optimizer.param_groups:
                    for param in group['params']:
                        if param.requires_grad:
                            weight_decay_loss += torch.sum(param.data ** 2)
                
                weight_decay_loss *= optimizer.param_groups[0]['weight_decay']

                train_loss_total += loss_value + weight_decay_loss
                train_iters += 1

                # Logging metrics
                train_acc = train_correct / train_total if train_total > 0 else 0.0
                 
                batch_accuracy, batch_incorrect_samples = compute_per_sample_accuracy(
                            input_ids, labels, logits
                        )
                
                message = {
                    'train_loss': loss_value,
                    'base_loss': base_loss,
                    'kl_loss': kl_loss,
                    'weight_decay_loss': weight_decay_loss.item(),
                    'train_acc': train_correct / num_total,
                    'lr': scheduler.get_last_lr()[0],
                    'weight_decay': optimizer.param_groups[0]['weight_decay'],
                    'train_iters': train_iters,
                    'train_acc_per_sample':batch_accuracy
                }
                
                print(message)
                wandb.log(message)
                if train_iters % 100 == 0:  # Adjust the frequency as needed

                    # Output the incorrect samples
                    print(f"Train Sample Accuracy: {batch_accuracy*100:.2f}%")
                    print("Incorrect Samples:")
                    for sample in batch_incorrect_samples:
                       
                        print(f"Input: {tokenizer.tokenize(sample['input'])}")
                        print(sample)
                        print("-" * 40)

                if batch_accuracy >= 0.98:
                    patience_counter += 1
                else:
                    patience_counter = 0

                if patience_counter >= args.training_at_100_patience:
                    break

            if patience_counter >= args.training_at_100_patience:
                break

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_base_loss_total = 0.0
        val_kl_loss_total = 0.0
        val_weight_decay_loss = 0.0
        val_correct = 0
        val_total = 0
        val_mse_total = 0.0
        val_count = 0
        val_incorrect_samples = []
        print("STARTING VALIDATION")
        with torch.no_grad():
            for val_batch in val_dataloader:
                if val_total>100 and val_count<30:
                    break
                input_ids = val_batch['input_ids'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                labels = val_batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                base_loss = outputs.loss

                # Compute KLSparsity loss if enabled
                if args.klsparsity:
                    kl_loss = compute_klsparsity_loss(model, args.klsparsity_pi)
                else:
                    kl_loss = 0.0

                total_loss = base_loss + args.klsparsity_lambda * kl_loss

                # Compute weight decay loss
                weight_decay_loss = 0.0
                for group in optimizer.param_groups:
                    for param in group['params']:
                        if param.requires_grad:
                            weight_decay_loss += torch.sum(param.data ** 2)
                weight_decay_loss *= optimizer.param_groups[0]['weight_decay']

                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_mask = shift_labels != -100
                pred_tokens = shift_logits.argmax(dim=-1)
                correct = (pred_tokens == shift_labels) & loss_mask
                num_correct = correct.sum().item()
                num_total = loss_mask.sum().item()
                
                val_loss_total += total_loss.item()
                val_base_loss_total += base_loss.item()
                val_kl_loss_total += kl_loss.item() if args.klsparsity else 0.0
                val_weight_decay_loss += weight_decay_loss.item()
                batch_accuracy, batch_incorrect_samples = compute_per_sample_accuracy(
                    input_ids, labels, logits
                )
                val_correct += batch_accuracy * input_ids.size(0)
                val_total += input_ids.size(0)
                val_incorrect_samples.extend(batch_incorrect_samples)

                # MSE on parsed answers
                mse = compute_mse_on_parsed_answers(input_ids, labels, logits)
                if mse is not None:
                    val_mse_total += mse
                    val_count += 1

        val_acc_per_sample = val_correct / val_total if val_total > 0 else 0.0
        val_loss_avg = val_loss_total / len(val_dataloader)
        val_base_loss_avg = val_base_loss_total / len(val_dataloader)
        val_kl_loss_avg = val_kl_loss_total / len(val_dataloader)
        val_weight_decay_loss_avg = val_weight_decay_loss / len(val_dataloader)
        val_mse_avg = val_mse_total / val_count if val_count > 0 else None

        # Output the incorrect samples
        print(f"Validation Accuracy: {val_acc_per_sample*100:.2f}%")
        print("Incorrect Samples:")
        # import pdb
        # pdb.set_trace()
        for sample in val_incorrect_samples:
            print(sample)
            print("-" * 40)

        # Update train domain
        train_domain2 += 10
        
        # Update weight decay if schedule is enabled
        if args.weight_decay_schedule:
            optimizer.param_groups[0]['weight_decay'] *= args.weight_decay_k
        # Log validation metrics
        message = {
            'val_loss': val_loss_avg,
            'val_base_loss': val_base_loss_avg,
            'val_kl_loss': val_kl_loss_avg,
            'val_weight_decay_loss': val_weight_decay_loss_avg,
            'val_acc_per_sample': val_acc_per_sample,
            'val_mse': val_mse_avg,
            'train_iters_to_100_acc': train_iters,
            'updated_weight_decay': optimizer.param_groups[0]['weight_decay'],
            'train_domain2':train_domain2,
            
        }
        print(message)
        wandb.log(message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LLM to learn addition")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="Micro batch size")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--training_at_100_patience", type=int, default=100, help="Patience after reaching 100% train accuracy")
    parser.add_argument("--max_iters", type=int, default=100000, help="Maximum number of iterations per domain")
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
    parser.add_argument("--raw_weights", action="store_true", help="Initialize the model with raw weights (default: True)")
    parser.add_argument("--limited_tokens", action="store_true", help="Initialize the model with raw weights (default: True)")
    args = parser.parse_args()
    train(args)
