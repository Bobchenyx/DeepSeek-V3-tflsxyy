import os
import json
from argparse import ArgumentParser
from typing import List
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from safetensors.torch import load_model
from datasets import load_dataset, Dataset

from model import Transformer, ModelArgs, MoE


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        assert len(prompts) <= args.max_batch_size, f"Number of prompts exceeds maximum batch size ({args.max_batch_size})"
        prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()


def track_expert_scores(ckpt_path: str, config_path: str):
    # Initialize distributed setup
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)

    # Load model and tokenizer
    print("Loading model...")
    with open(config_path) as f:
        args = ModelArgs(**json.load(f))
    batch_size = args.max_batch_size  # Use batch size from config
    print(f"Using batch size from config: {batch_size}")
    
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    model.eval()
    print(f"Model loaded successfully. Processing with world_size={world_size}")

    # Initialize expert score tracking
    expert_scores = {}
    expert_activations = {}  # Track number of times each expert is activated
    
    # Initialize scores for each MoE layer
    for layer_id, block in enumerate(model.layers):
        if isinstance(block.ffn, MoE):
            n_local_experts = block.ffn.n_local_experts
            expert_scores[layer_id] = torch.zeros(n_local_experts, device='cuda', dtype=torch.float32)
            expert_activations[layer_id] = torch.zeros(n_local_experts, device='cuda', dtype=torch.long)

    def hook_fn(layer_id):
        def _hook(module, input, output):
            weights, indices = output
            # Process weights and indices immediately
            for expert_idx in range(model.layers[layer_id].ffn.n_local_experts):
                mask = (indices == (rank * model.layers[layer_id].ffn.n_local_experts + expert_idx))
                expert_scores[layer_id][expert_idx] += weights[mask].to(torch.float32).sum()
                expert_activations[layer_id][expert_idx] += mask.sum()
        return _hook

    # Register hooks for each MoE layer's gate
    hooks = []
    for layer_id, block in enumerate(model.layers):
        if isinstance(block.ffn, MoE):
            hooks.append(block.ffn.gate.register_forward_hook(hook_fn(layer_id)))

    # Comment out WikiText dataset loading
    """
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    print(f"Loaded WikiText-2 test dataset with {len(dataset)} examples")

    # Tokenize the entire text at once
    all_tokens = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")["input_ids"][0]
    print(f"Total tokens in dataset: {len(all_tokens)}")
    """
    
    # Load MMLU dataset - both test and validation splits
    print("Loading MMLU datasets...")
    # Original code: test_dataset = load_dataset("cais/mmlu", "all", split="test")
    # Load from local parquet file instead
    print("Loading from local parquet file...")
    test_dataset = Dataset.from_parquet("/root/DeepSeek-V3/inference/test-00000-of-00001.parquet")
    print(f"Loaded MMLU test dataset with {len(test_dataset)} examples")

    # # Combine datasets and limit to 30000 samples
    # all_examples = []
    # for example in test_dataset:
    #     all_examples.append(example)
    # for example in val_dataset:
    #     all_examples.append(example)
    # for example in train_dataset:
    #     all_examples.append(example)
    
    # # Limit to first 30000 samples
    # all_examples = all_examples[:30000]
    # print(f"Using first 30000 examples from combined dataset")
    
    # Extract text from MMLU dataset
    all_texts = []
    for example in test_dataset:
        question = example['question']
        choices = example['choices']
        subject = example['subject']
        # Format as a text prompt
        prompt = f"Subject: {subject}\nQuestion: {question}\nChoices:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        all_texts.append(prompt)
    
    # Tokenize all prompts
    print("Tokenizing MMLU prompts...")
    all_token_lists = [tokenizer(text, return_tensors="pt")["input_ids"][0] for text in all_texts]
    
    # Concatenate all tokens
    all_tokens = torch.cat(all_token_lists)
    print(f"Total tokens in MMLU dataset: {len(all_tokens)}")

    # Ensure we don't exceed max_seq_len
    max_seq_len = min(model.max_seq_len, 16384)  # Use 16384 as a reasonable default max length
    print(f"Using sequence length: {max_seq_len}")

    total_loss = 0.0
    total_tokens = 0

    # Process batches
    for i in range(0, len(all_tokens), max_seq_len * batch_size):
        # Get batch_size chunks of max_seq_len tokens each
        batch_tokens = []
        for j in range(batch_size):
            start_idx = i + j * max_seq_len
            tokens = all_tokens[start_idx:start_idx + max_seq_len]
            if len(tokens) == 0:  # No more tokens to process
                break
            if len(tokens) < max_seq_len:
                tokens = F.pad(tokens, (0, max_seq_len - len(tokens)), value=tokenizer.pad_token_id)
            batch_tokens.append(tokens)
        
        if not batch_tokens:  # No more tokens to process
            break
            
        # Stack into a batch
        tokens = torch.stack(batch_tokens).cuda()
        
        # Get predictions and targets
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        
        # Forward pass
        with torch.no_grad():
            logits = model(input_tokens)
            # Clear GPU cache after forward pass
            torch.cuda.empty_cache()
        
        # Ensure tensors are contiguous and use reshape
        logits = logits.contiguous()
        target_tokens = target_tokens.contiguous()
        
        # Calculate loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_tokens.reshape(-1),
            ignore_index=tokenizer.pad_token_id,
            reduction='sum'
        )
        
        # Count non-padding tokens
        num_tokens = (target_tokens != tokenizer.pad_token_id).sum().item()
        
        # Accumulate loss and tokens
        total_loss += loss.item()
        total_tokens += num_tokens
        
        # Print progress with current perplexity
        print(f"Processing sequence {i//max_seq_len + 1}, shape: {tokens.shape}, current ppl: {math.exp(loss.item()/num_tokens):.2f}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Gather scores from all processes
    if world_size > 1:
        for layer_id in expert_scores:
            all_scores = [torch.zeros_like(expert_scores[layer_id]) for _ in range(world_size)]
            all_activations = [torch.zeros_like(expert_activations[layer_id]) for _ in range(world_size)]
            dist.all_gather(all_scores, expert_scores[layer_id])
            dist.all_gather(all_activations, expert_activations[layer_id])
            if rank == 0:
                expert_scores[layer_id] = torch.cat(all_scores)
                expert_activations[layer_id] = torch.cat(all_activations)

    # Calculate and print final perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    print(f"\nFinal perplexity: {perplexity:.2f}")
    print(f"Total tokens processed: {total_tokens}")

    # Save expert scores to JSON file
    if rank == 0:
        # Extract config name from config path (e.g., "config_16B" from "configs/config_16B.json")
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        output_file = f"{config_name}_expert_scores.json"
        
        scores_dict = {}
        for layer_id in expert_scores:
            scores_dict[f"layer_{layer_id}"] = {
                "scores": expert_scores[layer_id].cpu().tolist(),
                "activations": expert_activations[layer_id].cpu().tolist()
            }
        
        with open(output_file, 'w') as f:
            json.dump(scores_dict, f, indent=2)
        print(f"Expert scores and activation counts saved to {output_file}")

    if world_size > 1:
        dist.destroy_process_group()


def track_expert_scores_layerwise(ckpt_path: str, config_path: str, prune_expert: int):
    # Initialize distributed setup
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)

    # Load model and tokenizer
    print("Loading model...")
    with open(config_path) as f:
        args = ModelArgs(**json.load(f))
    batch_size = args.max_batch_size  # Use batch size from config
    print(f"Using batch size from config: {batch_size}")
    
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    model.eval()
    print(f"Model loaded successfully. Processing with world_size={world_size}")

    # Load MMLU dataset - both test and validation splits
    print("Loading MMLU datasets...")
    # Original code: test_dataset = load_dataset("cais/mmlu", "all", split="test")
    # Load from local parquet file instead
    print("Loading from local parquet file...")
    test_dataset = Dataset.from_parquet("/root/DeepSeek-V3/inference/test-00000-of-00001.parquet")
    print(f"Loaded MMLU test dataset with {len(test_dataset)} examples")
    
    # Combine datasets and limit to 1000 samples
    all_examples = []
    for example in test_dataset:
        all_examples.append(example)
    
    # Limit to first 10000 samples
    all_examples = all_examples[:10000]
    print(f"Using first 10000 examples from combined dataset")
    
    # Extract text from MMLU dataset
    all_texts = []
    for example in all_examples:
        question = example['question']
        choices = example['choices']
        subject = example['subject']
        # Format as a text prompt
        prompt = f"Subject: {subject}\nQuestion: {question}\nChoices:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        all_texts.append(prompt)
    
    # Tokenize all prompts
    print("Tokenizing MMLU prompts...")
    all_token_lists = [tokenizer(text, return_tensors="pt")["input_ids"][0] for text in all_texts]
    
    # Concatenate all tokens
    all_tokens = torch.cat(all_token_lists)
    print(f"Total tokens in MMLU dataset: {len(all_tokens)}")

    # Ensure we don't exceed max_seq_len
    max_seq_len = min(model.max_seq_len, 16384)  # Use 16384 as a reasonable default max length
    print(f"Using sequence length: {max_seq_len}")
    
    # Prepare batches
    all_batches = []
    for i in range(0, len(all_tokens), max_seq_len * batch_size):
        # Get batch_size chunks of max_seq_len tokens each
        batch_tokens = []
        for j in range(batch_size):
            start_idx = i + j * max_seq_len
            tokens = all_tokens[start_idx:start_idx + max_seq_len]
            if len(tokens) == 0:  # No more tokens to process
                break
            if len(tokens) < max_seq_len:
                tokens = F.pad(tokens, (0, max_seq_len - len(tokens)), value=tokenizer.pad_token_id)
            batch_tokens.append(tokens)
        
        if not batch_tokens:  # No more tokens to process
            break
            
        # Stack into a batch
        tokens = torch.stack(batch_tokens).cuda()
        all_batches.append(tokens)
    
    print(f"Prepared {len(all_batches)} batches")
    
    # Initial embedding and setup
    with torch.no_grad():
        freqs_cis = model.freqs_cis[:max_seq_len]
        mask = None
        if max_seq_len > 1:
            mask = torch.full((max_seq_len, max_seq_len), float("-inf"), device="cuda").triu_(1)
    
    # Store all layer data for final output
    all_layer_data = {}
    
    # Process each layer
    print("\nStarting layer-wise processing...")
    current_inputs = []
    
    # Initial embedding for all batches
    with torch.no_grad():
        for tokens in all_batches:
            current_inputs.append(model.embed(tokens))
    
    # Process each layer
    for layer_idx, layer in enumerate(model.layers):
        print(f"\nProcessing layer {layer_idx}...")
        
        # Skip non-MoE layers
        if not isinstance(layer.ffn, MoE):
            # Just forward through the layer and continue
            with torch.no_grad():
                next_inputs = []
                for current_input in current_inputs:
                    output = layer(current_input, 0, freqs_cis, mask)
                    next_inputs.append(output)
                current_inputs = next_inputs
            continue
        
        # Initialize expert score tracking for current layer
        n_local_experts = layer.ffn.n_local_experts
        layer_scores = torch.zeros(n_local_experts, device='cuda', dtype=torch.float32)
        layer_activations = torch.zeros(n_local_experts, device='cuda', dtype=torch.long)
        
        # Register hook for current layer's gate
        def hook_fn(module, input, output):
            weights, indices = output
            for expert_idx in range(n_local_experts):
                global_idx = rank * n_local_experts + expert_idx
                mask = (indices == global_idx)
                layer_scores[expert_idx] += weights[mask].to(torch.float32).sum()
                layer_activations[expert_idx] += mask.sum()
        
        hook = layer.ffn.gate.register_forward_hook(hook_fn)
        
        # First forward pass through current layer to get expert scores
        with torch.no_grad():
            for current_input in current_inputs:
                _ = layer(current_input, 0, freqs_cis, mask)
        
        # Remove hook after getting scores
        hook.remove()
        
        # Gather scores from all processes
        if world_size > 1:
            all_scores = [torch.zeros_like(layer_scores) for _ in range(world_size)]
            all_activations = [torch.zeros_like(layer_activations) for _ in range(world_size)]
            dist.all_gather(all_scores, layer_scores)
            dist.all_gather(all_activations, layer_activations)
            if rank == 0:
                layer_scores = torch.cat(all_scores)
                layer_activations = torch.cat(all_activations)
        
        # Create expert mapping for pruning
        expert_pairs = [(score.item(), idx) for idx, score in enumerate(layer_scores)]
        expert_pairs.sort(reverse=True)
        num_experts = len(expert_pairs)
        num_experts_to_keep = num_experts - prune_expert
        kept_experts = [idx for _, idx in expert_pairs[:num_experts_to_keep]]
        kept_experts.sort()
        
        # Create mapping from old to new consecutive indices
        expert_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_experts)}
        
        print(f"Layer {layer_idx} pruning map: {expert_mapping}")
        
        # Update expert list - only keep experts on this rank
        new_experts = []
        for global_idx in kept_experts:
            old_rank = global_idx // n_local_experts
            old_local_idx = global_idx % n_local_experts
            if old_rank == rank:
                # Expert was on this GPU, keep it
                new_experts.append(layer.ffn.experts[old_local_idx])
        
        # Update gate weights - only keep rows for kept experts
        old_gate_weight = layer.ffn.gate.weight
        new_gate_weight = old_gate_weight[kept_experts].clone()
        layer.ffn.gate.weight = torch.nn.Parameter(new_gate_weight)
        
        if layer.ffn.gate.bias is not None:
            old_gate_bias = layer.ffn.gate.bias
            new_gate_bias = old_gate_bias[kept_experts].clone()
            layer.ffn.gate.bias = torch.nn.Parameter(new_gate_bias)
        
        # Verify gate weight shape matches number of experts
        assert layer.ffn.gate.weight.size(0) == num_experts_to_keep, \
            f"Gate weight shape {layer.ffn.gate.weight.shape} does not match number of experts {num_experts_to_keep}"
        
        # Update MoE layer parameters
        layer.ffn.experts = torch.nn.ModuleList(new_experts)
        layer.ffn.n_routed_experts = num_experts_to_keep
        layer.ffn.n_local_experts = len(new_experts)
        layer.ffn.n_activated_experts = min(layer.ffn.n_activated_experts, num_experts_to_keep)
        layer.ffn.experts_start_idx = rank * layer.ffn.n_local_experts
        layer.ffn.experts_end_idx = layer.ffn.experts_start_idx + layer.ffn.n_local_experts
        
        # Update gate's topk to match new number of experts
        layer.ffn.gate.topk = min(layer.ffn.n_activated_experts, len(new_experts))
        
        print(f"Rank {rank} now handles {len(new_experts)} experts")
        print(f"Gate topk updated to {layer.ffn.gate.topk}")
        print(f"Gate weight shape: {layer.ffn.gate.weight.shape}")
        print(f"Expert range: {layer.ffn.experts_start_idx} to {layer.ffn.experts_end_idx}")
        
        # Save layer data for final output - immediately move to CPU to save GPU memory
        if rank == 0:
            # Move data to CPU immediately after collecting it
            all_layer_data[f"layer_{layer_idx}"] = {
                "scores": layer_scores.cpu().tolist(),  # Convert to CPU and Python list
                "activations": layer_activations.cpu().tolist(),  # Convert to CPU and Python list
                "pruning_map": expert_mapping,
                "experts_per_rank": [len(new_experts)]  # Save number of experts per rank
            }
            
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        # Second forward pass through pruned layer to get output for next layer
        with torch.no_grad():
            next_inputs = []
            for current_input in current_inputs:
                output = layer(current_input, 0, freqs_cis, mask)
                next_inputs.append(output)
            current_inputs = next_inputs
        
        # Clear memory after processing each layer
        torch.cuda.empty_cache()
        
        print(f"Layer {layer_idx} processed and pruned")
    
    # Write all layer data to a single file at the end
    if rank == 0:
        # No need to convert tensors to CPU since we've already done that for each layer
        
        # Extract config name from config path (e.g., "config_16B" from "configs/config_16B.json")
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        output_file = f"{config_name}_prune_{prune_expert}_expert_scores.json"
        
        with open(output_file, "w") as f:
            json.dump(all_layer_data, f, indent=2)
        print(f"All layer data saved to {output_file}")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.
        --track-expert-scores (bool, optional): Track expert utilization scores on WikiText-2.
        --track-expert-scores-layerwise (bool, optional): Track expert utilization scores layer-wise on WikiText-2.
        --prune-expert (int, optional): Number of experts to prune from each MoE layer. Defaults to 16.

    Raises:
        AssertionError: If neither input-file, interactive mode, track-expert-scores, or track-expert-scores-layerwise is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--track-expert-scores", action="store_true")
    parser.add_argument("--track-expert-scores-layerwise", action="store_true")
    parser.add_argument("--prune-expert", type=int, default=16,
                      help="Number of experts to prune from each MoE layer")
    args = parser.parse_args()
    
    assert args.input_file or args.interactive or args.track_expert_scores or args.track_expert_scores_layerwise, \
        "Either input-file, interactive mode, track-expert-scores, or track-expert-scores-layerwise must be specified"
    
    if args.track_expert_scores_layerwise:
        track_expert_scores_layerwise(args.ckpt_path, args.config, args.prune_expert)
    elif args.track_expert_scores:
        track_expert_scores(args.ckpt_path, args.config)
    else:
        main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)
