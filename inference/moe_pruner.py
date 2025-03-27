import os
import json
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file


def get_expert_mapping(expert_scores, num_experts_to_prune):
    """
    Creates a mapping from old expert indices to new consecutive indices after pruning.
    
    Args:
        expert_scores (list): List of expert scores for a layer
        num_experts_to_prune (int): Number of experts to remove
        
    Returns:
        dict: Mapping from old expert indices to new consecutive indices
        list: List of experts to keep (sorted by score)
    """
    # Create (score, index) pairs and sort by score in descending order
    expert_pairs = [(score, idx) for idx, score in enumerate(expert_scores)]
    expert_pairs.sort(reverse=True)
    
    # Keep the top scoring experts
    num_experts = len(expert_scores)
    num_experts_to_keep = num_experts - num_experts_to_prune
    kept_experts = [idx for _, idx in expert_pairs[:num_experts_to_keep]]
    kept_experts.sort()  # Sort by index for consistent mapping
    
    # Create mapping from old to new consecutive indices
    expert_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_experts)}
    
    return expert_mapping, kept_experts


def copy_additional_files(input_path, output_path):
    """
    Copies tokenizer, config, and other necessary files to the output path.
    
    Args:
        input_path (str): Input model directory path
        output_path (str): Output model directory path
    """
    patterns = [
        "token*",
        "config*",
        "generation_config.json",
        "modeling_deepseek.py"
    ]
    
    for pattern in patterns:
        files = glob(os.path.join(input_path, pattern))
        for file in files:
            shutil.copy2(file, output_path)


def update_config(output_path, num_experts_to_keep):
    """
    Updates the config.json file with the new number of experts.
    
    Args:
        output_path (str): Output model directory path
        num_experts_to_keep (int): New number of experts after pruning
    """
    config_file = os.path.join(output_path, "config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Update number of routed experts
        if 'n_routed_experts' in config:
            config['n_routed_experts'] = num_experts_to_keep
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)


def main(input_hf_path, input_expert_scores, prune_expert, output_hf_path):
    """
    Prunes experts from MoE layers based on activation scores.
    """
    os.makedirs(output_hf_path, exist_ok=True)
    
    # Load expert scores and determine which experts to keep for each layer
    with open(input_expert_scores, 'r') as f:
        expert_scores_dict = json.load(f)
    
    # Get layer IDs and create expert mappings for each layer
    layer_mappings = {}
    layer_ids = [int(key.split('_')[1]) for key in expert_scores_dict if key.startswith('layer_')]
    layer_ids.sort()  # Sort to ensure consistent processing
    
    # Find the first valid MoE layer to use as a reference for calculations
    reference_layer_id = layer_ids[0] if layer_ids else None
    reference_layer = f"layer_{reference_layer_id}" if reference_layer_id is not None else None
    
    if not reference_layer:
        raise ValueError("No valid layer found in the expert scores file")
    
    for layer_id in layer_ids:
        layer_key = f"layer_{layer_id}"
        # Check if the format is the new nested format or old format
        if isinstance(expert_scores_dict[layer_key], dict) and "scores" in expert_scores_dict[layer_key]:
            # New format: scores are under "scores" key
            scores = expert_scores_dict[layer_key]["scores"]
        else:
            # Old format: scores are directly in the layer key
            scores = expert_scores_dict[layer_key]
        mapping, kept_experts = get_expert_mapping(scores, prune_expert)
        layer_mappings[layer_id] = (mapping, kept_experts)
    
    # Process each safetensor file
    safetensor_files = list(glob(os.path.join(input_hf_path, "*.safetensors")))
    safetensor_files.sort()
    weight_map = {}  # Start with empty weight map and only add tensors we keep
    
    os.makedirs(output_hf_path, exist_ok=True)
    
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cpu")
        new_state_dict = {}
        
        # Process all tensors
        for key, tensor in current_state_dict.items():
            # Check if this is an expert-related tensor
            is_expert_tensor = False
            current_layer_id = None
            for layer_id in layer_mappings:
                if f"layers.{layer_id}.mlp" in key:
                    is_expert_tensor = True
                    current_layer_id = layer_id
                    break
            
            if not is_expert_tensor:
                # Keep non-expert tensors as is
                new_state_dict[key] = tensor
                weight_map[key] = file_name
                continue
                
            expert_mapping, kept_experts = layer_mappings[current_layer_id]
            
            if "gate.weight" in key:
                # Create new gate weights tensor with correct dimensions
                old_M, N = tensor.size()
                new_M = len(kept_experts)
                new_tensor = torch.empty((new_M, N), dtype=tensor.dtype, device="cpu")
                for new_idx, old_idx in enumerate(kept_experts):
                    new_tensor[new_idx] = tensor[old_idx]
                new_state_dict[key] = new_tensor
                weight_map[key] = file_name
            elif "gate.e_score_correction_bias" in key:
                # Create new gate bias tensor with correct dimensions
                old_M = tensor.size(0)
                new_M = len(kept_experts)
                new_tensor = torch.empty(new_M, dtype=tensor.dtype, device="cpu")
                for new_idx, old_idx in enumerate(kept_experts):
                    new_tensor[new_idx] = tensor[old_idx]
                new_state_dict[key] = new_tensor
                weight_map[key] = file_name
            elif "experts" in key and "shared_experts" not in key:
                # Handle expert weights - only keep specified experts
                try:
                    expert_idx = int(key.split("experts.")[1].split(".")[0])
                    if expert_idx in expert_mapping:
                        # Map to new consecutive index
                        new_idx = expert_mapping[expert_idx]
                        new_key = key.replace(f"experts.{expert_idx}", f"experts.{new_idx}")
                        new_state_dict[new_key] = tensor
                        weight_map[new_key] = file_name
                except (IndexError, ValueError):
                    continue
            elif "shared_experts" in key:
                # Keep shared expert weights as is
                new_state_dict[key] = tensor
                weight_map[key] = file_name
        
        # Save pruned weights
        new_safetensor_file = os.path.join(output_hf_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        
        # Clear memory
        del current_state_dict
        del new_state_dict
        torch.cuda.empty_cache()
    
    # Save updated model index with only kept tensors
    new_model_index_file = os.path.join(output_hf_path, "model.safetensors.index.json")
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
    
    # Copy additional files
    copy_additional_files(input_hf_path, output_hf_path)
    
    # Get expert count from reference layer
    if isinstance(expert_scores_dict[reference_layer], dict) and "scores" in expert_scores_dict[reference_layer]:
        original_experts = len(expert_scores_dict[reference_layer]["scores"])
    else:
        original_experts = len(expert_scores_dict[reference_layer])
    
    num_experts_to_keep = original_experts - prune_expert
    update_config(output_hf_path, num_experts_to_keep)
    
    # Print summary
    print(f"\nPruning summary:")
    print(f"Original experts: {original_experts}")
    print(f"Experts pruned: {prune_expert}")
    print(f"Experts kept: {num_experts_to_keep}")
    print(f"Expert mapping for {reference_layer}: {layer_mappings[reference_layer_id][0]}")
    print(f"Output path: {output_hf_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-hf-path", type=str, required=True)
    parser.add_argument("--input-expert-scores", type=str, required=True)
    parser.add_argument("--prune-expert", type=int, required=True)
    parser.add_argument("--output-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_hf_path, args.input_expert_scores, args.prune_expert, args.output_hf_path) 