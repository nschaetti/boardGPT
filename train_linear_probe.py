
"""
Linear Probe Training Script for BoardGPT
=========================================

This script trains a linear probe on top of a pre-trained GPT model for board games.
It loads a model from safetensors format and sets up the training environment.

The script handles:
- Model loading from safetensors
- Configuration loading from YAML
- Device setup (CPU/CUDA)
- Optimizer configuration

Examples:
---------
Basic usage with default paths:
$ python train_linear_probe.py

Specify custom paths:
$ python train_linear_probe.py --model_file=/path/to/model.safetensors --model_config_file=/path/to/config.json

Use a configuration file:
$ python train_linear_probe.py --config=/path/to/config.yaml
"""

# Imports
from typing import Dict, List, Optional, Tuple
import os
import pickle
import glob
import torch
import yaml
import argparse
import random
import time
import math
from contextlib import nullcontext

from boardGPT.utils import load_safetensors
from boardGPT.models import GPTConfig, GPT
from boardGPT.utils import game_to_board, show_linear_probe_samples
from boardGPT.simulators import create_id_to_move_mapping


def parse_args():
    """
    Parse command line arguments for the linear probe training script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Train a linear probe for a GPT model')
    parser.add_argument('--model_file', type=str, required=True, help='Path to the model safetensors file')
    parser.add_argument(
        '--model_config_file', type=str, required=True, help='Path to the model configuration JSON file'
    )
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to a checkpoint file (.pt) to load weights from. If not specified, a new model will be initialized from scratch.')
    parser.add_argument('--num-iter', type=int, default=None,
                        help='Iteration number to start from when resuming from a checkpoint. Used with --ckpt.')
    return parser.parse_args()
# end def parse_args


def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # end with
        print(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found. Using default configuration.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}")
        return {}
    # end try
# end def load_config


def initialize_optimizer(
        model: GPT,
        learning_rate: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        scaler_enabled: bool,
        device_type: str,
):
    # Initialize a GradScaler for mixed precision training
    # If enabled=False (not using float16), scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    # Set up the optimizer
    optimizer = model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        betas=(beta1, beta2),
        device_type=device_type
    )

    return optimizer, scaler
# end def initialize_optimizer


_train_data: Optional[List[List[int]]] = None
_val_data: Optional[List[List[int]]] = None


def load_game_sequences(
        split: str,
        config: Dict
) -> List[List[int]]:
    """
    Load data from the specified split.

    Args:
        split (str): Split to load
        config (Dict): Configuration dictionary
    """
    global _train_data, _val_data

    print(f"Loading {split} data into memory...")

    # Data dir for the specified split (train or val)
    data_dir = os.path.join(config['data_dir'], split)

    # Pattern for bin files
    pattern = "*.bin"

    # Find all matching bin files
    bin_files = glob.glob(os.path.join(data_dir, pattern))

    if not bin_files:
        # If no bin files found in the specified directory, print an error message
        print(
            f"Error: No bin files found in {data_dir}. Make sure the data directory contains "
            f"'train' and 'val' folders with bin files."
        )

        # Fallback to old method if no matching files found
        fallback_data_dir = os.path.join("data", config['board_game'])
        data_filename = config['train_data_filename'] if split == 'train' else config['val_data_filename']

        print(f"Falling back to {os.path.join(fallback_data_dir, data_filename)}")

        with open(os.path.join(fallback_data_dir, data_filename), 'rb') as f:
            game_sequences: List[List[int]] = pickle.load(f)
        # end with
    else:
        # Load all bin files and combine their data
        print(f"Found {len(bin_files)} bin files for {split} split")
        game_sequences: List[List[int]] = []
        for bin_file in bin_files:
            print(f"Loading {bin_file}...")
            with open(bin_file, 'rb') as f:
                sequences = pickle.load(f)
                game_sequences.extend(sequences)
            # end with
        # end for
    # end if

    # Get the appropriate data based on the split
    game_sequences = _train_data if split == 'train' else _val_data

    return game_sequences
# end def load_game_sequences


def get_boards(
        game_sequences: List[List[int]],
) -> torch.LongTensor:
    """
    Get boards from a list of game sequences as a class labels.

    Args:
        game_sequences (List[List[int]]): List of game sequences with same length

    Returns:
        List[List[int]]: List of boards
    """
    # Mapping
    mapping = create_id_to_move_mapping()

    # Transform idx to moves (str)
    game_sequences = [
        [mapping[idx] for idx in game]
        for game in game_sequences
    ]

    # Transform each game in a board representation
    boards = [
        game_to_board(moves)
        for moves in game_sequences
    ]

    return torch.LongTensor(boards)
# end get_boards


def get_partial_games(
        game_sequences: List[List[int]],
        length: int,
) -> List[List[int]]:
    """
    Get partial game from a list of game sequences with length equal or above length.

    Args:
        game_sequences (List[List[int]]): List of game sequences with same length
        length (int): Length of the partial game

    Return:
        List[List[int]]: List of partial games with same length
    """
    return [
        sequence[:length] for sequence in game_sequences if len(sequence) >= length
    ]
# end def get_partial_games


def save_checkpoint(
        model,
        optimizer,
        iter_num,
        best_val_loss,
        config,
        model_args=None,
        linear_probe=None
):
    """
    Save a checkpoint of the model and optimizer state.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        iter_num (int): Current iteration number
        best_val_loss (float): Best validation loss so far
        config (dict): Configuration dictionary
        model_args (dict, optional): Model arguments
        linear_probe (torch.nn.Module, optional): Linear probe model if applicable
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    
    # Add model_args if provided
    if model_args is not None:
        checkpoint['model_args'] = model_args
    
    # Add linear probe if provided
    if linear_probe is not None:
        checkpoint['linear_probe'] = linear_probe.state_dict()
    
    print(f"Saving checkpoint to {config['out_dir']}")
    # Save with standard name for backward compatibility
    torch.save(checkpoint, os.path.join(config['out_dir'], 'linear_probe_ckpt.pt'))
    
    # Save with iteration number in filename
    iter_filename = f'linear_probe_ckpt_iter{iter_num}.pt'
    torch.save(checkpoint, os.path.join(config['out_dir'], iter_filename))
    print(f"Also saved checkpoint as {iter_filename}")
# end save_checkpoint


def load_checkpoint(ckpt_path, device, linear_probe=None):
    """
    Load a checkpoint for resuming training.
    
    Args:
        ckpt_path (str): Path to the checkpoint file
        device (torch.device): Device to load the checkpoint to
        linear_probe (torch.nn.Module, optional): Linear probe model to load weights into
        
    Returns:
        tuple: Contains checkpoint data including optimizer state, iteration number, etc.
    """
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Load linear probe if provided
    if linear_probe is not None and 'linear_probe' in checkpoint:
        linear_probe.load_state_dict(checkpoint['linear_probe'])
        print("Loaded linear probe weights from checkpoint")
    
    return checkpoint
# end load_checkpoint


MAX_GAME_LENGTH = 60


@torch.no_grad()
def evaluate_linear_probe(model, linear_probe, data, ctx, config, device):
    """
    Evaluate the linear probe model on validation data.
    
    Args:
        model: The base GPT model
        linear_probe: The linear probe model
        data: Dictionary containing data for different game lengths
        ctx: Context manager for mixed precision
        config: Configuration dictionary
        device: Device to use for tensors
        
    Returns:
        dict: Contains metrics like average loss and accuracy
    """
    print("Evaluating linear probe...")
    model.eval()
    linear_probe.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Evaluate on a subset of game lengths for efficiency
    eval_lengths = list(range(0, MAX_GAME_LENGTH, config.get('eval_stride', 5)))
    
    for length in eval_lengths:
        if length not in data:
            continue
            
        x, y = data[length]
        
        # Move data to device if not already there
        if x.device != device:
            x = x.to(device)
            y = y.to(device)
        
        with ctx:
            # Get model residuals from the last layer
            # Request residuals from the last layer (n_layer - 1)
            last_layer = model.config.n_layer - 1
            residuals_key = f"residuals{last_layer}"
            
            # Forward pass with to_return parameter to get residuals
            # Only request residuals, not logits or loss since we don't need them
            outputs = model(x, to_return=[residuals_key])
            
            # Get residuals from the outputs (first element since it's the first in to_return)
            residuals = outputs[0]  # Shape: [batch_size, seq_len, n_embd]
            
            # Reshape for linear probe
            batch_size, seq_len, n_embd = residuals.shape
            residuals = residuals.reshape(-1, n_embd)  # Shape: [batch_size*seq_len, n_embd]
            
            # Forward pass through linear probe
            logits = linear_probe(residuals)  # Shape: [batch_size*seq_len, 64*3]
            
            # Reshape logits and targets for loss calculation
            logits = logits.view(batch_size, seq_len, 64, 3)  # Shape: [batch_size, seq_len, 64, 3]
            
            # Calculate loss (cross entropy for each cell)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 3),  # Shape: [batch_size*seq_len*64, 3]
                y.reshape(-1)           # Shape: [batch_size*seq_len*64]
            )
            
            # Calculate accuracy
            pred = logits.argmax(dim=-1)  # Shape: [batch_size, seq_len, 64]
            correct = (pred == y).float().sum().item()
            
            # Update metrics
            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size * seq_len * 64
    
    # Calculate average metrics
    avg_loss = total_loss / len(eval_lengths)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # Print metrics
    print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    model.train()
    linear_probe.train()
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }
# end evaluate_linear_probe


def setup_data(
        split: str,
        config: Dict
) -> Dict[int, Tuple[torch.Tensor, torch.LongTensor]]:
    """
    Setup data for linear probe training.

    Args:
        split (str): Split from which to extract data.
        config (Dict): Configuration dictionary.
    """
    # Load game sequences
    game_sequences: List[List[int]] = load_game_sequences(split, config)

    # Final data
    data: Dict[int, Tuple[torch.Tensor, torch.LongTensor]] = {}

    # For each length
    for length in range(MAX_GAME_LENGTH):
        # Get partial game sequences by length
        partial_sequences: List[List[int]] = get_partial_games(
            game_sequences=game_sequences,
            length=length
        )

        # Get y by converting sequences to board representations
        y: torch.LongTensor = get_boards(partial_sequences)

        # Select subsequences
        x: torch.Tensor = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in partial_sequences])

        # Add to data
        data[length] = (x, y)
    # end for length

    return data
# end def setup_data


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    
    # Load configuration from YAML file if provided
    config = load_config(args.config) if args.config else {}
    
    # Create output directory if it doesn't exist
    os.makedirs(config['out_dir'], exist_ok=True)
    
    # Set up precision type for training
    dtype = config.get('dtype', 'float32')
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    
    # Create context manager for mixed precision training
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Load the model
    model = load_safetensors(args.model_file, args.model_config_file)
    model = model.to(device)
    
    # Initialize linear probe
    n_embd = model.config.n_embd
    linear_probe = torch.nn.Linear(n_embd, 64 * 3)  # 64 cells, 3 states per cell
    linear_probe = linear_probe.to(device)
    
    # Initialize optimizer
    optimizer, scaler = initialize_optimizer(
        model=model,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        beta1=config['beta1'],
        beta2=config['beta2'],
        scaler_enabled=(dtype == 'float16'),
        device_type=device_type,
    )
    
    # Initialize iteration counter and best validation metrics
    iter_num = 0
    best_val_loss = float('inf')
    
    # Load checkpoint if provided
    if args.ckpt is not None:
        checkpoint = load_checkpoint(args.ckpt, device, linear_probe)
        
        # Set iteration number - use command line value if provided, otherwise use checkpoint value
        if args.num_iter is not None:
            iter_num = args.num_iter
            print(f"Starting from iteration {iter_num} as specified by --num-iter")
        else:
            iter_num = checkpoint['iter_num']
            print(f"Resuming from iteration {iter_num} from checkpoint")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load best validation loss
        best_val_loss = checkpoint['best_val_loss']
        print(f"Best validation loss from checkpoint: {best_val_loss:.4f}")
    
    # Load training and validation data
    print("Loading training data...")
    train_data = setup_data(
        split='train',
        config=config
    )
    
    print("Loading validation data...")
    val_data = setup_data(
        split='val',
        config=config
    )
    
    # Display random samples to verify board representation matches move sequences
    if config.get('show_samples', True):
        print("Displaying random samples to verify board representation matches move sequences...")
        show_linear_probe_samples(train_data, num_samples=3)
    
    # Set models to training mode
    model.eval()  # Base model stays in eval mode
    linear_probe.train()
    
    # Training loop
    print(f"Beginning training from iteration {iter_num}")
    t0 = time.time()
    
    # Main training loop
    while iter_num < config.get('max_iters', 10000):
        # Sample a random game length for this iteration
        length = random.randint(0, MAX_GAME_LENGTH - 1)
        
        # Skip if no data for this length
        if length not in train_data:
            continue
        
        # Get batch for this length
        x, y = train_data[length]
        
        # Move data to device if not already there
        if x.device != device:
            x = x.to(device)
            y = y.to(device)
        
        # Forward pass with gradient computation
        with ctx:
            # Get model residuals from the last layer (no gradient computation for the base model)
            with torch.no_grad():
                # Request residuals from the last layer (n_layer - 1)
                last_layer = model.config.n_layer - 1
                residuals_key = f"residuals{last_layer}"
                
                # Forward pass with to_return parameter to get residuals
                # Only request residuals, not logits or loss since we don't need them
                outputs = model(x, to_return=[residuals_key])
                
                # Get residuals from the outputs (first element since it's the first in to_return)
                residuals = outputs[0]  # Shape: [batch_size, seq_len, n_embd]
            
            # Reshape for linear probe
            batch_size, seq_len, n_embd = residuals.shape
            residuals = residuals.reshape(-1, n_embd)  # Shape: [batch_size*seq_len, n_embd]
            
            # Forward pass through linear probe
            logits = linear_probe(residuals)  # Shape: [batch_size*seq_len, 64*3]
            
            # Reshape logits and targets for loss calculation
            logits = logits.view(batch_size, seq_len, 64, 3)  # Shape: [batch_size, seq_len, 64, 3]
            
            # Calculate loss (cross entropy for each cell)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 3),  # Shape: [batch_size*seq_len*64, 3]
                y.reshape(-1)           # Shape: [batch_size*seq_len*64]
            )
        
        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        
        if dtype == 'float16':
            # Use gradient scaling for float16
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular backward pass for float32
            loss.backward()
            optimizer.step()
        
        # Logging
        if iter_num % config.get('log_interval', 10) == 0:
            # Calculate accuracy
            with torch.no_grad():
                pred = logits.argmax(dim=-1)  # Shape: [batch_size, seq_len, 64]
                correct = (pred == y).float().sum().item()
                accuracy = correct / (batch_size * seq_len * 64)
            
            # Calculate time per iteration
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            
            # Print metrics
            print(f"iter {iter_num}: loss {loss.item():.4f}, accuracy {accuracy:.4f}, time {dt*1000:.2f}ms, length {length}")
        
        # Evaluation
        if iter_num % config.get('eval_interval', 500) == 0:
            metrics = evaluate_linear_probe(model, linear_probe, val_data, ctx, config, device)
            val_loss = metrics['loss']
            val_accuracy = metrics['accuracy']
            
            print(f"Evaluation at iter {iter_num}: val_loss {val_loss:.4f}, val_accuracy {val_accuracy:.4f}")
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss or config.get('always_save_checkpoint', False):
                best_val_loss = val_loss
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iter_num=iter_num,
                    best_val_loss=best_val_loss,
                    config=config,
                    linear_probe=linear_probe
                )
        
        # Increment iteration counter
        iter_num += 1
        
        # Exit if max iterations reached
        if iter_num >= config.get('max_iters', 10000):
            break
    
    # Final evaluation
    metrics = evaluate_linear_probe(model, linear_probe, val_data, ctx, config, device)
    val_loss = metrics['loss']
    val_accuracy = metrics['accuracy']
    
    print(f"Final evaluation: val_loss {val_loss:.4f}, val_accuracy {val_accuracy:.4f}")
    
    # Save final checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iter_num=iter_num,
        best_val_loss=best_val_loss,
        config=config,
        linear_probe=linear_probe
    )
    
    print("Training complete!")
# end def main


# Execute main function if script is run directly
if __name__ == "__main__":
    main()
# end if


