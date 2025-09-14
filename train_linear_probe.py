
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
import random
# Imports
from typing import Dict, List, Optional, Tuple
import os
import pickle
import glob

import numpy as np
import torch
import yaml
import time
import argparse
from contextlib import nullcontext
from rich.console import Console
from rich import traceback

# Initialize Rich console and traceback
console = Console()
traceback.install()

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
    parser.add_argument('--model-file', type=str, required=True, help='Path to the model safetensors file')

    parser.add_argument(
        '--model-config-file', type=str, required=True, help='Path to the model configuration JSON file'
    )

    parser.add_argument(
        '--config', type=str, required=True, help='Path to the YAML configuration file'
    )

    parser.add_argument(
        '--data-dir', type=str, required=True,
        help='Path to the directory containing training and validation data'
    )
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
        console.print(f"Loaded configuration from [bold]{config_path}[/bold]")
        return config
    except FileNotFoundError:
        console.print(f"[yellow]Configuration file {config_path} not found. Using default configuration.[/yellow]")
        return {}
    except yaml.YAMLError as e:
        console.print(f"[bold red]Error parsing YAML configuration file:[/bold red] {e}")
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
        data_dir: str,
        split: str,
        config: Dict
) -> List[List[int]]:
    """
    Load data from the specified split.

    Args:
        data_dir (str): Path to the data directory
        split (str): Split to load
        config (Dict): Configuration dictionary
    """
    global _train_data, _val_data

    # Load all data into memory if not already loaded
    if (split == 'train' and _train_data is None) or (split == 'val' and _val_data is None):
        console.print(f"Loading [bold cyan]{split}[/bold cyan] data into memory...")

        # Data dir for the specified split (train or val)
        data_dir = os.path.join(data_dir, split)

        # Pattern for bin files
        pattern = "*.bin"

        # Find all matching bin files
        bin_files = glob.glob(os.path.join(data_dir, pattern))

        if not bin_files:
            # If no bin files found in the specified directory, print an error message
            console.print(
                f"[bold red]Error:[/bold red] No bin files found in {data_dir}. Make sure the data directory contains "
                f"'train' and 'val' folders with bin files."
            )

            # Fallback to old method if no matching files found
            fallback_data_dir = os.path.join("data", config['board_game'])
            data_filename = config['train_data_filename'] if split == 'train' else config['val_data_filename']

            console.print(f"Falling back to [yellow]{os.path.join(fallback_data_dir, data_filename)}[/yellow]")

            with open(os.path.join(fallback_data_dir, data_filename), 'rb') as f:
                game_sequences: List[List[int]] = pickle.load(f)
            # end with
        else:
            # Load all bin files and combine their data
            console.print(f"Found [bold green]{len(bin_files)}[/bold green] bin files for [bold cyan]{split}[/bold cyan] split")
            game_sequences: List[List[int]] = []
            for bin_file in bin_files:
                console.print(f"Loading [blue]{bin_file}[/blue]...")
                with open(bin_file, 'rb') as f:
                    sequences = pickle.load(f)
                    game_sequences.extend(sequences)
                # end with
            # end for
        # end if

        # Concatenate game sequences
        print(f"Total sequences for {split}: {len(game_sequences)}")

        # Store in the appropriate global variable
        if split == 'train':
            _train_data = game_sequences
            print(f"Train data loaded into memory")
        else:
            _val_data = game_sequences
            print(f"Validation data loaded into memory")
        # end if
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
        min_seq_number: int,
        sequence_lengths: Optional[List[int]] = None
) -> Optional[List[np.array]]:
    """
    Get partial game from a list of game sequences with length equal or above length.

    Args:
        game_sequences (List[List[int]]): List of game sequences with same length
        length (int): Length of the partial game
        min_seq_number (int): Batch size
        sequence_lengths (Optional[List[int]]): Precomputed lengths of all sequences in game_sequences

    Return:
        List[List[int]]: List of partial games with same length
    """
    if sequence_lengths is not None:
        # Use precomputed lengths to directly get sequences of required length
        indices = [i for i, seq_len in enumerate(sequence_lengths) if seq_len >= length]
        len_seq = [np.array(game_sequences[i]) for i in indices]
    else:
        # Fallback to original implementation
        len_seq = [np.array(seq) for seq in game_sequences if len(seq) >= length]
    # end if

    if len(len_seq) >= min_seq_number:
        return [
            sequence[:length] for sequence in len_seq
        ]
    else:
        return None
    # end if
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
    
    console.print(f"Saving checkpoint to [bold blue]{config['out_dir']}[/bold blue]")
    # Save with standard name for backward compatibility
    torch.save(checkpoint, os.path.join(config['out_dir'], 'linear_probe_ckpt.pt'))
    
    # Save with iteration number in filename
    iter_filename = f'linear_probe_ckpt_iter{iter_num}.pt'
    torch.save(checkpoint, os.path.join(config['out_dir'], iter_filename))
    console.print(f"Also saved checkpoint as [bold green]{iter_filename}[/bold green]")
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
    console.print(f"Loading checkpoint from [bold blue]{ckpt_path}[/bold blue]")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Load linear probe if provided
    if linear_probe is not None and 'linear_probe' in checkpoint:
        linear_probe.load_state_dict(checkpoint['linear_probe'])
        console.print("[bold green]Loaded linear probe weights from checkpoint[/bold green]")
    
    return checkpoint
# end load_checkpoint


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
    console.print("[bold]Evaluating linear probe...[/bold]")
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
    console.print(f"Evaluation - [bold]Loss:[/bold] [red]{avg_loss:.4f}[/red], [bold]Accuracy:[/bold] [green]{accuracy:.4f}[/green]")
    
    model.train()
    linear_probe.train()
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }
# end evaluate_linear_probe


MAX_GAME_LENGTH = 60


def setup_data(
        data_dir: str,
        split: str,
        config: Dict,
        max_game_sequences: int
) -> Dict[int, Tuple[torch.Tensor, torch.LongTensor]]:
    """
    Setup data for linear probe training.

    Args:
        data_dir: Path to the data directory
        split (str): Split from which to extract data.
        config (Dict): Configuration dictionary.
        max_game_sequences (int): Maximum number of game sequences to load per length
    """
    # Check if pickle file exists
    pickle_filename = f"{split}_linear_probe_data.pkl"
    pickle_path = os.path.join(data_dir, pickle_filename)
    
    # If pickle file exists, load it and return
    if os.path.exists(pickle_path):
        console.print(f"Loading {split} data from pickle file: {pickle_path}")
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            # end with
            return data
        except Exception as e:
            console.print(f"[bold red]Error loading pickle file:[/bold red] {e}")
            console.print("Falling back to processing data from scratch...")
        # end try
    # end if
    
    # If no pickle file or error loading, process data from scratch
    # Load game sequences
    game_sequences: List[List[int]] = load_game_sequences(
        data_dir=data_dir,
        split=split,
        config=config
    )

    # Precompute the length of all sequences in game_sequences
    console.print("Precomputing sequence lengths...")
    sequence_lengths: List[int] = [len(seq) for seq in game_sequences]
    console.print(f"Precomputed lengths for {len(sequence_lengths)} sequences")

    # Final data
    data: Dict[int, Tuple[torch.Tensor, torch.LongTensor]] = {}

    # For each length
    for length in range(2, MAX_GAME_LENGTH):
        console.print(f"Looking at {length} game sequences...")

        # Get partial game sequences by length
        partial_sequences: Optional[List[np.array]] = get_partial_games(
            game_sequences=game_sequences,
            length=length,
            min_seq_number=config['batch_size'],
            sequence_lengths=sequence_lengths
        )

        # If longer, then select randomly
        if len(partial_sequences) > max_game_sequences:
            # Random ix
            ix = random.sample(range(0, len(partial_sequences) + 1), max_game_sequences)
            partial_sequences = [partial_sequences[i] for i in ix]
        # end if

        # Get y by converting sequences to board representations
        y: torch.LongTensor = get_boards(partial_sequences)

        # Select subsequences
        x: torch.Tensor = torch.stack([torch.tensor(seq, dtype=torch.long) for seq in partial_sequences])

        # Load
        console.print(f"Found {len(partial_sequences)} sequences in {length} games!")

        # Add to data
        data[length] = (x, y)
    # end for length

    return data
# end def setup_data


def setup_model(
        model_file: str,
        model_config_file: str,
        device: torch.device,
):
    # Load the model
    model, _ = load_safetensors(
        model_file,
        model_config_file
    )
    model = model.to(device)

    # Initialize linear probe
    n_embd = model.config.n_embd
    linear_probe = torch.nn.Linear(n_embd, 64 * 3)  # 64 cells, 3 states per cell
    linear_probe = linear_probe.to(device)

    return model, linear_probe
# end def setup_model


def save_probe_data(
        data_dir: str,
        split: str,
        data
):
    # Save training data to pickle file
    pickle_path = os.path.join(data_dir, f"{split}_linear_probe_data.pkl")
    console.print(f"Saving training data to pickle file: {pickle_path}")
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        # end with
        console.print("[bold green]Training data saved successfully![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error saving training data:[/bold red] {e}")
    # end try
# end def save_probe_data


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

    # Setup models
    model, linear_probe = setup_model(
        model_file=args.model_file,
        model_config_file=args.model_config_file,
        device=device,
    )
    
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
    
    # Load training and validation data
    console.print("Loading training data...")
    train_data = setup_data(
        data_dir=args.data_dir,
        split='train',
        config=config,
        max_game_sequences=config['max_game_sequences'],
    )
    
    # Save training data to pickle file
    save_probe_data(args.data_dir, split='train', data=train_data)

    console.print("Loading validation data...")
    val_data = setup_data(
        data_dir=args.data_dir,
        split='val',
        config=config,
        max_game_sequences=config['max_game_sequences'],
    )

    # Save training data to pickle file
    save_probe_data(args.data_dir, split='val', data=val_data)

    # Display random samples to verify board representation matches move sequences
    if config.get('show_samples', True):
        console.print("Displaying random samples to verify board representation matches move sequences...")
        show_linear_probe_samples(train_data, num_samples=3)
    # end if

    # Set models to training mode
    model.eval()  # Base model stays in eval mode
    linear_probe.train()

    # Training loop
    console.print(f"Beginning training from iteration {iter_num}")
    t0 = time.time()

    # Main training loop
    while iter_num < config.get('max_iters', 10000):
        # Sample a random game length for this iteration
        length = random.randint(0, MAX_GAME_LENGTH - 1)

        # Skip if no data for this length
        if length not in train_data:
            continue
        # end if length

        # Get batch for this length
        x, y = train_data[length]

        # Move data to device if not already there
        if x.device != device:
            x = x.to(device)
            y = y.to(device)
        # end if

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
            # end with

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
        # end with ctx

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
        # end if

        # Logging
        if iter_num % config.get('log_interval', 10) == 0:
            # Calculate accuracy
            with torch.no_grad():
                pred = logits.argmax(dim=-1)  # Shape: [batch_size, seq_len, 64]
                correct = (pred == y).float().sum().item()
                accuracy = correct / (batch_size * seq_len * 64)
            # end with no_grad

            # Calculate time per iteration
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            # Print metrics
            print(f"iter {iter_num}: loss {loss.item():.4f}, accuracy {accuracy:.4f}, time {dt*1000:.2f}ms, length {length}")
        # end if

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
            # end if
        # end if

        # Increment iteration counter
        iter_num += 1

        # Exit if max iterations reached
        if iter_num >= config.get('max_iters', 10000):
            break
        # end if
    # end while iter

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


