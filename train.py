"""
BoardGPT Training Script
========================

This script trains a GPT model for board games. It supports both single-GPU and
distributed data parallel (DDP) training configurations.

The script handles:
- Model initialization (from scratch, resume from checkpoint, or pretrained GPT-2)
- Data loading and batching
- Training loop with gradient accumulation
- Learning rate scheduling
- Evaluation and checkpointing
- Logging (console and optional wandb integration)

Examples:
---------
To run on a single GPU:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

# Imports
import os
import time
import random
import math
import pickle
import argparse
import yaml
from contextlib import nullcontext
from typing import List, Dict, Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from boardGPT.models import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration handling
# -----------------------------------------------------------------------------

def parse_args():
    """
    Parse command line arguments for the training script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Train a GPT model for board games')
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to the YAML configuration file')
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

def setup_training_environment(config):
    """
    Set up the training environment including distributed training, 
    random seeds, and device configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: Contains various setup parameters including master_process flag,
               device_type, context manager for mixed precision, etc.
    """
    # Check if this is a distributed data parallel (DDP) run
    ddp = int(os.environ.get('RANK', -1)) != -1
    
    # Initialize variables
    ddp_rank = None
    ddp_local_rank = None
    ddp_world_size = None
    master_process = True
    seed_offset = 0
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    device = config['device']
    
    if ddp:
        # Initialize the distributed process group
        init_process_group(backend=config['backend'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # This process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # Each process gets a different seed
        
        # Scale down gradient accumulation steps proportionally to world size
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # If not DDP, we are running on a single GPU with one process
        ddp_world_size = 1
        # Use the device specified in the configuration
    # end if
    
    # Calculate tokens per iteration for logging
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * config['batch_size'] * config['block_size']
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    
    # Create output directory if needed (only on master process)
    if master_process:
        os.makedirs(config['out_dir'], exist_ok=True)
    # end if
    
    # Set random seed for reproducibility
    torch.manual_seed(1337 + seed_offset)
    
    # Enable TF32 precision on CUDA devices (faster and usually sufficient precision)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Determine device type for later use
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    # Set up precision type for training
    dtype = config['dtype']
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    
    # Create context manager for mixed precision training
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    return ddp, master_process, seed_offset, ddp_world_size, ddp_local_rank, device, device_type, ctx, gradient_accumulation_steps
# end setup_training_environment

def get_board_batch(split, config, device, device_type):
    """
    Get a random batch of board game data from the specified split.
    
    Args:
        split (str): 'train' or 'val' to specify which data split to use
        config (dict): Configuration dictionary
        device (str): Device to use for tensors
        device_type (str): Type of device ('cuda' or 'cpu')
        
    Returns:
        tuple: (x, y) where x is input tokens and y is target tokens
    """
    # Data dir
    data_dir = os.path.join("data", config['board_game'])

    # Data filename
    data_filename = config['train_data_filename'] if split == 'train' else config['val_data_filename']

    # Load data
    # game_sequences is a list of list of ints (List[List[int])
    with open(os.path.join(data_dir, data_filename), 'rb') as f:
        game_sequences: List[np.array] = pickle.load(f)
    # end with

    # Concatenate games sequences
    game_sequences: List[int] = [x for sublist in game_sequences for x in sublist.tolist()]

    # Sample random sequence start position
    ix = [random.randint(0, len(game_sequences)-config['block_size']-1) for _ in range(config['batch_size'])]

    # Select subsequences
    x = torch.stack([torch.tensor(game_sequences[i:i+config['block_size']], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(game_sequences[i+1:i+1+config['block_size']], dtype=torch.long) for i in ix])

    # Device type
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    # end if

    return x, y
# end get_board_batch

def get_batch(split, config, device, device_type):
    """
    Get a random batch of data from the specified split.
    
    Args:
        split (str): 'train' or 'val' to specify which data split to use
        config (dict): Configuration dictionary
        device (str): Device to use for tensors
        device_type (str): Type of device ('cuda' or 'cpu')
        
    Returns:
        tuple: (x, y) where x is input tokens and y is target tokens
    """
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    data_dir = os.path.join('data', config['dataset'])
    
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    # end if
    
    # Sample random indices for batch
    ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
    
    # Create input and target sequences
    x = torch.stack([torch.from_numpy((data[i:i+config['block_size']]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config['block_size']]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        # Pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    # end if
    
    return x, y
# end get_batch

def initialize_model(config, device):
    """
    Initialize the model based on the init_from parameter.
    
    Args:
        config (dict): Configuration dictionary
        device (str): Device to use for the model
        
    Returns:
        tuple: (model, iter_num, best_val_loss, model_args) where model is the initialized GPT model,
               iter_num is the starting iteration number, best_val_loss is the best
               validation loss (used for checkpointing), and model_args are the model arguments.
    """
    # Initialize iteration counter and best validation loss
    iter_num = 0
    best_val_loss = 1e9
    
    # Set up model arguments from configuration
    model_args = dict(
        n_layer=config['n_layer'], 
        n_head=config['n_head'], 
        n_embd=config['n_embd'], 
        block_size=config['block_size'],
        bias=config['bias'], 
        vocab_size=None, 
        dropout=config['dropout']
    )

    # Vocab size
    vocab_size = config['vocab_size']
    
    # Initialize model based on init_from parameter
    if config['init_from'] == 'scratch':
        # Initialize a new model from scratch
        print("Initializing a new model from scratch")

        # Create the model
        model_args['vocab_size'] = vocab_size if vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif config['init_from'] == 'resume':
        print(f"Resuming training from {config['out_dir']}")
        
        # Resume training from a checkpoint
        ckpt_path = os.path.join(config['out_dir'], 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        
        # Force these config attributes to be equal otherwise we can't resume training
        # The rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # end for
        
        # Create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
        # Load the state dict
        state_dict = checkpoint['model']
        
        # Fix the keys of the state dictionary if needed
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            # end if
        # end for

        # Load weights
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif config['init_from'].startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {config['init_from']}")
        
        # Initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=config['dropout'])
        model = GPT.from_pretrained(config['init_from'], override_args)
        
        # Read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
        # end for
    # end if
    
    # Crop down the model block size if desired, using model surgery
    if config['block_size'] < model.config.block_size:
        model.crop_block_size(config['block_size'])
        model_args['block_size'] = config['block_size']  # So that the checkpoint will have the right value
    # end if
    
    # Move model to the specified device
    model.to(device)
    
    return model, iter_num, best_val_loss, model_args
# end initialize_model

@torch.no_grad()
def estimate_loss(model, ctx, config, device, device_type):
    """
    Estimate loss over train and validation splits using multiple batches.
    
    Args:
        model: The model to evaluate
        ctx: Context manager for mixed precision
        config (dict): Configuration dictionary
        device (str): Device to use for tensors
        device_type (str): Type of device ('cuda' or 'cpu')
        
    Returns:
        dict: Contains average loss for 'train' and 'val' splits
    """
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(config['eval_iters'])
        
        for k in range(config['eval_iters']):
            # Use the appropriate data loading function based on the configuration
            if config.get('board_game'):
                X, Y = get_board_batch(split, config, device, device_type)
            else:
                X, Y = get_batch(split, config, device, device_type)
            # end if

            with ctx:
                logits, loss = model(X, Y)
            # end with
            losses[k] = loss.item()
        # end for
        
        out[split] = losses.mean()
    # end for
    
    model.train()
    return out

def get_lr(it, config):
    """
    Get learning rate for the current iteration according to the schedule.
    
    Args:
        it (int): Current iteration number
        config (dict): Configuration dictionary
        
    Returns:
        float: Learning rate for the current iteration
    """
    # 1) Linear warmup for warmup_iters steps
    if it < config['warmup_iters']:
        return config['learning_rate'] * (it + 1) / (config['warmup_iters'] + 1)
    # end if
    
    # 2) If it > lr_decay_iters, return min learning rate
    if it > config['lr_decay_iters']:
        return config['min_lr']
    # end if
    
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Coeff ranges 0..1
    return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])

def setup_optimizer(model, config, device_type, checkpoint=None):
    """
    Set up the optimizer for training.
    
    Args:
        model: The model to optimize
        config (dict): Configuration dictionary
        device_type (str): Type of device ('cuda' or 'cpu')
        checkpoint: Optional checkpoint dictionary for resuming training
        
    Returns:
        tuple: (optimizer, scaler) where optimizer is the configured optimizer and
               scaler is the GradScaler for mixed precision training
    """
    # Initialize a GradScaler for mixed precision training
    # If enabled=False (not using float16), scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(config['dtype'] == 'float16'))
    
    # Set up the optimizer
    optimizer = model.configure_optimizers(
        config['weight_decay'], 
        config['learning_rate'], 
        (config['beta1'], config['beta2']), 
        device_type
    )
    
    # Load optimizer state if resuming training
    if config['init_from'] == 'resume' and checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    # end if
    
    return optimizer, scaler

def save_checkpoint(
        model,
        optimizer,
        iter_num,
        best_val_loss,
        config,
        model_args
):
    """
    Save a checkpoint of the model and optimizer state.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        iter_num (int): Current iteration number
        best_val_loss (float): Best validation loss so far
        config (dict): Configuration dictionary
        model_args (dict): Model arguments
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    
    print(f"saving checkpoint to {config['out_dir']}")
    torch.save(checkpoint, os.path.join(config['out_dir'], 'ckpt.pt'))
# end save_checkpoint

def main():
    """
    Main training function that orchestrates the entire training process.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Set up the training environment
    ddp, master_process, seed_offset, ddp_world_size, ddp_local_rank, device, device_type, ctx, gradient_accumulation_steps = setup_training_environment(config)
    
    # Initialize the model
    model, iter_num, best_val_loss, model_args = initialize_model(config, device)
    
    # Set up the optimizer and gradient scaler
    optimizer, scaler = setup_optimizer(model, config, device_type)
    
    # Compile the model if requested (requires PyTorch 2.0+)
    if config['compile']:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)
    # end if
    
    # Wrap model into DDP container for distributed training
    if ddp:
        model = DDP(
            model,
            device_ids=[ddp_local_rank] if ddp_local_rank is not None else None
        )
    # end if
    
    # Set up wandb logging if enabled
    if config['wandb_log'] and master_process:
        import wandb
        wandb.init(
            project=config['wandb_project'],
            name=config['wandb_run_name'],
            config=config
        )
    # end if
    
    # Training loop initialization
    # Use the appropriate data loading function based on the configuration
    if config.get('board_game'):
        X, Y = get_board_batch('train', config, device, device_type)  # Fetch the very first batch for board game
    else:
        X, Y = get_batch('train', config, device, device_type)  # Fetch the very first batch for text
    # end if

    t0 = time.time()
    local_iter_num = 0  # Number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # Unwrap DDP container if needed
    running_mfu = -1.0
    
    # Main training loop
    while True:
        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num, config) if config['decay_lr'] else config['learning_rate']
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # end for
        
        # Evaluate the loss on train/val sets and write checkpoints
        if iter_num % config['eval_interval'] == 0 and master_process:
            losses = estimate_loss(model, ctx, config, device, device_type)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Log to wandb if enabled
            if config['wandb_log']:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100,  # Convert to percentage
                })
            # end if
            
            # Save checkpoint if validation loss improved or if always_save_checkpoint is True
            if losses['val'] < best_val_loss or config['always_save_checkpoint']:
                best_val_loss = losses['val']
                if iter_num > 0:
                    save_checkpoint(raw_model, optimizer, iter_num, best_val_loss, config, model_args)
            # end if
        # end if
        
        # Exit if eval_only flag is set
        if iter_num == 0 and config['eval_only']:
            break
        # end if
        
        # Forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # In DDP training we only need to sync gradients at the last micro step.
                # The official way to do this is with model.no_sync() context manager, but
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            # end if
            
            # Forward pass
            with ctx:
                logits, loss = model(X, Y)
                # Scale the loss to account for gradient accumulation
                loss = loss / gradient_accumulation_steps
            # end with
            
            # Immediately async prefetch next batch while model is doing the forward pass on the GPU
            # Use the appropriate data loading function based on the configuration
            if config.get('board_game'):
                X, Y = get_board_batch('train', config, device, device_type)
            else:
                X, Y = get_batch('train', config, device, device_type)
            # end if
            
            # Backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # end for
        
        # Clip the gradient if grad_clip is set
        if config['grad_clip'] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        # end if
        
        # Step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        
        # Flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % config['log_interval'] == 0 and master_process:
            # Get loss as float. Note: this is a CPU-GPU sync point
            # Scale up to undo the division above, approximating the true total loss
            lossf = loss.item() * gradient_accumulation_steps
            
            # Calculate model flops utilization (MFU)
            if local_iter_num >= 5:  # Let the training loop settle a bit
                mfu = raw_model.estimate_mfu(config['batch_size'] * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            # end if
            
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        # end if
        
        # Increment iteration counters
        iter_num += 1
        local_iter_num += 1
        
        # Check termination condition
        if iter_num > config['max_iters']:
            break
        # end if
    # end while
    
    # Clean up distributed process group if using DDP
    if ddp:
        destroy_process_group()
    # end if
# end def main

# Execute main function if script is run directly
if __name__ == "__main__":
    main()
# end if