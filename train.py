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
from contextlib import nullcontext
from typing import List

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from boardGPT.models import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Default configuration values
# These values are designed to train a gpt2 (124M) model on OpenWebText
# They can be overridden via command line arguments or config file
# -----------------------------------------------------------------------------

# I/O settings
out_dir = 'out'                  # Output directory for checkpoints and logs
eval_interval = 2000             # How often to evaluate the model
log_interval = 1                 # How often to log training progress
eval_iters = 200                 # Number of batches to use for evaluation
eval_only = False                # If True, script exits right after the first eval
always_save_checkpoint = True    # If True, always save a checkpoint after each eval
init_from = 'scratch'            # 'scratch', 'resume', or 'gpt2*'

# wandb logging settings
wandb_log = False                # Whether to use wandb logging (disabled by default)
wandb_project = 'owt'            # Project name for wandb
wandb_run_name = 'gpt2'          # Run name for wandb

# Data settings
board_game = "othello"
data_filename = "synthetic.bin"
dataset = 'openwebtext'                 # Dataset name
gradient_accumulation_steps = 5 * 8     # Used to simulate larger batch sizes
batch_size = 12                         # If gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 61                       # Context size for the model

# Model architecture settings
n_layer = 8                      # Number of transformer layers
n_head = 8                       # Number of attention heads
n_embd = 512                     # Embedding dimension
dropout = 0.0                    # Dropout rate (0 for pretraining, try 0.1+ for finetuning)
bias = False                     # Whether to use bias in LayerNorm and Linear layers

# Optimizer settings (AdamW)
learning_rate = 6e-4             # Maximum learning rate
max_iters = 600000               # Total number of training iterations
weight_decay = 1e-1              # Weight decay coefficient
beta1 = 0.9                      # AdamW beta1 parameter
beta2 = 0.95                     # AdamW beta2 parameter
grad_clip = 1.0                  # Clip gradients at this value (disable if == 0.0)

# Learning rate decay settings
decay_lr = True                  # Whether to decay the learning rate
warmup_iters = 2000              # Number of warmup steps
lr_decay_iters = 600000          # Should be ~= max_iters per Chinchilla
min_lr = 6e-5                    # Minimum learning rate (~= learning_rate/10 per Chinchilla)

# DDP settings
backend = 'nccl'                 # Backend for distributed training ('nccl', 'gloo', etc.)

# System settings
device = 'cuda'                  # Device to use ('cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps' on macbooks)

# Choose appropriate dtype based on hardware capabilities
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True                   # Whether to use PyTorch 2.0 compilation for speed


# -----------------------------------------------------------------------------
# Configuration handling
# -----------------------------------------------------------------------------
# Collect all config keys for logging
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]

# Load overrides from command line or config file
exec(open('configurator.py').read())

# Create config dictionary for logging
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------


def setup_training_environment():
    """
    Set up the training environment including distributed training, 
    random seeds, and device configuration.
    
    Returns:
        tuple: Contains various setup parameters including master_process flag,
               device_type, context manager for mixed precision, etc.
    """
    global gradient_accumulation_steps
    
    # Check if this is a distributed data parallel (DDP) run
    ddp = int(os.environ.get('RANK', -1)) != -1
    
    # Initialize variables
    ddp_rank = None
    ddp_local_rank = None
    ddp_world_size = None
    master_process = True
    seed_offset = 0
    
    if ddp:
        # Initialize the distributed process group
        init_process_group(backend=backend)
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
    # end if
    
    # Calculate tokens per iteration for logging
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    
    # Create output directory if needed (only on master process)
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    # end if
    
    # Set random seed for reproducibility
    torch.manual_seed(1337 + seed_offset)
    
    # Enable TF32 precision on CUDA devices (faster and usually sufficient precision)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Determine device type for later use
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    # Set up precision type for training
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    
    # Create context manager for mixed precision training
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    return ddp, master_process, seed_offset, ddp_world_size, ddp_local_rank, device, device_type, ctx
# end setup_training_environment


def get_board_batch(split):
    """
    ...
    """
    # Data dir
    data_dir = os.path.join("data", board_game)

    # Load data
    # game_sequences is a list of list of ints (List[List[int])
    with open(os.path.join(data_dir, data_filename), 'rb') as f:
        game_sequences: List[List[int]] = pickle.load(f)
    # end with

    # Concatenate games sequences
    game_sequences: List[int] = [x for sublist in game_sequences for x in sublist]

    # Get the selection
    game_sequences = random.sample(game_sequences, batch_size)

    # Sample random sequence length
    ix = [random.randint(0, len(game_sequences)-block_size) for _ in range(batch_size)]

    # Select subsequences
    x = torch.stack([torch.tensor(game_sequences[i:i+block_size], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(game_sequences[i+1:i+1+block_size], dtype=torch.long) for i in ix])

    # Device type
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    # end if

    return x, y
# end def get_board_batch



def get_batch(split):
    """
    Get a random batch of data from the specified split.
    
    Args:
        split (str): 'train' or 'val' to specify which data split to use
        
    Returns:
        tuple: (x, y) where x is input tokens and y is target tokens
    """
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    data_dir = os.path.join('data', dataset)
    
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    # end if
    
    # Sample random indices for batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Create input and target sequences
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        # Pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    # end if
    
    return x, y

def initialize_model():
    """
    Initialize the model based on the init_from parameter.
    
    Returns:
        tuple: (model, iter_num, best_val_loss) where model is the initialized GPT model,
               iter_num is the starting iteration number, and best_val_loss is the best
               validation loss (used for checkpointing).
    """
    global model_args
    
    # Initialize iteration counter and best validation loss
    iter_num = 0
    best_val_loss = 1e9
    
    # Attempt to derive vocab_size from the dataset
    data_dir = os.path.join('data', dataset)
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # end with
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    # end if
    
    # Set up model arguments from configuration
    model_args = dict(
        n_layer=n_layer, 
        n_head=n_head, 
        n_embd=n_embd, 
        block_size=block_size,
        bias=bias, 
        vocab_size=None, 
        dropout=dropout
    )
    
    # Initialize model based on init_from parameter
    if init_from == 'scratch':
        # Initialize a new model from scratch
        print("Initializing a new model from scratch")
        
        # Determine the vocab size for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        # end if

        # Create the model
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        
        # Resume training from a checkpoint
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
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
        # end for

        # Load weights
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        
        # Initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        
        # Read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
        # end for
    # end if
    
    # Crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size  # So that the checkpoint will have the right value
    # end if
    
    # Move model to the specified device
    model.to(device)
    
    return model, iter_num, best_val_loss
# end def initialize_model

@torch.no_grad()
def estimate_loss():
    """
    Estimate loss over train and validation splits using multiple batches.
    
    Returns:
        dict: Contains average loss for 'train' and 'val' splits
    """
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            # end with
            losses[k] = loss.item()
        # end for
        
        out[split] = losses.mean()
    # end for
    
    model.train()
    return out

def get_lr(it):
    """
    Get learning rate for the current iteration according to the schedule.
    
    Args:
        it (int): Current iteration number
        
    Returns:
        float: Learning rate for the current iteration
    """
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # end if
    
    # 2) If it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # end if
    
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
# end get_lr

def setup_optimizer(model, checkpoint=None):
    """
    Set up the optimizer for training.
    
    Args:
        model: The model to optimize
        checkpoint: Optional checkpoint dictionary for resuming training
        
    Returns:
        optimizer: Configured optimizer
        scaler: GradScaler for mixed precision training
    """
    # Initialize a GradScaler for mixed precision training
    # If enabled=False (not using float16), scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # Set up the optimizer
    optimizer = model.configure_optimizers(
        weight_decay, 
        learning_rate, 
        (beta1, beta2), 
        device_type
    )
    
    # Load optimizer state if resuming training
    if init_from == 'resume' and checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    # end if
    
    return optimizer, scaler
# end def setup_optimizer

def save_checkpoint(
        model,
        optimizer,
        iter_num,
        best_val_loss
):
    """
    Save a checkpoint of the model and optimizer state.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        iter_num (int): Current iteration number
        best_val_loss (float): Best validation loss so far
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
# end def save_checkpoint

def main():
    """
    Main training function that orchestrates the entire training process.
    """
    global model, ctx, device, device_type
    
    # Set up the training environment main
    ddp, master_process, seed_offset, ddp_world_size, ddp_local_rank, device, device_type, ctx = setup_training_environment()
    
    # Initialize the model
    model, iter_num, best_val_loss = initialize_model()
    
    # Set up the optimizer and gradient scaler
    optimizer, scaler = setup_optimizer(model)
    
    # Compile the model if requested (requires PyTorch 2.0+)
    if compile:
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
    if wandb_log and master_process:
        import wandb
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=config
        )
    # end if
    
    # Training loop initialization
    X, Y = get_batch('train')  # Fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # Number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # Unwrap DDP container if needed
    running_mfu = -1.0
    
    # Main training loop
    while True:
        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # end for
        
        # Evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Log to wandb if enabled
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100,  # Convert to percentage
                })
            # end if
            
            # Save checkpoint if validation loss improved or if always_save_checkpoint is True
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    save_checkpoint(raw_model, optimizer, iter_num, best_val_loss)
            # end if
        # end if
        
        # Exit if eval_only flag is set
        if iter_num == 0 and eval_only:
            break
        
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
            X, Y = get_batch('train')
            
            # Backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # end for
        
        # Clip the gradient if grad_clip is set
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
        
        if iter_num % log_interval == 0 and master_process:
            # Get loss as float. Note: this is a CPU-GPU sync point
            # Scale up to undo the division above, approximating the true total loss
            lossf = loss.item() * gradient_accumulation_steps
            
            # Calculate model flops utilization (MFU)
            if local_iter_num >= 5:  # Let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            # end if
            
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        # end if
        
        # Increment iteration counters
        iter_num += 1
        local_iter_num += 1
        
        # Check termination condition
        if iter_num > max_iters:
            break
    # end while
    
    # Clean up distributed process group if using DDP
    if ddp:
        destroy_process_group()
    # end if

# Execute main function if script is run directly
if __name__ == "__main__":
    main()
# end if