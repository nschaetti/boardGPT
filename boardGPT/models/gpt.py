"""
Full definition of a GPT Language Model, all of it in this single file.

This module implements a complete GPT (Generative Pre-trained Transformer) model
architecture. It includes all necessary components such as layer normalization,
self-attention mechanisms, feed-forward networks, and the full transformer model.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from distutils.command.config import config
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F

from .hooks import HookPoint
from .layer_norm import LayerNorm
from .block import Block, MLP, CausalSelfAttention

@dataclass
class GPTConfig:
    """
    Configuration class for GPT model parameters.

    This dataclass stores all hyperparameters needed to initialize a GPT model.
    Default values are set to match the original GPT-2 model.

    Attributes:
        block_size (int): Maximum sequence length the model can handle
        vocab_size (int): Size of the token vocabulary
        n_layer (int): Number of transformer blocks
        n_head (int): Number of attention heads in each block
        n_embd (int): Embedding dimension
        dropout (float): Dropout probability
        bias (bool): Whether to use bias in Linear and LayerNorm layers
    """
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


# end class GPTConfig

class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) language model.

    This class implements the full GPT architecture, including token and position
    embeddings, a stack of transformer blocks, and a language modeling head.
    It supports various operations like forward passes, model initialization,
    loading pre-trained weights (from Hugging Face or safetensors files), and text generation.

    Args:
        config (GPTConfig): Configuration object containing model parameters
    """

    def __init__(
            self,
            config: GPTConfig,
            use_flash: bool = True
    ):
        """
        Initialize the GPT model.
        """
        super().__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config

        # Hooks
        self.token_emb_hook = HookPoint()
        self.pos_emb_hook = HookPoint()
        self.residual_hook = [HookPoint() for _ in range(config.n_layer)]
        self.pre_logits_hook = HookPoint()

        # Create transformer components
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),  # position embeddings
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config, use_flash) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),  # final layer norm
            )
        )

        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: token embedding and output projection share weights
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # Initialize all weights
        self.apply(self._init_weights)

        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
            # end if
        # end for
    # end __init__

    def set_flash(self, use_flash: bool):
        """
        Activate/deactivate flash layers.

        Args:
            use_flash (bool): Whether to activate flash layers.
        """
        for block_i in range(self.config.n_layer):
            self.transformer['h'][block_i].attn.flash = use_flash
        # end for
    # end set_flash

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.

        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.

        Args:
            non_embedding (bool): Whether to exclude embedding parameters from the count

        Returns:
            int: Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        # end if
        return n_params
    # end def get_num_params

    def _init_weights(self, module):
        """
        Initialize the weights of the model.

        Linear layers are initialized from N(0, 0.02) and biases are set to zero.
        Embedding layers are initialized from N(0, 0.02).

        Args:
            module (nn.Module): Module to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            # end if
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # end if
    # end def _init_weights

    def forward(self, idx, targets=None):
        """
        Forward pass through the model.

        Args:
            idx (torch.Tensor): Input token indices of shape (batch_size, seq_len)
            targets (torch.Tensor, optional): Target token indices of shape (batch_size, seq_len)

        Returns:
            tuple: (logits, loss) where logits is the output predictions and loss is the
                  cross-entropy loss if targets are provided, otherwise None
        """
        device = idx.device

        # Batch size, sequence length
        b, t = idx.size()

        # Check max sequence length
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Pos is (t)
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # Forward the GPT model itself
        # tok_emb is (b, t, 512)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        tok_emb = self.token_emb_hook(tok_emb)

        # pos_emb is (b, t, 512)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        pos_emb = self.pos_emb_hook(pos_emb)

        # x is (b, t, 512)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Process through transformer blocks
        for (block_i, block) in enumerate(self.transformer.h):
            x = block(x)
            x = self.residual_hook[block_i](x)
            # output is (b, t, 512)
        # end for

        # Layer norm
        # output is (b, t, 512)
        x = self.transformer.ln_f(x)
        x = self.pre_logits_hook(x)

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            # output is (b, 1, voc_size)
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None
        # end if

        return logits, loss
    # end def forward

    def crop_block_size(self, block_size):
        """
        Reduce the block size of the model if necessary.

        This performs model surgery to decrease the block size, which might be needed
        when loading a pretrained model with a larger block size than required.

        Args:
            block_size (int): New block size (must be smaller than the current one)
        """
        # Model surgery to decrease the block size if necessary,
        # e.g. we may load the GPT-2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
            # end if
        # end for
    # end crop_block_size

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        Load a pretrained GPT model from Hugging Face.

        This method initializes a GPT model with the architecture matching the specified
        model type, then loads pretrained weights from Hugging Face's transformers library.

        Args:
            model_type (str): One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            override_args (dict, optional): Arguments to override in the config (only 'dropout' supported)

        Returns:
            GPT: Initialized model with pretrained weights
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # Only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]

        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints

        # We can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # end if

        # Create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # Init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # This means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
            # end if
        # end for

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure the optimizer for training.

        This method sets up an AdamW optimizer with weight decay applied only to
        appropriate parameters (weights but not biases or layer norms).

        Args:
            weight_decay (float): Weight decay coefficient
            learning_rate (float): Learning rate
            betas (tuple): Adam betas parameters (β1, β2)
            device_type (str): Device type ('cuda' or 'cpu')

        Returns:
            torch.optim.AdamW: Configured optimizer
        """
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups. Any parameter that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    # end def configure_optimizers

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS.

        This method calculates how efficiently the model is using the available
        computational resources during training.

        Args:
            fwdbwd_per_iter (int): Number of forward-backward passes per iteration
            dt (float): Time per iteration in seconds

        Returns:
            float: Model FLOPs utilization as a fraction of theoretical peak
        """
        # First estimate the number of flops we do per iteration.
        # See PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # Express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    # end def estimate_mfu

    def load_safetensors(self, filepath):
        """
        Load model weights from a safetensors file.

        This method loads weights from a safetensors file into the model.
        Safetensors is a safer alternative to PyTorch's native format,
        preventing arbitrary code execution during model loading.

        Args:
            filepath (str): Path to the safetensors file

        Returns:
            self: The model instance with loaded weights
        """
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "Could not import safetensors. Please install safetensors with: pip install safetensors"
            )

        print(f"Loading weights from safetensors file: {filepath}")

        # Load the safetensors file
        state_dict = load_file(filepath)

        # Get the model's current state dict
        model_state_dict = self.state_dict()
        model_keys = set(model_state_dict.keys())
        file_keys = set(state_dict.keys())

        # Check for missing and unexpected keys
        missing_keys = model_keys - file_keys
        unexpected_keys = file_keys - model_keys

        if missing_keys:
            print(f"Warning: Missing keys in safetensors file: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in safetensors file: {unexpected_keys}")

        # Load weights for matching keys
        for key in model_keys.intersection(file_keys):
            if model_state_dict[key].shape != state_dict[key].shape:
                print(
                    f"Warning: Shape mismatch for {key}. Expected {model_state_dict[key].shape}, got {state_dict[key].shape}")
                continue
            model_state_dict[key].copy_(state_dict[key])

        # Load the state dict into the model
        self.load_state_dict(model_state_dict)

        print(f"Successfully loaded weights from {filepath}")
        return self
    # end def load_safetensors

    # Create a mapping from move notation to ID
    # The vocabulary size is 61: BOS token (0) + 60 possible moves (1-60)
    @staticmethod
    def create_move_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Create a mapping from move notation to ID.

        The mapping includes:
        - BOS (Beginning of Sequence) token with ID 0
        - All possible moves on an 8x8 board with IDs 1-60

        Returns:
            Tuple[Dict[str, int], Dict[int, str]]: Mapping from move notation to ID
        """
        # Create a dictionary to store the mapping
        move_to_id = {"BOS": 0}  # BOS token has ID 0

        # Generate all possible move notations (a1-h8)
        id_counter = 1
        for col in range(8):  # a-h
            for row in range(8):  # 1-8
                # Move not possible on the centered square
                if (3 <= col <= 4) and (3 <= row <= 4):
                    continue
                # end if
                notation = chr(97 + col) + str(row + 1)
                move_to_id[notation] = id_counter
                id_counter += 1
            # end for
        # end for

        id_to_move = {i:m for m, i in move_to_id.items()}

        return move_to_id, id_to_move
    # end create_move_mapping

    @staticmethod
    def to_idx(sequence: List[str], add_pos: bool = True) -> List[int]:
        """
        Convert sequence to ID.

        Args:
            sequence (List[str]): Sequence to convert to ID
            add_pos (bool): If True, add position to sequence

        Returns:
            List[int]: Converted sequence
        """
        move_to_id, _ = GPT.create_move_mapping()
        if add_pos:
            return [
                0,
                *[move_to_id[m.lower()] for m in sequence],
            ]
        else:
            return [
                move_to_id[m.lower()]
                for m in sequence
            ]
        # end if
    # end def to_idx

    @staticmethod
    def to_moves(idx: List[int]) -> List[str]:
        """
        Convert ID to move.

        Args:
            idx (List[int]): ID to convert to move

        Returns:
            List[str]: Converted move (excluding BOS tokens)
        """
        _, id_to_move = GPT.create_move_mapping()
        # Filter out BOS tokens (ID 0) before converting to move notations
        return [
            id_to_move[i] for i in idx if i != 0  # Skip BOS tokens
        ]
    # end to_moves

    def generate_moves(
            self,
            sequence: List[str],
            max_new_tokens: int,
            device: torch.device,
            add_pos: bool = True,
            temperature: float = 1.0,
            top_k: int = None,
    ) -> List[str]:
        """
        Generate moves from sequence.

        Args:
            sequence (List[str]): Sequence to generate
            max_new_tokens (int): Maximum number of tokens to generate
            device (torch.device): Device to use
            add_pos (bool): If True, add position to sequence
            temperature (float): Temperature parameter
            top_k (int): If specified, only generate tokens with this many tokens
            device (torch.device): Device to use
        """
        # Transform sequence to idx
        idx = GPT.to_idx(sequence, add_pos=add_pos)

        # Make tensor
        move_idx = torch.LongTensor(idx).unsqueeze(0).to(device)

        # Generate tokens
        # gen_seq is (seq_len + max_new_token)
        gen_seq: torch.Tensor = self.generate(
            idx=move_idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )[0]

        # Transform into str sequence
        return GPT.to_moves(gen_seq.tolist())
    # end generate_tokens

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text by sampling from the model's distribution.

        This method takes a conditioning sequence and generates new tokens
        autoregressively by sampling from the model's predicted distribution.

        Args:
            idx (torch.Tensor): Starting token indices of shape (batch_size, seq_len)
            max_new_tokens (int): Number of new tokens to generate
            temperature (float): Sampling temperature (1.0 = no change, <1.0 = less random, >1.0 = more random)
            top_k (int, optional): If specified, only sample from the top k most probable tokens

        Returns:
            torch.Tensor: Generated token indices of shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)

            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # end if

            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        # end for

        return idx
    # end def generate

# end class GPT