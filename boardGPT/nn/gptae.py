

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
from torch.nn import functional as F
import json
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .hooks import HookPoint
from .layer_norm import LayerNorm
from .block import Block, MLP, CausalSelfAttention
from .register import ActivationRecorder


class GPTAEConfig:
    vocab_size = 61
    block_size = 60
    n_embd = 512
    n_layer = 2
    n_head = 8
    dropout = 0.1
    bias = False
    n_latent = 32
# end Config


# GPT Autoencodeur
class GPTAE(nn.Module):

    def __init__(
            self,
            config,
            use_flash: bool = True
    ):
        """
        ...
        """
        super().__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config

        # Create transformer encodeur components
        self.encoder = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),  # position embeddings
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config, use_flash) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias)
            )
        )

        # Projection to latent space (per token)
        self.to_latent = nn.Linear(config.n_embd, config.n_latent)

        # Projection from latent
        self.from_latent = nn.Linear(config.n_latent, config.n_embd)

        # Create transformer decoder components
        self.decoder = nn.ModuleDict(
            dict(
                wpe=nn.Embedding(config.block_size, config.n_embd),  # position embeddings
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config, use_flash) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),  # final layer norm
            )
        )

        # --- Output layer ---
        self.output_head = nn.Linear(config.n_embd, config.vocab_size)
    # end def __init__

    def forward_transformer(
            self,
            module: nn.Module,
            idx: torch.LongTensor
    ):
        device = idx.device

        # Batch size, sequence length
        b, t = idx.size()

        # Check max sequence length
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Pos is (t)
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # Forward the GPT model itself
        # tok_emb is (b, t, 512)
        tok_emb = module.wte(idx)  # token embeddings of shape (b, t, n_embd)

        # pos_emb is (b, t, 512)
        pos_emb = module.wpe(pos)  # position embeddings of shape (t, n_embd)

        # x is (b, t, 512)
        x = module.drop(tok_emb + pos_emb)

        # Process through transformer blocks
        for (block_i, block) in enumerate(module.h):
            x = block(x=x)
            # output is (b, t, 512)
        # end for

        # Layer norm
        # output is (b, t, 512)
        x = module.ln_f(x)

        return x
    # end def forward_encoder

    def forward_decoder(
            self,
            x: torch.Tensor,
    ):
        """
        ...
        """
        device = x.device

        # Batch size, sequence length
        b, t, e = x.size()
        print(f"b: {b}, t: {t}, e: {e}")
        # Check max sequence length
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Pos is (t)
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # pos_emb is (b, t, 512)
        pos_emb = self.decoder.wpe(pos)  # position embeddings of shape (t, n_embd)
        print(f"pos_emb: {pos_emb.shape}")
        # x is (b, t, 512)
        x = self.decoder.drop(x + pos_emb)
        print(f"x: {x.shape}")
        # Process through transformer blocks
        for (block_i, block) in enumerate(self.decoder.h):
            x = block(x=x)
            # output is (b, t, 512)
        # end for

        # Layer norm
        # output is (b, t, 512)
        x = self.decoder.ln_f(x)

        return x
    # end def forward_decoder

    def forward(self, idx: torch.LongTensor):
        """
        Args:
            idx (torch.LongTensor): Input sequence indices.

        Returns:
            logits: FloatTensor (batch, seq_len, vocab_size)
        """
        print(f"idx.shape: {idx.shape}")
        # Encoder
        x = self.forward_transformer(
            module=self.encoder,
            idx=idx
        )
        print(f"x.shape: {x.shape}")
        # Compression
        z = self.to_latent(x)  # (B, L, n_latent)
        print(f"z.shape: {z.shape}")
        # Expansion
        x = self.from_latent(z)  # (B, L, 512)
        print(f"x.shape: {x.shape}")
        # Decoder
        x = self.forward_decoder(x)  # (B, L, 512)

        # Output logits
        logits = self.output_head(x)  # (B, L, vocab_size)

        return logits
    # end def forward

# end class GPTAE


