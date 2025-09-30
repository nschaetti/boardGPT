

import torch
from boardGPT.nn import GPTAEConfig, GPTAE


# GPT auto-encodeur
config = GPTAEConfig()
model = GPTAE(config)

# A random sequence
batch_size = 2
seq_len = config.block_size
x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

# Put a sequence in a model
logits = model(x)

# Show input and output shapes
print("Input shape:", x.shape)             # (2, 60)
print("Output shape:", logits.shape)       # (2, 60, 61)

