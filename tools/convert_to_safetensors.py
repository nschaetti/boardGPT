
import argparse
import json

import torch
from safetensors.torch import save_model
from boardGPT.models import GPTConfig, GPT


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file")
    parser.add_argument("--safetensors", type=str, required=True, help="Output safetensors file")
    parser.add_argument("--config", type=str, required=True, help="Output JSON config file")
    args = parser.parse_args()

    # Set up model arguments from configuration
    model_args = dict(
        n_layer=8,
        n_head=8,
        n_embd=523,
        block_size=61,
        bias=False,
        dropout=0.0
    )

    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
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

    # Save tensor file
    save_model(model, args.safetensors)

    # Save json config
    with open(args.config, 'w') as f:
        json.dump(model_args, f)
    # end with
# end if

