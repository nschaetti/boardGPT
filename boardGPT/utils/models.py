
from typing import Union, Tuple
import json
from safetensors.torch import safe_open
from boardGPT.models import GPT, GPTConfig


def load_safetensors(
        path: str,
        config_path: str = None
) -> Tuple[GPT, GPTConfig]:
    """
    Load a model from a safe tensor file.

    Args:
        path (str): Path to the safe tensor file.
        config_path (str, optional): Path to the JSON configuration file.
            If not provided, assumes it's the same as path but with .json extension.

    Returns:
        A GPT model loaded from a safe tensor file.
    """
    # Determine the config path if not provided
    if config_path is None:
        config_path = path.rsplit('.', 1)[0] + '.json'
    # end if
    
    # Load the model configuration from JSON
    with open(config_path, 'r') as f:
        model_args = json.load(f)
    # end with
    
    # Create a GPTConfig object from the configuration
    gptconf = GPTConfig(**model_args)
    
    # Initialize a GPT model with this config
    model = GPT(gptconf)
    
    # Load the weights from the safetensors file
    model.load_safetensors(path)
    
    return model, model_args
# end load_safetensors

