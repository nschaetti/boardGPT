# Changes Summary

## Task Completed

The task was to transfer all experiment configuration from the beginning of `train.py` into a YAML file and add a parser to specify the configuration file.

## Changes Made

### 1. Created YAML Configuration File

Created a new file `config.yaml` in the project root directory that contains all the configuration parameters previously defined at the beginning of `train.py`. The configuration is organized into sections:

- I/O settings
- wandb logging settings
- Data settings
- Model architecture settings
- Optimizer settings
- Learning rate decay settings
- DDP settings
- System settings

### 2. Modified `train.py` to Support YAML Configuration

#### Added Required Imports

```python
import argparse
import yaml
```

#### Added Command Line Argument Parser

Added a function to parse command line arguments, allowing users to specify a custom configuration file:

```python
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
```

#### Added Configuration Loading Function

Added a function to load configuration from a YAML file:

```python
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
        print(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found. Using default configuration.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}")
        return {}
```

#### Maintained Global Variables for Compatibility

Kept the global variables for configuration parameters to maintain compatibility with the rest of the code, but now they're updated from the YAML configuration:

```python
# Define global variables for configuration
# I/O settings
out_dir = 'out'                  # Output directory for checkpoints and logs
eval_interval = 2000             # How often to evaluate the model
# ... (other configuration variables)

# Parse command line arguments
args = parse_args()

# Load configuration from YAML file
yaml_config = load_config(args.config)

# Update global variables with values from YAML file
for key, value in yaml_config.items():
    if key in globals():
        globals()[key] = value
    else:
        print(f"Warning: Unknown configuration parameter '{key}' in YAML file")

# Create config dictionary for logging
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}
```

### 3. Fixed Variable Scope Issues

Modified the `setup_training_environment` function to properly access global variables:

```python
def setup_training_environment():
    """
    Set up the training environment including distributed training, 
    random seeds, and device configuration.
    """
    global gradient_accumulation_steps, device
    
    # ... (rest of the function)
```

### 4. Fixed Data Loading Functions

Updated the code to use the appropriate data loading function based on the configuration:

```python
# In the main function
if board_game:
    X, Y = get_board_batch('train')  # Fetch the very first batch for board game
else:
    X, Y = get_batch('train')  # Fetch the very first batch for text
```

```python
# In the estimate_loss function
if board_game:
    X, Y = get_board_batch(split)
else:
    X, Y = get_batch(split)
```

```python
# In the training loop
if board_game:
    X, Y = get_board_batch('train')
else:
    X, Y = get_batch('train')
```

### 5. Created Documentation

Created a comprehensive documentation file `README_config.md` that explains:
- How to use the new configuration system
- All available configuration parameters
- Example configurations for different use cases
- Guidelines for creating custom configurations

## Benefits of the Changes

1. **Improved Reproducibility**: Configuration is now stored in a separate file, making it easier to reproduce experiments.
2. **Better Organization**: Configuration parameters are organized by category in the YAML file, making them easier to understand and modify.
3. **Easier Experimentation**: Users can create multiple configuration files for different experiments without modifying the code.
4. **Command Line Flexibility**: The `--config` argument allows users to specify different configuration files at runtime.
5. **Backward Compatibility**: The code still works with the original approach of using global variables, ensuring compatibility with existing code.

## Usage Example

To train a model using the default configuration:

```bash
python train.py
```

To train a model using a custom configuration:

```bash
python train.py --config custom_config.yaml
```