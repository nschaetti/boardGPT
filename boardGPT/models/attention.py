"""
Causal self-attention implementation for the boardGPT model.

This module provides a causal self-attention implementation with optional Flash Attention support.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

