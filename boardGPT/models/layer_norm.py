"""
Custom LayerNorm implementation for the boardGPT model.

This module provides a custom implementation of LayerNorm with optional bias parameter.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """
    Custom LayerNorm implementation with optional bias parameter.

    This implementation differs from PyTorch's built-in LayerNorm by allowing
    the bias parameter to be disabled, which can improve performance in some cases.

    Args:
        ndim (int): The feature dimension to normalize over
        bias (bool): Whether to include a bias parameter
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    # end def __init__

    def forward(self, input):
        """
        Apply layer normalization to the input.

        Args:
            input (torch.Tensor): Input tensor of shape (..., ndim)

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    # end forward

# end class LayerNorm