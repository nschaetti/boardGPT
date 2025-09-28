"""
Copyright (C) 2025 Nils Schaetti <n.schaetti@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


# Imports
import math
import inspect
from dataclasses import dataclass
from distutils.command.config import config
from typing import Dict, Tuple, List, Optional, Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from boardGPT.nn.register import ActivationRecorder
from boardGPT.nn import GPT


class GameGPT(GPT):

     # Generate moves
     def generate_moves(
             self,
             sequence: List[str],
             max_new_tokens: int,
             device: torch.device,
             add_pos: bool = True,
             temperature: float = 1.0,
             top_k: int = None,
             recorder: ActivationRecorder = None,
             to_return: List[str] = None  # end def generate_moves
     ) -> Tuple[List[str], Any]:
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
              recorder (ActivationRecorder): Recorder to record moves
              to_return (List[str]): List of information to return.
          """
          # Transform sequence to idx
          idx = GPT.to_idx(sequence, add_pos=add_pos)

          # Make tensor
          move_idx = torch.LongTensor(idx).unsqueeze(0).to(device)

          # Generate tokens
          # gen_seq is (seq_len + max_new_token)
          gen_seq, ret_list = self.generate(
               idx=move_idx,
               max_new_tokens=max_new_tokens,
               temperature=temperature,
               top_k=top_k,
               recorder=recorder,
               to_return=to_return
          )
          gen_seq = gen_seq[0]

          # Transform into str sequence
          return (
               GPT.to_moves(gen_seq.tolist()),
               ret_list
          )
     # end generate_tokens

     def forward(
             self,
             idx: torch.LongTensor,
             targets: torch.Tensor = None,
             recorder: ActivationRecorder = None,
             to_return: Optional[List[str]] = None,  # end def forward
     ):
          """
          Forward pass through the model.

          Args:
              idx (torch.Tensor): Input token indices of shape (batch_size, seq_len)
              targets (torch.Tensor, optional): Target token indices of shape (batch_size, seq_len)
              recorder (ActivationRecorder, optional): Activation recorder
              to_return (List[str]): List of object to return

          Returns:
              tuple: (logits, loss) where logits is the output predictions and loss is the
                    cross-entropy loss if targets are provided, otherwise None
          """
          x, logits, obj_to_return = super().forward(
               idx=idx,
               recorder=recorder,
               to_return=to_return
          )

          if targets is not None:
               # If we are given some desired targets also calculate the loss
               loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)  # end if
          else:
               # Inference-time mini-optimization: only forward the lm_head on the very last position
               # output is (b, 1, voc_size)
               logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
               loss = None
          # end if

          # Add
          obj_to_return.append(logits)
          obj_to_return.append(loss)

          return x, logits, loss, obj_to_return
     # end def forward

# end class GameGPT