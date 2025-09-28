"""
Copyright (C) 2025 Nils Schaetti

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
import collections
# Imports
import os
import pickle
import random
import glob
from typing import List, Tuple, Dict

import numpy as np
import torch
from seaborn._core.moves import Move
from torch.utils.data import Dataset


MAP_START_INDEX = 1


# Game Dataset
class GameDataset(Dataset):
    """
    Game Dataset
    """

    # Constructor
    def __init__(
            self,
            data_dir: str,
            max_len: int = 60,
            block_size: int = 60,
            split: str = 'train',
            ood_perc: float = 0.,
            num_samples: int = 10000,
            padding_int: int = 0,  # end def __init__
    ):
        """
        Othello Dataset

        Args:
             data_dir: path to Othello dataset
             max_len (int, optional): max length of game sequence
             block_size (int, optional): block size of game sequence
             split: train or test
             ood_perc: percentage of Othello dataset to use
             num_samples: number of samples to use
             padding_int (int): padding integer for Othello dataset
        """
        # Super
        super().__init__()

        # Properties
        self.data_dir = data_dir
        self.max_len = max_len
        self.block_size = block_size
        self.ood_perc = ood_perc
        self.split = split
        self.num_samples = num_samples
        self.padding_int = padding_int

        # Load the dataset
        # self.data, self.stoi, self.itos = self.load_data()
        self.data = self.load_data()
    # end def __init__

    # region PUBLIC

    def load_data(self) -> List[np.array]:
        # Data dir for the specified split (train or val)
        data_dir = os.path.join(self.data_dir, self.split)

        # Pattern for bin files
        pattern = "*.bin"

        # Find all matching bin files
        bin_files = glob.glob(os.path.join(data_dir, pattern))

        # Mapping dicts
        # idx_index = MAP_START_INDEX
        # stoi = dict()

        if not bin_files:
            # If no bin files found in the specified directory, print an error message
            raise FileNotFoundError(
                f"No bin files found in {data_dir}. "
                f"Error: No bin files found in {data_dir}. Make sure the data directory contains "
                f"'train' and 'val' folders with bin files."
            )  # end if
        else:
            # Load all bin files and combine their data
            game_sequences = []
            for bin_file in bin_files:
                with open(bin_file, 'rb') as f:
                    sequences = pickle.load(f)
                    game_sequences.extend(sequences)
                    # for seq in sequences:
                    #     for v in np.unique(seq):
                    #         if v not in stoi:
                    #             stoi[v] = idx_index
                    #             idx_index += 1
                    #         # end if
                    #     # end for
                    # # end for
                # end with
            # end for
        # end if

        # Build itos
        # itos = {i:v for v,i in stoi.items()}

        # Limit samples
        if self.num_samples > 0:
            game_sequences = game_sequences[:self.num_samples]
        # end if

        # Store in the appropriate global variable
        return game_sequences
    # end def load_data

    # Get a raw sequence
    def get_sequence(self, idx: int) -> List[str]:
        """
        Get a sequence as a list of strings.

        Args:
            idx (int): index of a sequence
        """
        game_sequences: np.bytes_ = self.data[idx]
        return [s.decode() for s in game_sequences]
    # end def get_sequence

    # Transform to str-move
    def to_moves(self, moves_i: Tuple[List[int], torch.LongTensor]) -> List[str]:
        """
        Transform a sequence of moves to a list of moves.

        Args:
            moves_i (List[int]): list of moves

        Returns:
            List[str]: list of moves
        """
        if type(moves_i) == list:
            return [self.itos[mi].decode() for mi in moves_i if mi != self.padding_int]
        elif type(moves_i) == torch.Tensor:
            return [self.itos[mi].decode() for mi in moves_i.tolist() if mi != self.padding_int]
        else:
            raise NotImplementedError
        # end if│ ❱ 157 │   │   │   return [self.itos[mi].decode() for mi in moves_i if mi != self.padding_int]                                                                                                                                        │
    # end def to_moves

    # Transform to an int-move
    def to_indices(self, moves: List[str]) -> List[int]:
        """
        Transform a sequence of moves to a list of indices.

        Args:
            moves (List[str]): list of moves

        Returns:
            List[int]: list of indices
        """
        return [self.stoi[ms] for ms in moves]
    # end def to_indices

    # endregion PUBLIC

    # region OVERWRITE

    def __len__(self):
        """
        Return length of dataset
        """
        return len(self.data)  # end def __len__
    # end __len__

    def __getitem__(self, idx):
        """
        Get item from dataset
        """
        # Get a game sequence
        game_sequence: np.bytes_ = self.data[idx]

        # Transforme into a list of int
        # game_sequence: List[int] = [self.stoi[x] for x in game_sequence]

        # Get a random position
        ix = random.randint(1, len(game_sequence) - 1)

        # Get subsequence and pad
        x = game_sequence[:ix]
        y = game_sequence[:ix + 1]
        # x = [self.padding_int] * (self.block_size - ix) + game_sequence[:ix]
        # y =  [self.padding_int] * (self.block_size - ix - 1) + game_sequence[:ix+1]

        # Decode
        x = [s.decode() for s in x]
        y = [s.decode() for s in y]

        x = ["<pad>"] * (self.block_size - ix) + x
        y = ["<pad>"] * (self.block_size - ix - 1) + y

        # Transform in text
        x = " ".join(x)
        y = " ".join(y)

        # Get x and y
        # x = torch.tensor(x, dtype=torch.long)
        # y = torch.tensor(y, dtype=torch.long)

        # Check that x and y have the same sizes
        # if x.shape[0] != y.shape[0]:
        #     raise ValueError(
        #         f"x and y must have the same length. x is {x.shape[0]} but y is {y.shape[0]}, "
        #         f"x: {x}, "
        #         f"y: {y}, "
        #         f"game_sequence: {game_sequence} ({len(game_sequence)}) "
        #         f"ix: {ix}"
        #     )
        # # end if
        #
        # # Must be 60
        # if x.shape[0] != 60:
        #     raise ValueError(
        #         f"x and y must be 60 but x is {x.shape[0]} but y is {y.shape[0]}, "
        #         f"x: {x}, "
        #         f"y: {y}, "
        #         f"game_sequence: {game_sequence} ({len(game_sequence)}), "
        #         f"ix: {ix}"
        #     )
        # # end if

        return x, y
    # end __getitem__

    # endregion OVERWRITE

# end class GameDataset

