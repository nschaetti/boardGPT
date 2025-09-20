"""
Copyright (C) 2025 boardGPT Contributors

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

import os
import pickle
from typing import Tuple, List, Union

import numpy as np


def load_othello_dataset(
        data_dir: str,
        split: str,
        data_filename: str,
        flatten: bool = True,
        log: bool = True,  # end def load_othello_dataset
) -> Union[List[int], List[List[int]]]:
    """
    ...
    """
    if log: print(f"Loading {split} data into memory...")

    # Data dir for the specified split (train or val)
    data_dir = os.path.join(data_dir, split)

    # Pattern for bin files
    pattern = "*.bin"

    # Find all matching bin files
    import glob
    bin_files = glob.glob(os.path.join(data_dir, pattern))

    if not bin_files:
        # If no bin files found in the specified directory, print an error message
        print(
            f"Error: No bin files found in {data_dir}. Make sure the data directory contains 'train' and "
            f"'val' folders with bin files."
        )

        # Fallback to old method if no matching files found
        fallback_data_dir = os.path.join("data", "othello")

        print(f"Falling back to {os.path.join(fallback_data_dir, data_filename)}")

        with open(os.path.join(fallback_data_dir, data_filename), 'rb') as f:
            game_sequences = pickle.load(f)  # end with
        # end with
    else:
        # Load all bin files and combine their data
        if log: print(f"Found {len(bin_files)} bin files for {split} split")
        game_sequences = []
        for bin_file in bin_files:
            if log: print(f"Loading {bin_file}...")
            with open(bin_file, 'rb') as f:
                sequences = pickle.load(f)
                game_sequences.extend(sequences)  # end with
            # end with
        # end for
    # end if

    # Concatenate game sequences
    if flatten:
        return [x for sublist in game_sequences for x in sublist.tolist()]  # end if
    else:
        return game_sequences  # end else
    # end if
# end def load_othello_dataset