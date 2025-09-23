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

# Imports
from typing import Union, List
from boardGPT.games.othello.othello_simulator import OthelloGame
from boardGPT.games.othello.othello_utils import verify_game


def othello(moves_str: Union[str, List[str]]) -> OthelloGame:
    """
    Create an OthelloGame object from a string of moves.
    
    Args:
        moves_str (Union[str, List[str]]): list of moves as a string or list of strings
    
    Returns:
        OthelloGame: An OthelloGame object with the moves applied
        
    Raises:
        ValueError: If the move sequence is not possible (contains invalid moves)
    """
    # Parse the input string into a list of moves
    if isinstance(moves_str, str):
        if ',' in moves_str:
            # Handle comma-separated moves
            moves = [move.strip() for move in moves_str.split(',')]
        else:
            # Handle space-separated moves
            moves = [move.strip() for move in moves_str.split()]
        # end if
    else:
        moves = moves_str
    # end if
    
    # Remove any empty moves
    moves = [move for move in moves if move]
    
    # Verify that the move sequence is valid
    is_valid, invalid_moves = verify_game(moves)
    
    if not is_valid:
        raise ValueError(f"Invalid move sequence. The following moves are invalid: {', '.join(invalid_moves)}")
    # end if
    
    # Create an OthelloGame object with the moves applied
    game = OthelloGame.load_moves(moves)
    
    return game
# end othello

