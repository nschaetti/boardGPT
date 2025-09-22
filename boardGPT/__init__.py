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

from boardGPT.games.othello.othello_simulator import OthelloGame
from boardGPT.games.othello.othello_utils import verify_game

def othello(moves_str: str) -> OthelloGame:
    """
    Create an OthelloGame object from a string of moves.
    
    Args:
        moves_str (str): A string of moves in standard notation, separated by spaces or commas
                         (e.g., "d3 c4 e3" or "d3,c4,e3")
    
    Returns:
        OthelloGame: An OthelloGame object with the moves applied
        
    Raises:
        ValueError: If the move sequence is not possible (contains invalid moves)
    """
    # Parse the input string into a list of moves
    if ',' in moves_str:
        # Handle comma-separated moves
        moves = [move.strip() for move in moves_str.split(',')]
    else:
        # Handle space-separated moves
        moves = [move.strip() for move in moves_str.split()]
    
    # Remove any empty moves
    moves = [move for move in moves if move]
    
    # Verify that the move sequence is valid
    is_valid, invalid_moves = verify_game(moves)
    
    if not is_valid:
        raise ValueError(f"Invalid move sequence. The following moves are invalid: {', '.join(invalid_moves)}")
    
    # Create an OthelloGame object with the moves applied
    game = OthelloGame.load_moves(moves)
    
    return game

