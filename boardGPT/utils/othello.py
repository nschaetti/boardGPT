import sys
import os
from typing import List, Tuple, Optional

# Add the project root to the path to import from simulators
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from simulators.othello import OthelloGame

def verify_game(moves: List[str]) -> Tuple[bool, List[str]]:
    """
    Verify the validity of an Othello game by checking if all moves are valid.
    
    Args:
        moves (List[str]): List of moves in standard notation (e.g., ['c4', 'd3', ...])
        
    Returns:
        Tuple[bool, List[str]]: 
            - First element is True if all moves are valid, False otherwise
            - Second element is an empty list if all moves are valid, 
              or a list of invalid moves if any are invalid
    """
    # Create a new board to replay and validate the game
    board = OthelloGame()
    invalid_moves = []
    
    # Track the current player (Black starts in Othello)
    current_player = board.BLACK
    
    for i, move in enumerate(moves):
        # Check move corresponds to a valid square (a1 to h8)
        if not (len(move) == 2 and 'a' <= move[0].lower() <= 'h' and '1' <= move[1] <= '8'):
            invalid_moves.append(move)
            continue
            
        # Convert notation to coordinates
        try:
            row, col = board.notation_to_coords(move)
        except ValueError:
            # Invalid notation
            invalid_moves.append(move)
            continue
        
        # Set the current player
        board.current_player = current_player
        
        # Check if the move is legal for the current player
        if not board.is_valid_move(row, col):
            # If not valid for current player, check if it's valid after a pass
            board.current_player = board.WHITE if current_player == board.BLACK else board.BLACK
            
            # Check if the move is valid for the other player
            if not board.is_valid_move(row, col):
                # Not valid for either player
                invalid_moves.append(move)
                continue
                
            # Valid for the other player (implies a pass)
            current_player = board.current_player
        
        # Make the move
        board.make_move(row, col)
        
        # Update current player for next move
        current_player = board.WHITE if current_player == board.BLACK else board.BLACK
    
    # Return True if all moves are valid, False otherwise
    return len(invalid_moves) == 0, invalid_moves