import sys
import os
from typing import List, Tuple, Optional

from boardGPT.simulators.othello import OthelloGame

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


def game_to_board(moves: List[str]) -> List[int]:
    """
    Transform a game (as a list of str) into a board representation as a list of int.
    
    Args:
        moves (List[str]): List of moves in standard notation (e.g., ['c4', 'd3', ...])
        
    Returns:
        List[int]: Board representation as a 1D list where:
            - 0 = empty
            - 1 = white
            - 2 = black
            - The first entry is "a1", the second "a2", etc.
            
    Raises:
        ValueError: If a move is invalid for both players
    """
    # Create a new Othello game
    game = OthelloGame()
    
    # Track the current player (Black starts in Othello)
    current_player = game.BLACK
    
    # Replay the game by making each move
    for move in moves:
        try:
            # Convert notation to coordinates
            row, col = game.notation_to_coords(move)
            
            # Check if the move is valid for the current player
            if not game.is_valid_move(row, col):
                # If not valid for current player, switch player and check again
                game.switch_player()
                
                # Check if the move is valid for the other player
                if not game.is_valid_move(row, col):
                    # Not valid for either player, raise an exception
                    raise ValueError(f"Move {move} is invalid for both players")
                # end if
            # end if
            
            # Make the move
            game.make_move(row, col)
        except ValueError as e:
            # Re-raise ValueError for invalid moves for both players
            if "invalid for both players" in str(e):
                raise
            # Skip moves with invalid notation
            continue
        # end try-except
    # end for
    
    # Convert the 2D board to a 1D list in the required order
    board_1d = []
    for col in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
        for row in range(game.SIZE):
            # Position
            pos = f"{col}{row+1}"
            print(f"pos: {pos}")
            # Get the coordinate
            r, c = game.notation_to_coords(pos)

            # Get the piece at this position using the board's get_piece method
            # Note: In OthelloGame, BLACK=1, WHITE=2, but the requirement is WHITE=1, BLACK=2
            piece = game.board.get_piece(r, c)

            # Convert from OthelloGame representation to required representation
            if piece == game.BLACK:
                board_1d.append(2)  # BLACK = 2
            elif piece == game.WHITE:
                board_1d.append(1)  # WHITE = 1
            else:
                board_1d.append(0)  # EMPTY = 0
            # end for
        # end for
    # end for

    assert len(board_1d) == 64, f"Error: board representation must contain 64 output"
    
    return board_1d
# end game_to_board
