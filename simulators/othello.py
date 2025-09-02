#!/usr/bin/env python3
"""
Generate valid Othello games.

This script generates valid Othello games and outputs them to a text file.
Each line in the output file represents one game, with moves separated by commas.
Moves are represented as column (a-h) + row (1-8), e.g., "d3", "e6".

The script implements the complete Othello game rules, including:
- Standard 8x8 board setup with initial four pieces in the center
- Legal move validation (must flip at least one opponent's piece)
- Piece flipping in all eight directions
- Game termination when neither player has valid moves
- Move notation conversion between board coordinates and standard notation

Usage:
    python othello.py --num-games 100 --output games.txt
"""

import argparse
import random
from typing import List, Tuple, Set


class OthelloBoard:
    """
    Represents an Othello game board and implements game rules.
    
    This class provides a complete implementation of the Othello game mechanics,
    including board representation, move validation, piece placement and flipping,
    player turn management, and game state evaluation.
    
    The board uses a 2D list to represent the 8x8 grid, with each cell containing
    one of three values: EMPTY (0), BLACK (1), or WHITE (2).
    
    Coordinates are zero-indexed, with (0,0) being the top-left corner of the board.
    Standard Othello notation is also supported, where 'a1' corresponds to (0,0).
    """

    # Board size (standard Othello board is 8x8)
    SIZE = 8

    # Player constants for board representation
    EMPTY = 0  # Empty cell
    BLACK = 1  # Black piece (typically goes first)
    WHITE = 2  # White piece

    # Eight directions for checking valid moves and flipping pieces
    # Format: (row_delta, column_delta)
    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),  # North-west, North, North-east
        (0, -1),           (0, 1),   # West, East
        (1, -1),  (1, 0),  (1, 1)    # South-west, South, South-east
    ]

    def __init__(self):
        """
        Initialize a new Othello board with the standard starting position.
        
        Creates an 8x8 empty board and sets up the initial four pieces in the center:
        - White pieces at positions (3,3) and (4,4)
        - Black pieces at positions (3,4) and (4,3)
        
        Black player moves first, as per standard Othello rules.
        """
        # Create an empty 8x8 board filled with EMPTY (0) values
        self.board = [[self.EMPTY for _ in range(self.SIZE)] for _ in range(self.SIZE)]
        # end for

        # Set up the initial four pieces in the center in the standard Othello pattern
        self.board[3][3] = self.WHITE  # d4 in standard notation
        self.board[3][4] = self.BLACK  # e4 in standard notation
        self.board[4][3] = self.BLACK  # d5 in standard notation
        self.board[4][4] = self.WHITE  # e5 in standard notation

        # Black moves first according to standard Othello rules
        self.current_player = self.BLACK

        # Keep track of moves made in standard notation (e.g., 'd3', 'e6')
        self.moves = []
    # end __init__

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """
        Return a list of valid moves for the current player.
        
        A valid move is one where placing a piece would flip at least one
        of the opponent's pieces. This method checks all board positions
        and returns coordinates of all valid moves.
        
        Returns:
            List[Tuple[int, int]]: A list of (row, col) tuples representing valid move positions
        """
        # Initialize empty list to store valid moves
        valid_moves = []

        # Check every position on the board
        for row in range(self.SIZE):
            for col in range(self.SIZE):
                # If the move is valid, add it to the list
                if self.is_valid_move(row, col):
                    valid_moves.append((row, col))
                # end if
            # end for
        # end for

        return valid_moves
    # end get_valid_moves

    def is_valid_move(self, row: int, col: int) -> bool:
        """
        Check if placing a piece at (row, col) is a valid move for the current player.
        
        A valid move in Othello must satisfy two conditions:
        1. The cell must be empty
        2. The move must flip at least one opponent's piece
        
        To flip an opponent's piece, there must be a straight line (horizontal, 
        vertical, or diagonal) of opponent's pieces, with the current player's 
        piece at the other end.
        
        Args:
            row (int): Row index (0-7)
            col (int): Column index (0-7)
            
        Returns:
            bool: True if the move is valid, False otherwise
        """
        # Rule 1: The cell must be empty
        if self.board[row][col] != self.EMPTY:
            return False
        # end if

        # Rule 2: The move must flip at least one opponent's piece
        # Determine the opponent's piece color
        opponent = self.WHITE if self.current_player == self.BLACK else self.BLACK

        # Check in all eight directions
        for dr, dc in self.DIRECTIONS:
            # Move one step in the current direction
            r, c = row + dr, col + dc
            
            # Check if we have at least one opponent's piece in this direction
            if (0 <= r < self.SIZE and 0 <= c < self.SIZE and
                    self.board[r][c] == opponent):
                
                # Continue in this direction looking for our own piece
                r += dr
                c += dc
                while 0 <= r < self.SIZE and 0 <= c < self.SIZE:
                    if self.board[r][c] == self.EMPTY:
                        # Empty cell, no flip possible in this direction
                        break
                    # end if
                    
                    if self.board[r][c] == self.current_player:
                        # Found our own piece at the other end, this is a valid move
                        return True
                    # end if
                    
                    # Continue in this direction
                    r += dr
                    c += dc
                # end while
            # end if
        # end for

        # No valid flips found in any direction
        return False
    # end is_valid_move

    def make_move(self, row: int, col: int) -> bool:
        """
        Make a move at (row, col) for the current player.
        
        This method:
        1. Validates the move
        2. Places the player's piece at the specified position
        3. Flips all opponent's pieces that are captured by this move
        4. Records the move in standard notation
        5. Switches to the other player
        
        Args:
            row (int): Row index (0-7)
            col (int): Column index (0-7)
            
        Returns:
            bool: True if the move was successful, False if invalid
        """
        # First check if the move is valid
        if not self.is_valid_move(row, col):
            return False
        # end if

        # Place the current player's piece at the specified position
        self.board[row][col] = self.current_player

        # Determine the opponent's piece color for flipping
        opponent = self.WHITE if self.current_player == self.BLACK else self.BLACK

        # Check in all eight directions for pieces to flip
        for dr, dc in self.DIRECTIONS:
            # List to store coordinates of pieces to flip in this direction
            to_flip = []

            # Move one step in the current direction
            r, c = row + dr, col + dc
            
            # Collect all opponent's pieces in this direction
            while (0 <= r < self.SIZE and 0 <= c < self.SIZE and
                   self.board[r][c] == opponent):
                to_flip.append((r, c))
                r += dr
                c += dc

                # If we reach the edge or an empty cell, no flip in this direction
                if not (0 <= r < self.SIZE and 0 <= c < self.SIZE) or self.board[r][c] == self.EMPTY:
                    to_flip = []  # Clear the list as we can't flip these pieces
                    break
                # end if

                # If we reach our own piece, we can flip all pieces in between
                if self.board[r][c] == self.current_player:
                    break
                # end if
            # end while

            # Flip all opponent's pieces that were captured
            for flip_r, flip_c in to_flip:
                self.board[flip_r][flip_c] = self.current_player
            # end for
        # end for

        # Record the move in standard notation (e.g., 'e4')
        move_notation = self.coords_to_notation(row, col)
        self.moves.append(move_notation)

        # Switch to the other player for the next turn
        self.current_player = self.WHITE if self.current_player == self.BLACK else self.BLACK

        return True
    # end make_move

    def has_valid_moves(self) -> bool:
        """
        Check if the current player has any valid moves.
        
        Returns:
            bool: True if the current player has at least one valid move, False otherwise
        """
        # Use the get_valid_moves method and check if the list is non-empty
        return len(self.get_valid_moves()) > 0
    # end has_valid_moves

    def switch_player(self) -> None:
        """
        Switch to the other player.
        
        Changes the current_player from BLACK to WHITE or from WHITE to BLACK.
        """
        # Toggle between BLACK and WHITE
        self.current_player = self.WHITE if self.current_player == self.BLACK else self.BLACK
    # end switch_player

    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        
        The game is over when neither player has any valid moves.
        
        Returns:
            bool: True if the game is over, False otherwise
        """
        # First check if current player has valid moves
        if self.has_valid_moves():
            return False
        # end if

        # If current player has no moves, check if the opponent has moves
        # Temporarily switch player to check
        self.switch_player()
        has_moves = self.has_valid_moves()
        self.switch_player()  # Switch back to original player
        
        # Game is over if neither player has valid moves
        return not has_moves
    # end is_game_over

    def coords_to_notation(self, row: int, col: int) -> str:
        """
        Convert board coordinates to standard Othello notation.
        
        In standard notation:
        - Columns are labeled a-h from left to right
        - Rows are labeled 1-8 from top to bottom
        
        Args:
            row (int): Row index (0-7)
            col (int): Column index (0-7)
            
        Returns:
            str: Position in standard notation (e.g., 'e4' for row=3, col=4)
        """
        # Convert column index to letter (a-h) and row index to number (1-8)
        return chr(97 + col) + str(row + 1)
    # end coords_to_notation

    def notation_to_coords(self, notation: str) -> Tuple[int, int]:
        """
        Convert standard Othello notation to board coordinates.
        
        Args:
            notation (str): Position in standard notation (e.g., 'e4')
            
        Returns:
            Tuple[int, int]: (row, col) coordinates (e.g., (3, 4) for 'e4')
        """
        # Convert letter (a-h) to column index (0-7) and number (1-8) to row index (0-7)
        col = ord(notation[0]) - 97  # 'a' is ASCII 97, so 'a' -> 0, 'b' -> 1, etc.
        row = int(notation[1]) - 1   # '1' -> 0, '2' -> 1, etc.
        return row, col
    # end notation_to_coords

    def get_moves_notation(self) -> List[str]:
        """
        Return the list of moves made in the game in standard notation.
        
        Returns:
            List[str]: List of moves in standard notation (e.g., ['d3', 'c4', 'e3'])
        """
        # Return the stored list of moves
        return self.moves
    # end get_moves_notation


def generate_game() -> List[str]:
    """
    Generate a single valid Othello game and return the list of moves.
    
    This function simulates a complete Othello game by making random valid moves
    until the game is over (no player has valid moves). The game follows standard
    Othello rules.
    
    Returns:
        List[str]: List of moves in standard notation (e.g., ['d3', 'c4', 'e3'])
    """
    # Create a new Othello board with standard starting position
    board = OthelloBoard()

    # Continue making moves until the game is over
    while not board.is_game_over():
        # Get all valid moves for the current player
        valid_moves = board.get_valid_moves()

        if not valid_moves:
            # No valid moves for current player, switch to other player
            board.switch_player()
            continue
        # end if

        # Choose a random valid move from the available options
        row, col = random.choice(valid_moves)
        board.make_move(row, col)
    # end while

    # Return the list of moves made during the game
    return board.get_moves_notation()
# end generate_game


def generate_games(num_games: int) -> List[List[str]]:
    """
    Generate multiple Othello games.
    
    Args:
        num_games (int): Number of games to generate
        
    Returns:
        List[List[str]]: List of games, where each game is a list of moves in standard notation
    """
    # Initialize empty list to store the generated games
    games = []
    
    # Generate the specified number of games
    for _ in range(num_games):
        # Generate a single game and add it to the list
        game = generate_game()
        games.append(game)
    # end for
    
    return games
# end generate_games


def save_games(games: List[List[str]], output_file: str) -> None:
    """
    Save games to a text file, one game per line.
    
    Each game is saved as a comma-separated list of moves in standard notation.
    
    Args:
        games (List[List[str]]): List of games to save
        output_file (str): Path to the output file
    """
    # Open the output file for writing
    with open(output_file, 'w') as f:
        # Write each game as a comma-separated list of moves
        for game in games:
            f.write(','.join(game) + '\n')
        # end for
    # end with
# end save_games


def main():
    """
    Main function to parse arguments and generate games.
    
    Command-line arguments:
        --num-games: Number of games to generate (default: 10)
        --output: Output file path (default: othello_games.txt)
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Generate valid Othello games.')
    parser.add_argument(
        '--num-games',
        type=int,
        default=10,
        help='Number of games to generate (default: 10)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='othello_games.txt',
        help='Output file path (default: othello_games.txt)'
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Generate the specified number of games
    print(f"Generating {args.num_games} Othello games...")
    games = generate_games(args.num_games)

    # Save the generated games to the output file
    save_games(games, args.output)
    print(f"Games saved to {args.output}")
# end main


if __name__ == '__main__':
    main()
# end if