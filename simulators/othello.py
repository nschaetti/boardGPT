#!/usr/bin/env python3
"""
Generate valid Othello games.

This script generates valid Othello games and outputs them to a binary file.
Each game is represented as a sequence of move IDs, starting with a BOS token (ID 0).
Moves are mapped from standard notation (e.g., "d3", "e6") to IDs (1-60).

The script implements the complete Othello game rules, including:
- Standard 8x8 board setup with initial four pieces in the center
- Legal move validation (must flip at least one opponent's piece)
- Piece flipping in all eight directions
- Game termination when neither player has valid moves
- Move notation conversion between board coordinates and standard notation

Usage:
    python othello.py --num-games 100 --output games.bin
"""

import argparse
import random
import numpy as np
import pickle
import sys
from collections import Counter
from typing import List, Tuple, Set, Dict
from rich.console import Console
from rich.text import Text
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


# Create a Rich console
console = Console()


class OthelloBoard:

    def __init__(
            self,
            size: int = 8,
            empty: int = 0,
            black: int = 1,
            white: int = 2
    ):
        """
        ...
        """
        # Set storage
        self.board = [[empty for _ in range(size)] for _ in range(size)]

        # Set up the initial four pieces in the center in the standard Othello pattern
        self.board[3][3] = white  # d4 in standard notation
        self.board[3][4] = black  # e4 in standard notation
        self.board[4][3] = black  # d5 in standard notation
        self.board[4][4] = white  # e5 in standard notation

        # Possible values
        self.values = [empty, black, white]

        # Board history
        self.history: List[List[Tuple[int, int, int]]] = []
        self.current_history: List[Tuple[int, int, int]] = []
    # end __init__

    # Set pieces
    def set_piece(self, row: int, col: int, value: int) -> None:
        """
        ...
        """
        assert value in self.values, f"Expected {value} to be in {self.values}"
        self.board[row][col] = value
        self.current_history.append((row, col, value))
    # end def set_piece

    def get_piece(self, row: int, col: int) -> int:
        """
        ...
        """
        return self.board[row][col]
    # end def get_piece

    def push_history(self):
        """
        ...
        """
        self.history.append(self.current_history)
        self.current_history = []
    # end def push_history

# end OthelloBoard


class OthelloGame:
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
        self.board = OthelloBoard(self.SIZE, self.EMPTY)

        # Black moves first according to standard Othello rules
        self.current_player = self.BLACK

        # Keep track of moves made in standard notation (e.g., 'd3', 'e6')
        self.moves: List[str] = []
        self.moves_player: List[int] = []
    # end __init__

    # region PUBLIC

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
        if self.board.get_piece(row, col) != self.EMPTY:
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
            if 0 <= r < self.SIZE and 0 <= c < self.SIZE and self.board.get_piece(r, c) == opponent:

                # Continue in this direction looking for our own piece
                r += dr
                c += dc
                while 0 <= r < self.SIZE and 0 <= c < self.SIZE:
                    if self.board.get_piece(r, c) == self.EMPTY:
                        # Empty cell, no flip possible in this direction
                        break
                    # end if

                    if self.board.get_piece(r, c) == self.current_player:
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
        self.board.set_piece(row, col, self.current_player)

        # Determine the opponent's piece color for flipping
        opponent = self.WHITE if self.current_player == self.BLACK else self.BLACK

        # Check in all eight directions for pieces to flip
        for dr, dc in self.DIRECTIONS:
            # List to store coordinates of pieces to flip in this direction
            to_flip = []

            # Move one step in the current direction
            r, c = row + dr, col + dc
            
            # Collect all opponent's pieces in this direction
            while 0 <= r < self.SIZE and 0 <= c < self.SIZE and self.board.get_piece(r, c) == opponent:
                to_flip.append((r, c))
                r += dr
                c += dc

                # If we reach the edge or an empty cell, no flip in this direction
                if not (0 <= r < self.SIZE and 0 <= c < self.SIZE) or self.board.get_piece(r, c) == self.EMPTY:
                    to_flip = []  # Clear the list as we can't flip these pieces
                    break
                # end if

                # If we reach our own piece, we can flip all pieces in between
                if self.board.get_piece(r, c) == self.current_player:
                    break
                # end if
            # end while

            # Flip all opponent's pieces that were captured
            for flip_r, flip_c in to_flip:
                self.board.set_piece(flip_r, flip_c, self.current_player)
            # end for
        # end for

        # Record the move in standard notation (e.g., 'e4')
        move_notation = self.coords_to_notation(row, col)

        # Add move and player
        self.moves.append(move_notation)
        self.moves_player.append(self.current_player)

        # Push board modification into history
        self.board.push_history()

        # Switch to the other player for the next turn
        self.switch_player()

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
        
        The game is over when:
        1. The board is full, or
        2. Both players pass consecutively (neither player has any valid moves)
        
        Returns:
            bool: True if the game is over, False otherwise
        """
        # Check if the board is full
        is_full = True
        for row in range(self.SIZE):
            for col in range(self.SIZE):
                if self.board.get_piece(row, col) == self.EMPTY:
                    is_full = False
                    break
            if not is_full:
                break
        
        if is_full:
            return True
        # end if
        
        # Check if both players have no valid moves
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

    def get_moves(self) -> List[str]:
        """
        Return the list of moves made in the game in standard notation.
        
        Returns:
            List[str]: List of moves in standard notation (e.g., ['d3', 'c4', 'e3'])
        """
        # Return the stored list of moves
        return self.moves
    # end get_moves_notation

    def set_moves(
            self,
            moves: List[str],
            moves_player: List[int],
    ) -> None:
        """
        ...
        """
        self.moves = moves
        self.moves_player = moves_player
    # end set_moves
    
    def validate_game(self) -> Tuple[bool, str]:
        """
        Validate the current game according to Othello rules.
        
        Checks:
        - The number of moves must equal the number of players, and game length must be between 1 and 60
        - Every move must correspond to a valid square on the Othello board (from a1 to h8)
        - No square can be played more than once during the game
        - Each player entry must be either 1 (black) or 2 (white)
        - At each step, the move played must be a legal move for the current player
        - After each move, the chosen square must belong to the current player
        - After t moves, the total number of discs on the board must equal 4 + t
        - Turns should normally alternate between black and white players
        - If the same player moves twice in a row, this must correspond to a valid pass
        - If the game ends at 60 moves, the final board must be completely full
        - If the game ends before 60 moves, it must be because both players had no legal moves
        - At all times, every square on the board must be either empty, black, or white
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message) - True if valid, False with error message if invalid
        """
        # Check if the game has moves
        if not self.moves:
            return False, "Game has no moves"
            
        # Check game length is between 1 and 60
        if len(self.moves) < 1 or len(self.moves) > 60:
            return False, f"Game length {len(self.moves)} is not between 1 and 60"
            
        # Check the number of moves equals the number of players
        if len(self.moves) != len(self.moves_player):
            return False, "Number of moves does not equal number of player entries"
            
        # Check each player entry is either 1 (black) or 2 (white)
        for i, player in enumerate(self.moves_player):
            if player != self.BLACK and player != self.WHITE:
                return False, f"Invalid player value {player} at move {i+1}"
                
        # Create a new board to replay and validate the game
        board = OthelloGame()
        played_squares = set()
        
        # Track the current player
        current_player = self.BLACK  # Black starts in Othello
        
        for i, move in enumerate(self.moves):
            # Check move corresponds to a valid square (a1 to h8)
            if not (len(move) == 2 and 'a' <= move[0] <= 'h' and '1' <= move[1] <= '8'):
                return False, f"Move {i+1} '{move}' is not a valid square (a1-h8)"
                
            # Convert notation to coordinates
            row, col = board.notation_to_coords(move)
            
            # Check no square is played more than once
            square = (row, col)
            if square in played_squares:
                return False, f"Square {move} at move {i+1} was already played"
            played_squares.add(square)
            
            # Check the player matches the expected player
            player = self.moves_player[i]
            if player != current_player:
                # If player doesn't match, check if it's a valid pass situation
                valid_moves = []
                for r in range(self.SIZE):
                    for c in range(self.SIZE):
                        if board.is_valid_move(r, c):
                            valid_moves.append((r, c))
                
                # If current player had valid moves, this is an invalid pass
                if valid_moves:
                    return False, f"Player {current_player} had valid moves but player {player} moved at move {i+1}"
                
                # Switch to the other player
                current_player = self.WHITE if current_player == self.BLACK else self.BLACK
                
                # Check if the new player matches
                if player != current_player:
                    return False, f"Invalid player sequence at move {i+1}"
            
            # Check if the move is legal for the current player
            board.current_player = player
            if not board.is_valid_move(row, col):
                return False, f"Move {i+1} '{move}' is not a legal move for player {player}"
                
            # Make the move
            board.make_move(row, col)
            
            # Check that after the move, the chosen square belongs to the current player
            if board.board.get_piece(row, col) != player:
                return False, f"After move {i+1}, square {move} does not belong to player {player}"
                
            # Count total pieces on the board
            total_pieces = 0
            for r in range(self.SIZE):
                for c in range(self.SIZE):
                    piece = board.board.get_piece(r, c)
                    if piece != self.EMPTY:
                        total_pieces += 1
                    # Check every square is either empty, black, or white
                    if piece not in [self.EMPTY, self.BLACK, self.WHITE]:
                        return False, f"Invalid piece value {piece} at position ({r},{c})"
            
            # Check total pieces equals 4 + number of moves
            if total_pieces != 4 + (i + 1):
                return False, f"After move {i+1}, total pieces {total_pieces} does not equal {4 + (i + 1)}"
                
            # Update current player for next move
            current_player = self.WHITE if player == self.BLACK else self.BLACK
        
        # Check end game conditions
        if len(self.moves) == 60:
            # If game ends at 60 moves, the board must be completely full
            empty_squares = 0
            for r in range(self.SIZE):
                for c in range(self.SIZE):
                    if board.board.get_piece(r, c) == self.EMPTY:
                        empty_squares += 1
            
            if empty_squares > 0:
                return False, f"Game ended at 60 moves but board has {empty_squares} empty squares"
        else:
            # If game ends before 60 moves, it must be because both players had no legal moves
            black_has_moves = False
            white_has_moves = False
            
            # Check if black has moves
            board.current_player = self.BLACK
            for r in range(self.SIZE):
                for c in range(self.SIZE):
                    if board.is_valid_move(r, c):
                        black_has_moves = True
                        break
                if black_has_moves:
                    break
            
            # Check if white has moves
            board.current_player = self.WHITE
            for r in range(self.SIZE):
                for c in range(self.SIZE):
                    if board.is_valid_move(r, c):
                        white_has_moves = True
                        break
                if white_has_moves:
                    break
            
            if black_has_moves or white_has_moves:
                return False, f"Game ended at {len(self.moves)} moves but at least one player still has valid moves"
        
        # All validation checks passed
        return True, "Game is valid"

    # endregion PUBLIC

    def __len__(self):
        """
        Return the number of moves made in the game in standard notation.
        """
        return len(self.moves)
    # end def __len__

    @staticmethod
    def load_moves(moves: List[str]) -> 'OthelloGame':
        """
        ...
        """
        # Instance class
        board = OthelloGame()

        # For each move
        for m in moves:
            # Transform to coordinates
            row, col = board.notation_to_coords(m)

            # Check if the move is valid
            if board.is_valid_move(row, col):
                # Make to move
                board.make_move(row, col)
            else:
                # Switch player
                board.switch_player()

                # Check that the move is valid
                assert board.is_valid_move(row, col), f"Invalid move detected: {m}"

                # Make the move
                board.make_move(row, col)
            # end if
        # end for

        return board
    # end load_moves

# end  OthelloBoard


def generate_game(seed: int = None, max_attempts: int = 100) -> List[str]:
    """
    Generate a single valid Othello game and return the list of moves.
    
    This function simulates an Othello game by making random valid moves
    until either the game is over (no player has valid moves) or the maximum
    number of moves is reached. It validates the generated game against all
    Othello rules and retries if the game is invalid.
    
    Args:
        seed (int, optional): Random seed for reproducibility. If None, no seed is set.
        max_attempts (int): Maximum number of attempts to generate a valid game (default: 100)
    
    Returns:
        List[str]: List of moves in standard notation (e.g., ['d3', 'c4', 'e3'])
    """
    global console

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # end if
    
    attempt = 0
    while attempt < max_attempts:
        # Create a new Othello board with standard starting position
        board = OthelloGame()

        # Continue making valid moves
        still_valid = True
        while still_valid:
            # Get all valid moves for the current player
            valid_moves = board.get_valid_moves()

            if not valid_moves:
                # No valid moves for current player, switch player
                board.switch_player()
                
                # Check if the other player also has no valid moves (game is over)
                if not board.has_valid_moves():
                    still_valid = False
                # end if
                
                continue
            # end if

            # Choose a random valid move from the available options
            row, col = random.choice(valid_moves)
            board.make_move(row, col)
        # end while

        # Validate the generated game
        is_valid, error_message = board.validate_game()
        
        if is_valid:
            # Return the list of moves made during the game
            return board.get_moves()
        else:
            console.print("[orange bold]Invalid Othello game! Retrying...[/]")
            # Try again with a different seed
            attempt += 1
            if seed is not None:
                # Use a different seed for each attempt
                random.seed(seed + attempt)
                np.random.seed(seed + attempt)
            # end if
        # end if
    # end while
    
    # If we've reached the maximum number of attempts, raise an exception
    raise RuntimeError(f"Failed to generate a valid Othello game after {max_attempts} attempts")
# end generate_game


def generate_games(num_games: int, seed: int = None, output_file: str = None, chunk_size: int = None) -> List[List[str]]:
    """
    Generate multiple Othello games.
    
    If output_file is provided, games will be saved in chunks as they are generated to save memory.
    Otherwise, all games will be kept in memory and returned.
    
    Args:
        num_games (int): Number of games to generate
        seed (int, optional): Random seed for reproducibility. If None, no seed is set.
        output_file (str, optional): Path to save the generated games. If None, games are only returned.
        chunk_size (int, optional): Number of games per file. If None or 0, all games are saved to a single file.
        
    Returns:
        List[List[str]]: List of games if output_file is None, otherwise an empty list
    """
    import os
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Initialize empty list to store the generated games
    games = []
    
    # If output_file is provided and chunk_size is specified, prepare for chunking
    save_in_chunks = output_file is not None and chunk_size is not None and chunk_size > 0
    chunk_counter = 0
    
    # Create a progress bar
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[bold green]{task.completed}/{task.total} games"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        # Create a task for game generation
        task = progress.add_task("[bold green]Generating games...", total=num_games)
        
        # Generate the specified number of games
        for i in range(num_games):
            # Generate a single game and add it to the list
            # If seed is provided, use a different seed for each game to ensure variety
            game_seed = None if seed is None else seed + i
            game = generate_game(game_seed)
            games.append(game)
            
            # Update the progress bar
            progress.update(task, advance=1)
            
            # If we're saving in chunks and have reached the chunk size, save and clear the games list
            if save_in_chunks and len(games) >= chunk_size:
                # Get the base filename and extension
                base_name, ext = os.path.splitext(output_file)
                
                # Create the chunk filename
                chunk_counter += 1
                chunk_filename = f"{base_name}_{chunk_counter}{ext}"
                
                # Save the chunk
                save_games(games, chunk_filename)
                
                # Clear the games list to free memory
                games = []
            # end if
        # end for
    
    # If there are remaining games and we're saving to a file
    if output_file is not None and games:
        if save_in_chunks:
            # Save the final chunk
            chunk_counter += 1
            base_name, ext = os.path.splitext(output_file)
            chunk_filename = f"{base_name}_{chunk_counter}{ext}"
            save_games(games, chunk_filename)
            
            # Return an empty list since games were saved to files
            return []
        else:
            # Save all games to a single file
            save_games(games, output_file)
            
            # Return an empty list since games were saved to a file
            return []
    
    # Return the games if they weren't saved to a file
    return games
# end generate_games


# Create a mapping from move notation to ID
# The vocabulary size is 61: BOS token (0) + 60 possible moves (1-60)
def create_move_mapping() -> Dict[str, int]:
    """
    Create a mapping from move notation to ID.
    
    The mapping includes:
    - BOS (Beginning of Sequence) token with ID 0
    - All possible moves on an 8x8 board with IDs 1-60
    
    Returns:
        Dict[str, int]: Dictionary mapping move notation to ID
    """
    # Create a dictionary to store the mapping
    move_to_id = {"BOS": 0}  # BOS token has ID 0
    
    # Generate all possible move notations (a1-h8)
    id_counter = 1
    for col in range(8):  # a-h
        for row in range(8):  # 1-8
            # Move not possible on the centered square
            if (3 <= col <= 4) and (3 <= row <= 4):
                continue
            # end if
            notation = chr(97 + col) + str(row + 1)
            move_to_id[notation] = id_counter
            id_counter += 1
        # end for
    # end for
    
    return move_to_id
# end create_move_mapping


def create_id_to_move_mapping() -> Dict[int, str]:
    """
    Create a mapping from ID to move notation.
    
    The mapping includes:
    - BOS (Beginning of Sequence) token with ID 0
    - All possible moves on an 8x8 board with IDs 1-60
    
    Returns:
        Dict[int, str]: Dictionary mapping ID to move notation
    """
    # Get the move_to_id mapping
    move_to_id = create_move_mapping()
    
    # Create the reverse mapping
    id_to_move = {id: move for move, id in move_to_id.items()}
    
    return id_to_move
# end create_id_to_move_mapping


def convert_ids_to_notation(game: List[int]) -> List[str]:
    """
    Convert a game represented as a list of move IDs to a list of move notations.
    
    Args:
        game (List[int]): Game as a list of move IDs
        
    Returns:
        List[str]: Game as a list of move notations
    """
    # Get the id_to_move mapping
    id_to_move = create_id_to_move_mapping()
    
    # Convert move IDs to move notations, skipping the BOS token (ID 0) if present
    return [id_to_move[move_id] for move_id in game if move_id != 0]
# end convert_ids_to_notation


def save_games(games: List[List[str]], output_file: str, chunk_size: int = None) -> None:
    """
    Save games to one or more binary files.
    
    Each game is saved as a sequence of move IDs, starting with a BOS token (ID 0).
    Token indexes are saved as 8-bit integers to save memory.
    
    Args:
        games (List[List[str]]): List of games to save
        output_file (str): Path to the output file
        chunk_size (int, optional): Number of games per file. If None, all games are saved to a single file.
    """
    import array
    import os
    
    # Create the move mapping
    move_to_id = create_move_mapping()
    
    # Convert games to sequences of IDs using byte arrays to save memory
    game_sequences = []
    for game in games:
        # Create a byte array for this game
        sequence = array.array('B')  # 'B' is unsigned char (8-bit)
        
        # Start with BOS token
        sequence.append(move_to_id["BOS"])
        
        # Add move IDs
        for move in game:
            if move != "pass":
                sequence.append(move_to_id[move])
            # end if
        # end for
        
        game_sequences.append(sequence)
    # end for
    
    # If chunk_size is None or 0, save all games to a single file
    if chunk_size is None or chunk_size <= 0:
        # Save to binary file using pickle
        with open(output_file, 'wb') as f:
            pickle.dump(game_sequences, f)
        # end with
    else:
        # Split games into chunks and save each chunk to a separate file
        num_chunks = (len(game_sequences) + chunk_size - 1) // chunk_size  # Ceiling division
        
        # Get the base filename and extension
        base_name, ext = os.path.splitext(output_file)
        
        # Save each chunk to a separate file
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(game_sequences))
            chunk = game_sequences[start_idx:end_idx]
            
            # Create the chunk filename
            chunk_filename = f"{base_name}_{i+1}{ext}"
            
            # Save the chunk
            with open(chunk_filename, 'wb') as f:
                pickle.dump(chunk, f)
            # end with
        # end for
    # end if
# end save_games


def load_games(input_file: str) -> List[List[int]]:
    """
    Load games from a binary file.
    
    Handles all formats:
    - Original format (lists of integers)
    - Numpy uint8 arrays
    - Python array.array('B') objects
    
    Args:
        input_file (str): Path to the input file
        
    Returns:
        List[List[int]]: List of games, where each game is a list of move IDs
    """
    with open(input_file, 'rb') as f:
        game_sequences = pickle.load(f)
    # end with
    
    # Convert arrays back to lists for compatibility with existing code
    result = []
    for game in game_sequences:
        if isinstance(game, np.ndarray):
            # Convert numpy array to regular Python list
            result.append(game.tolist())
        elif hasattr(game, 'typecode') and game.typecode == 'B':
            # Convert array.array to regular Python list
            result.append(list(game))
        else:
            # Already a list, no conversion needed
            result.append(game)
        # end if
    # end for
    
    return result
# end def load_games

def extract_game_by_index(games: List[List[int]], index: int) -> List[int]:
    """
    Extract a game by its index.
    
    Args:
        games (List[List[int]]): List of games
        index (int): Index of the game to extract
        
    Returns:
        List[int]: The extracted game as a list of move IDs
    """
    if index < 0 or index >= len(games):
        raise ValueError(f"Index {index} out of range. There are {len(games)} games.")
    return games[index]

def extract_games_by_length(games: List[List[int]], length: int) -> List[Tuple[int, List[int]]]:
    """
    Extract games with a specific length.
    
    Args:
        games (List[List[int]]): List of games
        length (int): Length of games to extract (including BOS token)
        
    Returns:
        List[Tuple[int, List[int]]]: List of tuples (index, game) with the specified length
    """
    return [(i, game) for i, game in enumerate(games) if len(game) - 1 == length]
# end def extract_games_by_length


def view_game(game_file: str, game_index: int) -> None:
    """
    Display an interactive visualization of an Othello game using matplotlib.
    
    Args:
        game_file (str): Path to the binary file containing games
        game_index (int): Index of the game to view
    """
    # Create a Rich console
    console = Console()
    
    # Load games from the input file
    console.print(f"Loading games from {game_file}...", style="blue")
    games = load_games(game_file)
    console.print(f"Loaded {len(games)} games.")
    
    # Extract the game at the specified index
    try:
        game_moves = extract_game_by_index(games, game_index)
        console.print(f"Viewing game at index {game_index} with {len(game_moves) - 1} moves.", style="green")
    except ValueError as e:
        console.print(f"Error: {e}", style="bold red")
        return
    # end try

    # Convert move IDs to notations (skipping BOS token)
    move_notations = convert_ids_to_notation(game_moves)

    # Log the game moves
    console.print(f"Game {game_index} has moves {move_notations}.", style="cyan")
    
    # Create a board to replay the game
    board = OthelloGame.load_moves(move_notations)
    
    # Current move index (start at -1 to show initial board)
    current_move = -1
    
    # Create the figure and axes for the board
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title(f"Othello Game Viewer - Game {game_index}")
    
    # Create button axes
    prev_button_ax = plt.axes([0.2, 0.05, 0.15, 0.05])
    next_button_ax = plt.axes([0.65, 0.05, 0.15, 0.05])
    
    # Create buttons
    prev_button = Button(prev_button_ax, 'Previous')
    next_button = Button(next_button_ax, 'Next')
    
    def draw_board():
        """
        Draw the current state of the board.
        """
        ax.clear()
        
        # Draw the green background
        ax.add_patch(plt.Rectangle((0, 0), 8, 8, color='green'))
        
        # Draw the grid lines
        for i in range(9):
            ax.plot([i, i], [0, 8], 'k-', lw=1)
            ax.plot([0, 8], [i, i], 'k-', lw=1)
        # end for

        # Draw the four initial pieces
        ax.add_patch(plt.Circle((3.5, 3.5), 0.4, color='white'))
        ax.add_patch(plt.Circle((4.5, 4.5), 0.4, color='white'))
        ax.add_patch(plt.Circle((3.5, 4.5), 0.4, color='black'))
        ax.add_patch(plt.Circle((4.5, 3.5), 0.4, color='black'))
        
        # Add column and row labels
        ax.set_xticks([i + 0.5 for i in range(8)])
        ax.set_yticks([i + 0.5 for i in range(8)])
        ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])

        # Set title with current move information
        if current_move == -1:
            ax.set_title("Initial Board")
        else:
            # Draw each move until current_move
            for m_i in range(current_move+1):
                # Get board modifications
                m: List[Tuple[int, int, int]] = board.board.history[m_i]

                # For each modification
                for row, col, p in m:
                    # Draw the piece
                    if p == board.BLACK:
                        ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='black'))
                    else:
                        ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='white'))
                    # end if
                # end for
            # end for

            # Get current move
            move_text = board.moves[current_move]

            # Use the stored player information
            player = "Black" if board.moves_player[current_move] == board.BLACK else "White"

            # Check if there was a pass before this move
            if current_move > 0 and player == board.moves_player[current_move-1]:
                # If the same player made two consecutive moves, it means the other player passed
                opposite_player = "White" if player == "Black" else "Black"
                ax.set_title(f"Move {current_move + 1}: {opposite_player} passed, {player} plays {move_text}")
            else:
                ax.set_title(f"Move {current_move + 1}: {player} plays {move_text}")
            # end if
        # end if

        # Set limits and aspect
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        
        # Update the figure
        fig.canvas.draw_idle()
    # end def draw_board
    
    def on_prev_click(event):
        """
        Handle click on Previous button.
        """
        nonlocal current_move, board
        
        if current_move > -1:
            # Reset the board and replay up to the previous move
            current_move -= 1

            # Update disply
            draw_board()
        # end if
    # end on_prev_click
    
    def on_next_click(event):
        """
        Handle click on Next button.
        """
        nonlocal current_move, board

        # Check that we are not at the end
        if current_move < len(board) - 1:
            # Update display
            current_move += 1
            draw_board()
        # end if
    # end def on_next_click
    
    # Connect button events
    prev_button.on_clicked(on_prev_click)
    next_button.on_clicked(on_next_click)
    
    # Initial draw
    draw_board()
    
    # Show the plot
    plt.tight_layout()
    plt.show()
# end def view_game

def main():
    """
    Main function to parse arguments and generate or extract games.
    
    Command-line arguments for generating games:
        --num-games: Number of games to generate (default: 10)
        --max-moves: Maximum number of moves per game (default: 60)
        --output: Output file path (default: othello_games.bin)
        --seed: Random seed for reproducibility
        --chunk-size: Number of games per file (default: None, all games in one file)
        
    Command-line arguments for extracting games:
        --extract: Extract games from a binary file
        --input: Input file path
        --index: Index of the game to extract
        --length: Length of games to extract
        --output: Output file path for extracted games
    """
    global console
    import os

    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Generate or extract valid Othello games.')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Parser for generate command
    generate_parser = subparsers.add_parser('generate', help='Generate Othello games')
    generate_parser.add_argument(
        '--num-games',
        type=int,
        required=True,
        help='Number of games to generate'
    )

    generate_parser.add_argument(
        '--output',
        type=str,
        default='othello_games.bin',
        help='Output file path (default: othello_games.bin)'
    )
    
    generate_parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    generate_parser.add_argument(
        '--num-val-games',
        type=int,
        default=None,
        help='Number of validation games to generate'
    )
    
    generate_parser.add_argument(
        '--val-output',
        type=str,
        default=None,
        help='Output file path for validation games'
    )
    
    generate_parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,
        help='Number of games per file. If not specified, all games are saved to a single file'
    )
    
    # Parser for extract command
    extract_parser = subparsers.add_parser('extract', help='Extract games from a binary file')
    extract_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input file path'
    )
    
    # Output option removed as per requirement - no file output for extract
    
    # Make index and length mutually exclusive
    group = extract_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--index',
        type=int,
        help='Index of the game to extract'
    )
    
    group.add_argument(
        '--length',
        type=int,
        help='Length of games to extract (including BOS token)'
    )
    
    # Parser for view command
    view_parser = subparsers.add_parser('view', help='View an Othello game with a graphical interface')
    view_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input file path containing games'
    )
    view_parser.add_argument(
        '--index',
        type=int,
        required=True,
        help='Index of the game to view'
    )
    
    # For backward compatibility, if no command is specified, assume generate
    if len(sys.argv) > 1 and sys.argv[1] not in ['generate', 'extract', 'view']:
        # Add the generate command
        sys.argv.insert(1, 'generate')
    # end if
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == 'generate':
        # Generate the specified number of games
        console.print(f"Generating {args.num_games} Othello games...", style="bold green")

        if args.seed is not None:
            console.print(f"Using random seed: {args.seed}", style="blue")
        # end if

        # Generate the games and save them directly to the output file
        games = generate_games(args.num_games, args.seed, args.output, args.chunk_size)

        # If games were saved directly to files, we don't have statistics to display
        if not games:
            console.print("\nGames were saved directly to files to save memory.", style="bold green")
        else:
            # Calculate and display statistics of game lengths
            game_lengths = [len(game) for game in games]
            length_counter = Counter(game_lengths)
            
            console.print("\nGame Length Statistics:", style="bold")
            console.print(f"Total games: {len(games)}")
            console.print(f"Average length: {sum(game_lengths) / len(games):.2f} moves")
            console.print(f"Minimum length: {min(game_lengths)} moves")
            console.print(f"Maximum length: {max(game_lengths)} moves")
            
            console.print("\nLength distribution:", style="bold")
            for length in sorted(length_counter.keys()):
                count = length_counter[length]
                percentage = (count / len(games)) * 100
                console.print(f"{length} moves: {count} games ({percentage:.1f}%)")
            # end for
            
            # Save the generated games to the output file if they weren't saved already
            save_games(games, args.output, args.chunk_size)
        
        # Print appropriate message based on whether chunking was used
        if args.chunk_size is not None and args.chunk_size > 0:
            base_name, ext = os.path.splitext(args.output)
            console.print(f"\nGames saved to {base_name}_N{ext} files with {args.chunk_size} games per file", style="bold green")
        else:
            console.print(f"\nGames saved to {args.output}", style="bold green")
        
        # Check if both validation parameters are provided
        if args.num_val_games is not None and args.val_output is not None:
            # Generate validation games
            console.print(f"\nGenerating {args.num_val_games} validation games...", style="bold green")
            
            # Use the same seed for reproducibility if provided
            if args.seed is not None:
                console.print(f"Using random seed: {args.seed}", style="blue")
            
            # Generate the validation games and save them directly to the output file
            val_games = generate_games(args.num_val_games, args.seed, args.val_output, args.chunk_size)
            
            # If validation games were saved directly to files, we don't have statistics to display
            if not val_games:
                console.print("\nValidation games were saved directly to files to save memory.", style="bold green")
            else:
                # Calculate and display statistics of validation game lengths
                val_game_lengths = [len(game) for game in val_games]
                val_length_counter = Counter(val_game_lengths)
                
                console.print("\nValidation Game Length Statistics:", style="bold")
                console.print(f"Total validation games: {len(val_games)}")
                console.print(f"Average length: {sum(val_game_lengths) / len(val_games):.2f} moves")
                console.print(f"Minimum length: {min(val_game_lengths)} moves")
                console.print(f"Maximum length: {max(val_game_lengths)} moves")
                
                console.print("\nValidation Length distribution:", style="bold")
                for length in sorted(val_length_counter.keys()):
                    count = val_length_counter[length]
                    percentage = (count / len(val_games)) * 100
                    console.print(f"{length} moves: {count} games ({percentage:.1f}%)")
                
                # Save the validation games to the specified output file if they weren't saved already
                save_games(val_games, args.val_output, args.chunk_size)
            
            # Print appropriate message based on whether chunking was used
            if args.chunk_size is not None and args.chunk_size > 0:
                base_name, ext = os.path.splitext(args.val_output)
                console.print(f"\nValidation games saved to {base_name}_N{ext} files with {args.chunk_size} games per file", style="bold green")
            else:
                console.print(f"\nValidation games saved to {args.val_output}", style="bold green")
    elif args.command == 'extract':
        # Create a Rich console
        console = Console()
        
        # Load games from the input file
        console.print(f"Loading games from {args.input}...")
        games = load_games(args.input)
        console.print(f"Loaded {len(games)} games.")
        
        # Extract games based on the specified criteria
        if args.index is not None:
            # Extract a single game by index
            try:
                extracted_game = extract_game_by_index(games, args.index)
                # Create a list with a single tuple (index, game)
                extracted_games_with_indices = [(args.index, extracted_game)]
                console.print(f"Extracted game at index {args.index} with length {len(extracted_game)}.")
            except ValueError as e:
                console.print(f"Error: {e}", style="bold red")
                return
            # end try
        else:
            # Extract games by length
            extracted_games_with_indices = extract_games_by_length(games, args.length)
            console.print(f"Extracted {len(extracted_games_with_indices)} games with length {args.length}.")
        # end if
        
        # Determine the number of digits needed for zero padding
        max_index = max([idx for idx, _ in extracted_games_with_indices]) if extracted_games_with_indices else 0

        # Set a minimum padding of 3 digits
        padding = max(3, len(str(max_index)))
        
        # Display the extracted games in the terminal with A-H, 1-8 notation
        console.print("\nExtracted games in A-H, 1-8 notation:", style="bold")
        for game_idx, game in extracted_games_with_indices:
            # Convert move IDs to move notations
            game_notation = convert_ids_to_notation(game)
            
            # Create a formatted text with the index in a different color
            text = Text()
            text.append(f"Game {game_idx:0{padding}d}: ", style="bold cyan")
            text.append(' '.join(game_notation))
            
            # Display the game
            console.print(text)
        # end for
        
        # No longer saving extracted games to a file - only displaying in terminal
    elif args.command == 'view':
        # Launch the interactive game viewer
        view_game(args.input, args.index)
# end main


if __name__ == '__main__':
    main()
# end if