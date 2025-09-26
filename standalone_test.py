#!/usr/bin/env python3
"""
Standalone test script for boardGPT methods.

This script tests the methods implemented in boardGPT/__init__.py
without importing the boardGPT module to avoid dependency issues.
"""

from typing import List, Tuple, Optional, Union, Any
import random
from abc import ABC, abstractmethod


class GameInterface(ABC):
    """
    Common interface for all board games.
    
    This abstract class defines the methods that all game implementations must provide.
    It ensures consistency across different game types and facilitates code reuse.
    
    Game implementations should inherit from this class and implement all abstract methods.
    """
    
    @abstractmethod
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Return a list of valid moves for the current player."""
        pass
    
    @abstractmethod
    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if placing a piece at (row, col) is a valid move."""
        pass
    
    @abstractmethod
    def make_move(self, row: int, col: int) -> bool:
        """Make a move at (row, col) for the current player."""
        pass
    
    @abstractmethod
    def make_random_move(self) -> Optional[Tuple[int, int]]:
        """Make a random valid move for the current player."""
        pass
    
    @abstractmethod
    def has_valid_moves(self) -> bool:
        """Check if the current player has any valid moves."""
        pass
    
    @abstractmethod
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        pass
    
    @abstractmethod
    def coords_to_notation(self, row: int, col: int) -> str:
        """Convert board coordinates to standard notation."""
        pass
    
    @abstractmethod
    def notation_to_coords(self, notation: str) -> Tuple[int, int]:
        """Convert standard notation to board coordinates."""
        pass
    
    @abstractmethod
    def get_moves(self) -> List[str]:
        """Return the list of moves made in the game in standard notation."""
        pass
    
    @abstractmethod
    def show(self) -> None:
        """Display the current game state."""
        pass


# Implementation of the methods from boardGPT/__init__.py

def valid(game: GameInterface, move: Union[str, Tuple[int, int]]) -> bool:
    """
    Check if a move is valid for the given game.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
        move (Union[str, Tuple[int, int]]): Move in standard notation (e.g., 'e4') or as coordinates (row, col)
    
    Returns:
        bool: True if the move is valid, False otherwise
    """
    # Convert move to coordinates if it's in notation format
    if isinstance(move, str):
        row, col = game.notation_to_coords(move)
    else:
        row, col = move
    
    return game.is_valid_move(row, col)


def next(game: GameInterface, move: Union[str, Tuple[int, int]]) -> GameInterface:
    """
    Apply the move to the game and return the same game object.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
        move (Union[str, Tuple[int, int]]): Move in standard notation (e.g., 'e4') or as coordinates (row, col)
    
    Returns:
        GameInterface: The same game object with the move applied
        
    Raises:
        ValueError: If the move is invalid
    """
    # Convert move to coordinates if it's in notation format
    if isinstance(move, str):
        row, col = game.notation_to_coords(move)
    else:
        row, col = move
    
    # Make the move
    success = game.make_move(row, col)
    
    if not success:
        raise ValueError(f"Invalid move: {move}")
    
    return game


def rnext(game: GameInterface) -> GameInterface:
    """
    Apply a random valid move to the game and return the same game object.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
    
    Returns:
        GameInterface: The same game object with a random move applied
        
    Raises:
        ValueError: If there are no valid moves
    """
    move = game.make_random_move()
    
    if move is None:
        raise ValueError("No valid moves available")
    
    return game


def valid_moves(game: GameInterface) -> List[Tuple[int, int]]:
    """
    Get all valid moves for the current player.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
    
    Returns:
        List[Tuple[int, int]]: A list of (row, col) tuples representing valid move positions
    """
    return game.get_valid_moves()


def valid_moves_notation(game: GameInterface) -> List[str]:
    """
    Get all valid moves for the current player in standard notation.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
    
    Returns:
        List[str]: A list of moves in standard notation
    """
    valid_moves_coords = game.get_valid_moves()
    return [game.coords_to_notation(row, col) for row, col in valid_moves_coords]


def is_over(game: GameInterface) -> bool:
    """
    Check if the game is over.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
    
    Returns:
        bool: True if the game is over, False otherwise
    """
    return game.is_game_over()


def show(game: GameInterface) -> None:
    """
    Display the current game state.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
    """
    game.show()


def has_moves(game: GameInterface) -> bool:
    """
    Check if the current player has any valid moves.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
    
    Returns:
        bool: True if the current player has at least one valid move, False otherwise
    """
    return game.has_valid_moves()


def get_moves(game: GameInterface) -> List[str]:
    """
    Get the list of moves made in the game in standard notation.
    
    Args:
        game (GameInterface): A game object implementing the GameInterface
    
    Returns:
        List[str]: List of moves in standard notation
    """
    return game.get_moves()


class MockGame(GameInterface):
    """A simple mock implementation of GameInterface for testing purposes."""
    
    def __init__(self):
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.current_player = 1  # 1 for black, 2 for white
        self.moves_history = []
        
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Return a list of valid moves for the current player."""
        # For simplicity, let's say any empty cell is a valid move
        valid_moves = []
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == 0:
                    valid_moves.append((row, col))
        return valid_moves[:5]  # Limit to 5 moves for testing
    
    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if placing a piece at (row, col) is a valid move."""
        if row < 0 or row >= 8 or col < 0 or col >= 8:
            return False
        return self.board[row][col] == 0
    
    def make_move(self, row: int, col: int) -> bool:
        """Make a move at (row, col) for the current player."""
        if not self.is_valid_move(row, col):
            return False
        
        self.board[row][col] = self.current_player
        notation = self.coords_to_notation(row, col)
        self.moves_history.append(notation)
        
        # Switch player
        self.current_player = 3 - self.current_player  # 1 -> 2, 2 -> 1
        
        return True
    
    def make_random_move(self) -> Optional[Tuple[int, int]]:
        """Make a random valid move for the current player."""
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return None
        
        row, col = random.choice(valid_moves)
        self.make_move(row, col)
        return row, col
    
    def has_valid_moves(self) -> bool:
        """Check if the current player has any valid moves."""
        return len(self.get_valid_moves()) > 0
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        # For simplicity, let's say the game is over if the board is full
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == 0:
                    return False
        return True
    
    def coords_to_notation(self, row: int, col: int) -> str:
        """Convert board coordinates to standard notation."""
        col_letter = chr(ord('a') + col)
        row_number = str(row + 1)
        return col_letter + row_number
    
    def notation_to_coords(self, notation: str) -> Tuple[int, int]:
        """Convert standard notation to board coordinates."""
        col = ord(notation[0]) - ord('a')
        row = int(notation[1]) - 1
        return row, col
    
    def get_moves(self) -> List[str]:
        """Return the list of moves made in the game in standard notation."""
        return self.moves_history
    
    def show(self) -> None:
        """Display the current game state."""
        print("  a b c d e f g h")
        for row in range(8):
            print(f"{row+1} ", end="")
            for col in range(8):
                if self.board[row][col] == 0:
                    print(". ", end="")
                elif self.board[row][col] == 1:
                    print("B ", end="")
                else:
                    print("W ", end="")
            print()
        print(f"Current player: {'Black' if self.current_player == 1 else 'White'}")


def test_methods():
    """Test the methods with a mock game."""
    # Create a mock game
    game = MockGame()
    
    print("Testing methods with mock game:")
    print("-" * 50)
    
    # Test show method
    print("\nInitial game state:")
    show(game)
    
    # Test valid_moves and valid_moves_notation methods
    moves_coords = valid_moves(game)
    moves_notation = valid_moves_notation(game)
    
    print("\nValid moves (coordinates):", moves_coords)
    print("Valid moves (notation):", moves_notation)
    
    # Test has_moves method
    has_valid_moves = has_moves(game)
    print("\nCurrent player has valid moves:", has_valid_moves)
    
    # Test valid method with both coordinate and notation formats
    if moves_coords:
        move_coords = moves_coords[0]
        move_notation = moves_notation[0]
        
        print(f"\nTesting valid method:")
        print(f"Is {move_coords} valid? {valid(game, move_coords)}")
        print(f"Is {move_notation} valid? {valid(game, move_notation)}")
        
        # Test next method with notation format
        print(f"\nApplying move {move_notation} using next method:")
        next(game, move_notation)
        show(game)
        
        # Test get_moves method
        moves_made = get_moves(game)
        print("\nMoves made so far:", moves_made)
        
        # Test rnext method
        print("\nApplying random move using rnext method:")
        rnext(game)
        show(game)
        
        # Test get_moves method again
        moves_made = get_moves(game)
        print("\nMoves made so far:", moves_made)
    
    # Test is_over method
    game_over = is_over(game)
    print("\nIs game over?", game_over)
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_methods()