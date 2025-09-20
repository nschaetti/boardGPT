#!/usr/bin/env python3
"""
Test script for Othello game validation.

This script tests the implementation of the validate_game method and the modified
generate_game function to ensure they work correctly.
"""

import sys
from simulators.othello import OthelloGame, generate_othello_game

def test_game_validation():
    """
    Test the game validation functionality by generating and validating games.
    """
    print("Testing Othello game validation...")
    
    # Test 1: Generate and validate a few games
    num_games = 5
    print(f"\nTest 1: Generating and validating {num_games} games")
    
    for i in range(num_games):
        print(f"\nGame {i+1}:")
        try:
            # Generate a game
            moves = generate_othello_game(seed=i)
            
            # Load the game into a board
            board = OthelloGame.load_moves(moves)
            
            # Validate the game
            is_valid, message = board.validate_game()
            
            # Print results
            print(f"  Moves: {len(moves)}")
            print(f"  Valid: {is_valid}")
            print(f"  Message: {message}")
            
            # Additional check - validate the game again
            if is_valid:
                # The game should be valid, but let's double-check
                is_valid2, message2 = board.validate_game()
                if not is_valid2:
                    print(f"  ERROR: Game validated initially but failed on second validation!")
                    print(f"  Second validation message: {message2}")
            
        except Exception as e:
            print(f"  Error generating game: {e}")
    
    # Test 2: Test validation with a generated valid game
    print("\nTest 2: Testing validation with a generated valid game")
    
    try:
        # Generate a valid game using a fixed seed for reproducibility
        valid_game = generate_othello_game(seed=42)
        print(f"  Generated game: {valid_game}")
        
        # Load the game
        board = OthelloGame.load_moves(valid_game)
        
        # Validate the game
        is_valid, message = board.validate_game()
        
        # Print results
        print(f"  Moves: {len(valid_game)}")
        print(f"  Valid: {is_valid}")
        print(f"  Message: {message}")
    except Exception as e:
        print(f"  Error validating generated game: {e}")
    
    # Test 3: Test validation with an invalid game
    print("\nTest 3: Testing validation with an invalid game")
    # Create an invalid game by repeating a move
    invalid_game = ['d3', 'd3', 'c4', 'e3']  # d3 is repeated
    
    try:
        # Load the game
        board = OthelloGame()
        board.set_moves(invalid_game, [1, 2, 1, 2])  # Set player information
        
        # Validate the game
        is_valid, message = board.validate_game()
        
        # Print results
        print(f"  Moves: {len(invalid_game)}")
        print(f"  Valid: {is_valid}")
        print(f"  Message: {message}")
        
        # The game should be invalid
        if is_valid:
            print("  ERROR: Invalid game was marked as valid!")
    except Exception as e:
        print(f"  Error validating invalid game: {e}")

if __name__ == "__main__":
    test_game_validation()