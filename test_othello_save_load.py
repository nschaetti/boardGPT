#!/usr/bin/env python3
"""
Test script to verify the memory-efficient save_games and load_games functions in othello.py
"""

import os
import sys
import numpy as np
from simulators.othello import save_games, load_games, create_move_mapping

def generate_sample_games(num_games=100, max_moves=30):
    """Generate sample game data for testing"""
    move_to_id = create_move_mapping()
    # Get all possible moves (excluding BOS)
    possible_moves = [move for move in move_to_id.keys() if move != "BOS"]
    
    # Generate random games
    games = []
    for _ in range(num_games):
        # Random number of moves for this game
        num_moves = np.random.randint(10, max_moves)
        # Generate random moves
        game = np.random.choice(possible_moves, size=num_moves).tolist()
        games.append(game)
    
    return games

def test_save_load():
    """Test saving and loading games with the memory-efficient format"""
    print("Generating sample games...")
    sample_games = generate_sample_games()
    
    # Save games
    output_file = "test_games.pkl"
    print(f"Saving {len(sample_games)} games to {output_file}...")
    save_games(sample_games, output_file)
    
    # Load games
    print(f"Loading games from {output_file}...")
    loaded_games = load_games(output_file)
    
    # Verify
    print(f"Loaded {len(loaded_games)} games.")
    print(f"Checking if games were saved and loaded correctly...")
    
    # Convert original games to IDs for proper comparison
    move_to_id = create_move_mapping()
    original_games_ids = []
    for game in sample_games:
        game_ids = []
        for move in game:
            if move != "pass" and move in move_to_id:
                game_ids.append(move_to_id[move])
        original_games_ids.append(game_ids)
    
    # Check if the loaded games match the original games (after conversion to IDs)
    all_match = True
    for i, (original_ids, loaded) in enumerate(zip(original_games_ids, loaded_games)):
        # Skip BOS token in loaded game (index 0)
        if original_ids != loaded[1:]:
            print(f"Game {i} doesn't match!")
            print(f"Original IDs: {original_ids}")
            print(f"Loaded IDs: {loaded[1:]}")
            all_match = False
            break
    
    if all_match:
        print("All games loaded correctly!")
    
    # Clean up
    os.remove(output_file)
    print(f"Removed test file: {output_file}")

if __name__ == "__main__":
    test_save_load()