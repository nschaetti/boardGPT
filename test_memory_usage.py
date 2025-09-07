#!/usr/bin/env python3
"""
Test script to verify memory savings from using uint8 arrays in save_games
"""

import os
import sys
import numpy as np
import pickle
from simulators.othello import create_move_mapping

def generate_sample_games(num_games=1000, max_moves=30):
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

def save_games_original(games, output_file):
    """Original implementation of save_games (using regular Python lists)"""
    move_to_id = create_move_mapping()
    
    # Convert games to sequences of IDs
    game_sequences = []
    for game in games:
        # Start with BOS token
        sequence = [move_to_id["BOS"]]
        # Add move IDs
        for move in game:
            if move != "pass":
                sequence.append(move_to_id[move])
        game_sequences.append(sequence)
    
    # Save to binary file using pickle
    with open(output_file, 'wb') as f:
        pickle.dump(game_sequences, f)

def save_games_optimized(games, output_file):
    """Optimized implementation of save_games (using array.array('B') for 8-bit integers)"""
    import array
    move_to_id = create_move_mapping()
    
    # Convert games to sequences of IDs using byte arrays
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
        
        game_sequences.append(sequence)
    
    # Save to binary file using pickle
    with open(output_file, 'wb') as f:
        pickle.dump(game_sequences, f)

def test_memory_usage():
    """Compare memory usage between original and optimized implementations"""
    print("Generating sample games...")
    sample_games = generate_sample_games(num_games=1000)
    
    # Save using original implementation
    original_file = "original_games.pkl"
    print(f"Saving games using original implementation to {original_file}...")
    save_games_original(sample_games, original_file)
    original_size = os.path.getsize(original_file)
    print(f"Original file size: {original_size} bytes")
    
    # Save using optimized implementation
    optimized_file = "optimized_games.pkl"
    print(f"Saving games using optimized implementation to {optimized_file}...")
    save_games_optimized(sample_games, optimized_file)
    optimized_size = os.path.getsize(optimized_file)
    print(f"Optimized file size: {optimized_size} bytes")
    
    # Calculate memory savings
    savings = original_size - optimized_size
    savings_percent = (savings / original_size) * 100
    print(f"Memory savings: {savings} bytes ({savings_percent:.2f}%)")
    
    # Clean up
    os.remove(original_file)
    os.remove(optimized_file)
    print(f"Removed test files: {original_file}, {optimized_file}")

if __name__ == "__main__":
    test_memory_usage()