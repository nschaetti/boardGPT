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

#!/usr/bin/env python3
"""
Command Line Interface for boardGPT games.
"""

import argparse
import sys
import importlib
from typing import List, Optional

from .othello import (
    othello_generate,
    othello_play,
    othello_simulator,
    othello_visualize,
    othello_utils,
    othello_convert,
    othello_verify,
    othello_load
)


    # end except
def checkers_info(args):
    """
    Display information about the Checkers implementation status.
    """
    print("Checkers implementation is not yet available.")
    print("This is a placeholder for future implementation.")
# def checkers_info

# end def checkers_info
def main():
    """
    Main entry point for the CLI.
    """
    parser = argparse.ArgumentParser(description="Command Line Interface for boardGPT games.")
    subparsers = parser.add_subparsers(dest="game", help="Game to interact with")
    
    # Othello commands
    othello_parser = subparsers.add_parser("othello", help="Commands for Othello game")
    othello_subparsers = othello_parser.add_subparsers(dest="command", help="Command to execute")

    # Othello generate
    othello_generate_parser = othello_subparsers.add_parser("generate", help="Generate Othello game sequences")
    othello_generate_parser.add_argument("--num-games", required=True, help="Othello game to generate")
    othello_generate_parser.add_argument("--seed", default=42, help="Othello game seed")
    othello_generate_parser.add_argument("--output", required=True, help="Path to the output .bin file")
    othello_generate_parser.add_argument("--chunk-size", default=None, help="Size of chunks (how many samples)")
    othello_generate_parser.set_defaults(func=othello_generate)
    
    # Othello play
    othello_play_parser = othello_subparsers.add_parser("play", help="Start an interactive Othello game")
    othello_play_parser.set_defaults(func=othello_play)
    
    # Othello visualize
    othello_visualize_parser = othello_subparsers.add_parser("visualize", help="Visualize an Othello game")
    othello_visualize_parser.add_argument("--moves", help="Comma-separated list of moves (e.g., 'd3,c4,e3')")
    othello_visualize_parser.add_argument("--file", help="File containing moves (one per line)")
    othello_visualize_parser.set_defaults(func=othello_visualize)
    
    # Othello verify
    othello_verify_parser = othello_subparsers.add_parser("verify", help="Verify if a sequence of moves forms a valid Othello game")
    othello_verify_parser.add_argument("--moves", help="Comma-separated list of moves (e.g., 'd3,c4,e3')")
    othello_verify_parser.add_argument("--file", help="File containing moves (one per line)")
    othello_verify_parser.set_defaults(func=othello_verify)
    
    # Othello convert
    othello_convert_parser = othello_subparsers.add_parser("convert", help="Convert between different representations of Othello games")
    othello_convert_parser.add_argument("--moves", help="Comma-separated list of moves (e.g., 'd3,c4,e3')")
    othello_convert_parser.add_argument("--file", help="File containing moves (one per line)")
    othello_convert_parser.add_argument("--to-board", action="store_true", help="Convert moves to board representation")
    othello_convert_parser.set_defaults(func=othello_convert)
    
    # Othello load
    othello_load_parser = othello_subparsers.add_parser("load", help="Load Othello games from a file")
    othello_load_parser.add_argument("--file", required=True, help="File containing games")
    othello_load_parser.add_argument("--index", type=int, help="Index of the game to display")
    othello_load_parser.set_defaults(func=othello_load)
    
    # Checkers commands
    checkers_parser = subparsers.add_parser("checkers", help="Commands for Checkers game")
    checkers_subparsers = checkers_parser.add_subparsers(dest="command", help="Command to execute")
    
    # Checkers info
    checkers_info_parser = checkers_subparsers.add_parser("info", help="Display information about the Checkers implementation status")
    checkers_info_parser.set_defaults(func=checkers_info)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate function
    if hasattr(args, "func"):
        args.func(args)  # end if
    else:
        parser.print_help()
     # end if
# end def main

    # end else
if __name__ == "__main__":
    main()
# end if

