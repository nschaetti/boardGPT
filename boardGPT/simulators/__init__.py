"""
Simulators for board games.
"""

from .othello import (
    create_id_to_move_mapping,
    convert_ids_to_notation
)

__all__ = [
    "create_id_to_move_mapping",
    "convert_ids_to_notation"
]
