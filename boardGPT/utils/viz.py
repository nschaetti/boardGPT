

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Tuple, Optional, Union
import sys
import os
import seaborn as sns


from boardGPT.simulators.othello import OthelloGame
from boardGPT.simulators import create_id_to_move_mapping
import random

# Function to detect if code is running in Jupyter notebook
def is_jupyter() -> bool:
    """
    Check if the code is running in a Jupyter notebook.
    
    Returns:
        bool: True if running in Jupyter notebook, False otherwise
    """
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return False
        if 'IPKernelApp' not in get_ipython().config:
            return False
        return True
    except ImportError:
        return False


def show_othello(moves: Optional[List[str]] = None) -> Union[plt.Figure, None]:
    """
    Display an interactive visualization of an Othello game using matplotlib.
    
    Args:
        moves (Optional[List[str]]): List of moves in standard notation (e.g., ['c4', 'd3', ...])
                                    If None, starts with an empty board
    
    Returns:
        Union[plt.Figure, None]: In Jupyter notebooks, returns the figure object for inline display.
                                In regular Python scripts, returns None after displaying the figure.
    """
    # Create a board to replay the game
    if moves:
        board = OthelloGame.load_moves(moves)
    else:
        board = OthelloGame()
    
    # Current move index (start at -1 to show initial board)
    current_move = -1
    
    # Track attempted illegal moves for highlighting
    illegal_move = None
    
    # Create the figure and axes for the board
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("Othello Game Viewer")
    
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
        
        # If there's an illegal move, highlight it with red background
        if illegal_move:
            row, col = illegal_move
            ax.add_patch(plt.Rectangle((col, row), 1, 1, color='red', alpha=0.5))
        
        # Draw the initial board state
        if current_move == -1:
            # Draw the four initial pieces
            ax.add_patch(plt.Circle((3.5, 3.5), 0.4, color='white'))
            ax.add_patch(plt.Circle((4.5, 4.5), 0.4, color='white'))
            ax.add_patch(plt.Circle((3.5, 4.5), 0.4, color='black'))
            ax.add_patch(plt.Circle((4.5, 3.5), 0.4, color='black'))
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
            
            # Get current move
            move_text = board.moves[current_move]

            # Use the stored player information
            player = "Black" if board.moves_player[current_move] == board.BLACK else "White"

            # Check if there was a pass before this move
            if current_move > 0 and board.moves_player[current_move] == board.moves_player[current_move-1]:
                # If the same player made two consecutive moves, it means the other player passed
                opposite_player = "White" if player == "Black" else "Black"
                ax.set_title(f"Move {current_move + 1}: {opposite_player} passed, {player} plays {move_text}")
            else:
                ax.set_title(f"Move {current_move + 1}: {player} plays {move_text}")
        
        # Add column and row labels
        ax.set_xticks([i + 0.5 for i in range(8)])
        ax.set_yticks([i + 0.5 for i in range(8)])
        ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])
        
        # Set limits and aspect
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        
        # Update the figure
        fig.canvas.draw_idle()
    
    def on_prev_click(event):
        """
        Handle click on Previous button.
        """
        nonlocal current_move, illegal_move
        
        # Clear any illegal move highlighting
        illegal_move = None
        
        if current_move > -1:
            # Reset the board and replay up to the previous move
            current_move -= 1
            draw_board()
    
    def on_next_click(event):
        """
        Handle click on Next button.
        """
        nonlocal current_move, illegal_move
        
        # Clear any illegal move highlighting
        illegal_move = None
        
        # Check that we are not at the end
        if current_move < len(board) - 1:
            current_move += 1
            draw_board()
    
    def on_board_click(event):
        """
        Handle click on the board to make a move.
        """
        nonlocal current_move, illegal_move
        
        # Only process clicks within the board area
        if event.xdata is None or event.ydata is None:
            return
        
        # Convert click coordinates to board indices
        col = int(event.xdata)
        row = int(event.ydata)
        
        # Check if the click is within the board boundaries
        if 0 <= row < 8 and 0 <= col < 8:
            # Create a temporary game state to check if the move is valid
            temp_game = OthelloGame()
            
            # Replay all moves up to the current point
            if current_move >= 0:
                for i in range(current_move + 1):
                    move_notation = board.moves[i]
                    move_row, move_col = temp_game.notation_to_coords(move_notation)
                    temp_game.make_move(move_row, move_col)
            
            # Check if the clicked position is a valid move
            if temp_game.is_valid_move(row, col):
                # Convert the move to notation
                move_notation = temp_game.coords_to_notation(row, col)
                
                # If we're not at the end of the existing moves list
                if current_move < len(board) - 1:
                    # Check if this move matches the next recorded move
                    next_move = board.moves[current_move + 1]
                    if move_notation == next_move:
                        # This is the correct next move, advance
                        current_move += 1
                        illegal_move = None
                        draw_board()
                    else:
                        # This is not the next recorded move, highlight as illegal
                        illegal_move = (row, col)
                        draw_board()
                else:
                    # We're at the end of the recorded moves, add a new move
                    temp_game.make_move(row, col)
                    board.moves.append(move_notation)
                    board.moves_player.append(temp_game.current_player)
                    board.board.history.append([(row, col, temp_game.current_player)])
                    current_move += 1
                    illegal_move = None
                    draw_board()
            else:
                # Highlight the illegal move
                illegal_move = (row, col)
                draw_board()
    
    # Connect button events
    prev_button.on_clicked(on_prev_click)
    next_button.on_clicked(on_next_click)
    
    # Connect board click event
    fig.canvas.mpl_connect('button_press_event', on_board_click)
    
    # Initial draw
    draw_board()
    
    # Apply tight layout to the figure
    plt.tight_layout()
    
    # Check if running in Jupyter notebook
    if is_jupyter():
        # In Jupyter, return the figure for inline display
        return fig
    else:
        # In regular Python scripts, show the figure and return None
        plt.show()


def plot_othello_game(moves: Optional[List[str]] = None) -> Union[FuncAnimation, None]:
    """
    Create an animation of an Othello game with 1 second per move.
    
    Args:
        moves (Optional[List[str]]): List of moves in standard notation (e.g., ['c4', 'd3', ...'])
                                    If None, starts with an empty board
    
    Returns:
        Union[FuncAnimation, None]: In Jupyter notebooks, returns the animation object for inline display.
                                   In regular Python scripts, returns None after displaying the animation.
    """
    # Create a board to replay the game
    if moves:
        # Create a new game
        game = OthelloGame()
        
        # Keep track of all moves (valid and invalid)
        all_moves = []
        valid_moves = []
        invalid_moves = []
        
        # Process each move
        for move in moves:
            try:
                # Convert to coordinates
                row, col = game.notation_to_coords(move)
                
                # Check if the move is valid
                if game.is_valid_move(row, col):
                    # Make the move
                    game.make_move(row, col)
                    valid_moves.append(move)
                    all_moves.append((move, True))  # (move, is_valid)
                else:
                    # Try for the other player
                    game.switch_player()
                    if game.is_valid_move(row, col):
                        # Make the move
                        game.make_move(row, col)
                        valid_moves.append(move)
                        all_moves.append((move, True))  # (move, is_valid)
                    else:
                        # Invalid move for both players
                        game.switch_player()  # Switch back
                        invalid_moves.append((move, row, col))
                        all_moves.append((move, False))  # (move, is_valid)
            except ValueError:
                # Skip invalid move notations
                pass
        
        # Use the board with only valid moves for the animation
        board = OthelloGame.load_moves(valid_moves)
    else:
        board = OthelloGame()
        all_moves = []
        invalid_moves = []
    
    # Current move index (start at -1 to show initial board)
    current_move = -1
    
    # Create the figure and axes for the board
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("Othello Game Animation")
    
    # Create button axes
    prev_button_ax = plt.axes([0.2, 0.05, 0.15, 0.05])
    next_button_ax = plt.axes([0.65, 0.05, 0.15, 0.05])
    
    # Create buttons
    prev_button = Button(prev_button_ax, 'Previous')
    next_button = Button(next_button_ax, 'Next')
    
    # Function to draw the board at a specific move index
    def draw_board(move_idx):
        """
        Draw the board state at the given move index.
        
        Args:
            move_idx (int): The move index to display (-1 for initial board)
        """
        ax.clear()
        
        # Draw the green background
        ax.add_patch(plt.Rectangle((0, 0), 8, 8, color='green'))
        
        # Draw the grid lines
        for i in range(9):
            ax.plot([i, i], [0, 8], 'k-', lw=1)
            ax.plot([0, 8], [i, i], 'k-', lw=1)
        
        # Check if we're showing an invalid move
        is_invalid_move = False
        invalid_move_coords = None
        
        if move_idx >= 0 and move_idx < len(all_moves):
            move_text, is_valid = all_moves[move_idx]
            if not is_valid:
                is_invalid_move = True
                # Find the coordinates for this invalid move
                for inv_move, row, col in invalid_moves:
                    if inv_move == move_text:
                        invalid_move_coords = (row, col)
                        break
        
        # Draw the initial board state
        if move_idx == -1:
            # Draw the four initial pieces
            ax.add_patch(plt.Circle((3.5, 3.5), 0.4, color='white'))
            ax.add_patch(plt.Circle((4.5, 4.5), 0.4, color='white'))
            ax.add_patch(plt.Circle((3.5, 4.5), 0.4, color='black'))
            ax.add_patch(plt.Circle((4.5, 3.5), 0.4, color='black'))
            ax.set_title("Initial Board")
        elif is_invalid_move:
            # For invalid moves, show the board state before the invalid move
            # and highlight the invalid move with a red background
            
            # Create a board state representation
            board_state = [[0 for _ in range(8)] for _ in range(8)]
            
            # Set initial pieces
            board_state[3][3] = board.WHITE  # d4 in standard notation
            board_state[3][4] = board.BLACK  # e4 in standard notation
            board_state[4][3] = board.BLACK  # d5 in standard notation
            board_state[4][4] = board.WHITE  # e5 in standard notation
            
            # Count valid moves up to this point
            valid_count = 0
            for i in range(move_idx):
                if all_moves[i][1]:  # If move is valid
                    valid_count += 1
            
            # Apply all valid moves up to this point
            for m_i in range(valid_count):
                # Get board modifications
                m: List[Tuple[int, int, int]] = board.board.history[m_i]
                
                # For each modification
                for row, col, p in m:
                    # Update the board state
                    board_state[row][col] = p
            
            # Draw all pieces based on the current board state
            for row in range(8):
                for col in range(8):
                    p = board_state[row][col]
                    if p == board.BLACK:
                        ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='black'))
                    elif p == board.WHITE:
                        ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='white'))
            
            # Highlight the invalid move with a red background
            if invalid_move_coords:
                row, col = invalid_move_coords
                ax.add_patch(plt.Rectangle((col, row), 1, 1, color='red', alpha=0.5))
            
            # Get current move
            move_text = all_moves[move_idx][0]
            
            # Determine whose turn it would be
            if valid_count > 0:
                last_player = board.moves_player[valid_count-1]
                current_player = board.BLACK if last_player == board.WHITE else board.WHITE
                player = "Black" if current_player == board.BLACK else "White"
            else:
                # First move is always black
                player = "Black"
            
            ax.set_title(f"Invalid Move: {player} tries {move_text}")
        else:
            # Count valid moves up to this point
            valid_count = 0
            for i in range(move_idx + 1):
                if all_moves[i][1]:  # If move is valid
                    valid_count += 1
            
            # Create a board state representation
            board_state = [[0 for _ in range(8)] for _ in range(8)]
            
            # Set initial pieces
            board_state[3][3] = board.WHITE  # d4 in standard notation
            board_state[3][4] = board.BLACK  # e4 in standard notation
            board_state[4][3] = board.BLACK  # d5 in standard notation
            board_state[4][4] = board.WHITE  # e5 in standard notation
            
            # Apply all valid moves up to valid_count
            for m_i in range(valid_count):
                # Get board modifications
                m: List[Tuple[int, int, int]] = board.board.history[m_i]
                
                # For each modification
                for row, col, p in m:
                    # Update the board state
                    board_state[row][col] = p
            
            # Draw all pieces based on the current board state
            for row in range(8):
                for col in range(8):
                    p = board_state[row][col]
                    if p == board.BLACK:
                        ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='black'))
                    elif p == board.WHITE:
                        ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='white'))
            
            # Get current move
            move_text = all_moves[move_idx][0]
            
            # Use the stored player information
            player = "Black" if board.moves_player[valid_count-1] == board.BLACK else "White"
            
            # Check if there was a pass before this move
            if valid_count > 1 and board.moves_player[valid_count-1] == board.moves_player[valid_count-2]:
                # If the same player made two consecutive moves, it means the other player passed
                opposite_player = "White" if player == "Black" else "Black"
                ax.set_title(f"Move {valid_count}: {opposite_player} passed, {player} plays {move_text}")
            else:
                ax.set_title(f"Move {valid_count}: {player} plays {move_text}")
        
        # Add column and row labels
        ax.set_xticks([i + 0.5 for i in range(8)])
        ax.set_yticks([i + 0.5 for i in range(8)])
        ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])
        
        # Set limits and aspect
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        
        return ax
    
    # Define button click handlers
    def on_prev_click(event):
        nonlocal current_move
        if current_move > -1:
            current_move -= 1
            draw_board(current_move)
            fig.canvas.draw_idle()
    
    def on_next_click(event):
        nonlocal current_move
        if current_move < len(all_moves) - 1:
            # Store the current move to check if we're showing an invalid move
            prev_move = current_move
            
            # Move to the next move
            current_move += 1
            
            # If we were showing an invalid move and clicked next, skip to the next valid move
            if prev_move >= 0 and prev_move < len(all_moves) and not all_moves[prev_move][1]:
                # Find the next valid move
                while current_move < len(all_moves) and not all_moves[current_move][1]:
                    current_move += 1
                
                # If we reached the end, stay at the last invalid move
                if current_move >= len(all_moves):
                    current_move = prev_move
            
            # Draw the board
            draw_board(current_move)
            fig.canvas.draw_idle()
    
    # Connect button events
    prev_button.on_clicked(on_prev_click)
    next_button.on_clicked(on_next_click)
    
    # Create frames for animation (from initial board to final move)
    frames = range(-1, len(all_moves))
    
    # Create the animation with 1 second per frame
    animation = FuncAnimation(
        fig, 
        draw_board, 
        frames=frames, 
        interval=1000,  # 1000 ms = 1 second
        blit=False,
        repeat=False
    )
    
    # Apply tight layout to the figure
    plt.tight_layout()
    
    # Check if running in Jupyter notebook
    if is_jupyter():
        # In Jupyter, return the animation for inline display
        return animation
    else:
        # In regular Python scripts, show the animation and return None
        plt.show()
        return animation


def plot_attention_matrix(
        qk_matrix,
        tokens,
        head_idx=0,
        normalize=False
):
    """
    Affiche la matrice de similarité QK (avant softmax).

    Args:
        qk_matrix: torch.Tensor de forme (batch, n_heads, seq_len, seq_len)
        tokens: liste de str, les tokens associés à la séquence
        head_idx: index du head à afficher
        normalize: si True applique softmax pour convertir en distribution d'attention
    """
    # on sélectionne le head demandé
    att = qk_matrix[0, head_idx].detach().cpu().numpy()

    # optionnel : normalisation softmax (sinon c’est juste QK^T brut)
    if normalize:
        import torch.nn.functional as F
        import torch
        att = torch.softmax(torch.tensor(att), dim=-1).numpy()
    # end if

    # Dégradé blanc → rouge
    cmap = plt.cm.Reds
    cmap.set_under("white")  # for exact 0

    plt.figure(figsize=(8, 6))
    plt.imshow(att, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
    plt.title(f"Attention (Head {head_idx})")
    plt.xlabel("Key tokens")
    plt.ylabel("Query tokens")
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.grid(False)
    plt.colorbar().remove()  # pas de barre latérale
    plt.show()
# end def plot_attention_matrix


def plot_heads_attention(qk_matrix, tokens, layer_idx=0):
    """
    Affiche toutes les têtes d'une couche dans une seule figure,
    avec un dégradé blanc (0.0) → rouge (1.0).

    Args:
        qk_matrix: torch.Tensor (batch, n_heads, seq_len, seq_len)
        tokens: liste de str (tokens associés)
        layer_idx: index de la couche
    """
    att = qk_matrix[0].detach().cpu().numpy()  # shape: (n_heads, seq_len, seq_len)
    n_heads = att.shape[0]

    # Grille auto (2 colonnes pour la lisibilité)
    ncols = 4 if n_heads >= 4 else n_heads
    nrows = int(np.ceil(n_heads / ncols))

    cmap = plt.cm.Reds
    cmap.set_under("white")

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()

    for h in range(n_heads):
        ax = axes[h]
        ax.imshow(att[h], cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"Layer {layer_idx}, Head {h}")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        ax.set_yticklabels(tokens, fontsize=6)
        ax.grid(False)
    # cacher les cases vides si n_heads < nrows*ncols

    for h in range(n_heads, len(axes)):
        axes[h].axis("off")

    plt.tight_layout()
    plt.show()
# end def plot_heads_attention


def show_linear_probe_samples(data, num_samples=3, random_seed=None):
    """
    Display random samples from the linear probe training data.
    
    This function visualizes the relationship between move sequences (x) and 
    board representations (y) for random samples from the linear probe training data.
    
    Args:
        data (Dict[int, Tuple[torch.Tensor, torch.LongTensor]]): Dictionary where keys are sequence lengths
            and values are tuples of (x, y) where:
            - x: tensor of move sequences (shape: [num_sequences, sequence_length])
            - y: tensor of board representations (shape: [num_sequences, 64])
        num_samples (int): Number of random samples to display (default: 3)
        random_seed (int, optional): Random seed for reproducibility
    
    Returns:
        Union[plt.Figure, None]: In Jupyter notebooks, returns the figure object for inline display.
                                In regular Python scripts, returns None after displaying the figure.
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Get the mapping from move IDs to move notations
    id_to_move = create_id_to_move_mapping()
    
    # Create a figure with subplots for each sample
    fig = plt.figure(figsize=(15, 5 * num_samples))
    
    # Get all available sequence lengths
    lengths = list(data.keys())
    
    # Counter for subplot positioning
    subplot_idx = 1
    
    # For each sample
    for i in range(num_samples):
        # Randomly select a sequence length
        length_idx = random.randint(0, len(lengths) - 1)
        length = lengths[length_idx]
        
        # Get the data for this length
        x, y = data[length]
        
        # Randomly select a sample from this length
        sample_idx = random.randint(0, x.shape[0] - 1)
        
        # Get the move sequence and board representation for this sample
        move_sequence = x[sample_idx].cpu().numpy()
        board_repr = y[sample_idx].cpu().numpy()
        
        # Convert move IDs to move notations
        move_notations = [id_to_move[move_id.item()] for move_id in x[sample_idx] if move_id.item() != 0]
        
        # Create subplot for the move sequence
        ax1 = fig.add_subplot(num_samples, 2, subplot_idx)
        subplot_idx += 1
        
        # Display the move sequence
        ax1.axis('off')
        ax1.set_title(f"Sample {i+1}: Move Sequence (length {length})")
        
        # Format the move sequence as a readable string
        move_text = "Moves: " + " → ".join(move_notations)
        ax1.text(0.5, 0.5, move_text, ha='center', va='center', wrap=True, fontsize=12)
        
        # Create subplot for the board representation
        ax2 = fig.add_subplot(num_samples, 2, subplot_idx)
        subplot_idx += 1
        
        # Display the board representation as a visual board
        ax2.set_title(f"Sample {i+1}: Board Representation")
        
        # Draw the green background
        ax2.add_patch(plt.Rectangle((0, 0), 8, 8, color='green'))
        
        # Draw the grid lines
        for j in range(9):
            ax2.plot([j, j], [0, 8], 'k-', lw=1)
            ax2.plot([0, 8], [j, j], 'k-', lw=1)
        
        # Draw the pieces based on the board representation
        for row in range(8):
            for col in range(8):
                # Calculate the index in the 1D board representation
                # The board representation is ordered by columns then rows (a1, a2, ..., h8)
                idx = col * 8 + row
                
                piece = board_repr[idx]
                
                if piece == 1:  # White
                    ax2.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='white'))
                elif piece == 2:  # Black
                    ax2.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.4, color='black'))
        
        # Add column and row labels
        ax2.set_xticks([i + 0.5 for i in range(8)])
        ax2.set_yticks([i + 0.5 for i in range(8)])
        ax2.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        ax2.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8'])
        
        # Set limits and aspect
        ax2.set_xlim(0, 8)
        ax2.set_ylim(0, 8)
        ax2.set_aspect('equal')
    
    # Apply tight layout to the figure
    plt.tight_layout()
    
    # Check if running in Jupyter notebook
    if is_jupyter():
        # In Jupyter, return the figure for inline display
        return fig
    else:
        # In regular Python scripts, show the figure and return None
        plt.show()
        return None
# end def show_linear_probe_samples
