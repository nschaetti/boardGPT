"""
Rich-based logging utility for boardGPT.

This module provides a consistent logging approach using Rich for the entire boardGPT codebase.
It supports backtrack logging (not verbose) as well as other logging levels.
"""

import os
import sys
from typing import Optional, Any, Dict

from rich.console import Console
from rich.traceback import install as install_rich_traceback
from rich.theme import Theme

# Define a custom theme for our logs
CUSTOM_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "debug": "dim white",
    "success": "green",
})

# Create a global console instance with our theme
console = Console(theme=CUSTOM_THEME)

# Install Rich traceback handler (not verbose)
install_rich_traceback(show_locals=False, width=None, word_wrap=True)


def info(message: str, **kwargs: Any) -> None:
    """
    Log an informational message.
    
    Args:
        message: The message to log
        **kwargs: Additional arguments to pass to console.print
    """
    console.print(f"[info]INFO:[/info] {message}", **kwargs)


def warning(message: str, **kwargs: Any) -> None:
    """
    Log a warning message.
    
    Args:
        message: The message to log
        **kwargs: Additional arguments to pass to console.print
    """
    console.print(f"[warning]WARNING:[/warning] {message}", **kwargs)


def error(message: str, **kwargs: Any) -> None:
    """
    Log an error message.
    
    Args:
        message: The message to log
        **kwargs: Additional arguments to pass to console.print
    """
    console.print(f"[error]ERROR:[/error] {message}", **kwargs)


def debug(message: str, **kwargs: Any) -> None:
    """
    Log a debug message.
    
    Args:
        message: The message to log
        **kwargs: Additional arguments to pass to console.print
    """
    console.print(f"[debug]DEBUG:[/debug] {message}", **kwargs)


def success(message: str, **kwargs: Any) -> None:
    """
    Log a success message.
    
    Args:
        message: The message to log
        **kwargs: Additional arguments to pass to console.print
    """
    console.print(f"[success]SUCCESS:[/success] {message}", **kwargs)


def print_exception(show_locals: bool = False, **kwargs: Any) -> None:
    """
    Print the current exception with a traceback.
    
    Args:
        show_locals: Whether to show local variables in the traceback
        **kwargs: Additional arguments to pass to console.print_exception
    """
    console.print_exception(show_locals=show_locals, **kwargs)


def log_exception(message: Optional[str] = None, show_locals: bool = False, **kwargs: Any) -> None:
    """
    Log an exception with an optional message.
    
    Args:
        message: An optional message to display before the exception
        show_locals: Whether to show local variables in the traceback
        **kwargs: Additional arguments to pass to console.print_exception
    """
    if message:
        error(message)
    print_exception(show_locals=show_locals, **kwargs)