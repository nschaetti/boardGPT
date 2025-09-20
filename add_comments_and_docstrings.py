#!/usr/bin/env python3
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

"""
Script to add GPLv3 license headers, docstrings, and end block comments to Python files.

This script processes all Python files in the boardGPT repository, adding:
1. GPLv3 license header to each file
2. File docstrings if missing
3. End block comments for control structures (if, for, while, def, class, etc.)
"""

import os
import re
import sys
import ast
import tokenize
from io import StringIO
from typing import List, Dict, Tuple, Set, Optional

# GPLv3 license header template
GPL_LICENSE = '''"""
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
'''

# File docstring template
FILE_DOCSTRING_TEMPLATE = '''"""
{file_description}

This module provides {module_functionality}.
"""
'''


def find_python_files(directory: str) -> List[str]:
    """
    Find all Python files in the given directory and its subdirectories.
    
    Args:
        directory (str): The directory to search in
        
    Returns:
        List[str]: List of paths to Python files
    """
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files
# end def find_python_files


def has_license_header(content: str) -> bool:
    """
    Check if the file already has a license header.
    
    Args:
        content (str): The content of the file
        
    Returns:
        bool: True if the file has a license header, False otherwise
    """
    return "Copyright (C)" in content and "GNU General Public License" in content
# end def has_license_header


def has_file_docstring(content: str) -> bool:
    """
    Check if the file already has a file docstring.
    
    Args:
        content (str): The content of the file
        
    Returns:
        bool: True if the file has a file docstring, False otherwise
    """
    try:
        tree = ast.parse(content)
        return (ast.get_docstring(tree) is not None)
    except SyntaxError:
        return False
# end def has_file_docstring


def add_license_and_docstring(content: str, filename: str) -> str:
    """
    Add license header and file docstring to the file content if they don't exist.
    
    Args:
        content (str): The content of the file
        filename (str): The name of the file
        
    Returns:
        str: The updated content
    """
    # Remove empty lines at the beginning of the file
    content = content.lstrip()
    
    # Add license header if it doesn't exist
    if not has_license_header(content):
        content = GPL_LICENSE + "\n" + content
    
    # Add file docstring if it doesn't exist
    if not has_file_docstring(content):
        # Extract module name from filename
        module_name = os.path.basename(filename).replace('.py', '')
        
        # Create a simple file description and functionality
        file_description = f"{module_name.replace('_', ' ').title()} module"
        module_functionality = f"functionality related to {module_name.replace('_', ' ')}"
        
        # Create the docstring
        file_docstring = FILE_DOCSTRING_TEMPLATE.format(
            file_description=file_description,
            module_functionality=module_functionality
        )
        
        # Find the position to insert the docstring (after license header if it exists)
        if '"""' in content:
            # Find the end of the first docstring (which should be the license header)
            end_of_license = content.find('"""', content.find('"""') + 3) + 3
            content = content[:end_of_license] + "\n\n" + file_docstring + content[end_of_license:]
        else:
            content = file_docstring + "\n" + content
    
    return content
# end def add_license_and_docstring


def add_end_block_comments(content: str) -> str:
    """
    Add end block comments to control structures (if, for, while, def, class, etc.).
    
    Args:
        content (str): The content of the file
        
    Returns:
        str: The updated content with end block comments
    """
    lines = content.split('\n')
    result_lines = []
    
    # Stack to keep track of indentation levels and block types
    stack = []  # (indentation_level, block_type, block_name)
    
    for i, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            result_lines.append(line)
            continue
        
        # Calculate indentation level
        indentation = len(line) - len(line.lstrip())
        
        # Check if this line closes any blocks
        while stack and indentation <= stack[-1][0]:
            indent_level, block_type, block_name = stack.pop()
            
            # Don't add end comment if the previous line already has one
            prev_line = result_lines[-1].strip()
            if not prev_line.startswith('# end'):
                # Add the end block comment to the previous line
                if block_type == 'def' or block_type == 'class':
                    end_comment = f"# end {block_type} {block_name}"
                else:
                    end_comment = f"# end {block_type}"
                
                # If the previous line is not empty, add the comment
                if prev_line:
                    result_lines[-1] = result_lines[-1] + "  " + end_comment
                else:
                    # If the previous line is empty, add the comment with proper indentation
                    result_lines[-1] = ' ' * indent_level + end_comment
        
        # Check if this line starts a new block
        stripped_line = line.strip()
        
        # Check for function definitions
        if stripped_line.startswith('def '):
            # Extract function name
            match = re.match(r'def\s+([a-zA-Z0-9_]+)', stripped_line)
            if match:
                function_name = match.group(1)
                stack.append((indentation, 'def', function_name))
        
        # Check for class definitions
        elif stripped_line.startswith('class '):
            # Extract class name
            match = re.match(r'class\s+([a-zA-Z0-9_]+)', stripped_line)
            if match:
                class_name = match.group(1)
                stack.append((indentation, 'class', class_name))
        
        # Check for if statements
        elif stripped_line.startswith('if ') and stripped_line.endswith(':'):
            stack.append((indentation, 'if', ''))
        
        # Check for elif statements
        elif stripped_line.startswith('elif ') and stripped_line.endswith(':'):
            # Close the previous if/elif block
            if stack and stack[-1][1] in ('if', 'elif'):
                indent_level, block_type, _ = stack.pop()
                
                # Don't add end comment if the previous line already has one
                prev_line = result_lines[-1].strip()
                if not prev_line.startswith('# end'):
                    end_comment = f"# end {block_type}"
                    if prev_line:
                        result_lines[-1] = result_lines[-1] + "  " + end_comment
                    else:
                        result_lines[-1] = ' ' * indent_level + end_comment
            
            stack.append((indentation, 'elif', ''))
        
        # Check for else statements
        elif stripped_line.startswith('else:'):
            # Close the previous if/elif block
            if stack and stack[-1][1] in ('if', 'elif'):
                indent_level, block_type, _ = stack.pop()
                
                # Don't add end comment if the previous line already has one
                prev_line = result_lines[-1].strip()
                if not prev_line.startswith('# end'):
                    end_comment = f"# end {block_type}"
                    if prev_line:
                        result_lines[-1] = result_lines[-1] + "  " + end_comment
                    else:
                        result_lines[-1] = ' ' * indent_level + end_comment
            
            stack.append((indentation, 'else', ''))
        
        # Check for for loops
        elif stripped_line.startswith('for ') and stripped_line.endswith(':'):
            stack.append((indentation, 'for', ''))
        
        # Check for while loops
        elif stripped_line.startswith('while ') and stripped_line.endswith(':'):
            stack.append((indentation, 'while', ''))
        
        # Check for try blocks
        elif stripped_line == 'try:':
            stack.append((indentation, 'try', ''))
        
        # Check for except blocks
        elif stripped_line.startswith('except ') and stripped_line.endswith(':'):
            # Close the previous try/except block
            if stack and stack[-1][1] in ('try', 'except'):
                indent_level, block_type, _ = stack.pop()
                
                # Don't add end comment if the previous line already has one
                prev_line = result_lines[-1].strip()
                if not prev_line.startswith('# end'):
                    end_comment = f"# end {block_type}"
                    if prev_line:
                        result_lines[-1] = result_lines[-1] + "  " + end_comment
                    else:
                        result_lines[-1] = ' ' * indent_level + end_comment
            
            stack.append((indentation, 'except', ''))
        
        # Check for finally blocks
        elif stripped_line == 'finally:':
            # Close the previous try/except block
            if stack and stack[-1][1] in ('try', 'except'):
                indent_level, block_type, _ = stack.pop()
                
                # Don't add end comment if the previous line already has one
                prev_line = result_lines[-1].strip()
                if not prev_line.startswith('# end'):
                    end_comment = f"# end {block_type}"
                    if prev_line:
                        result_lines[-1] = result_lines[-1] + "  " + end_comment
                    else:
                        result_lines[-1] = ' ' * indent_level + end_comment
            
            stack.append((indentation, 'finally', ''))
        
        # Check for with blocks
        elif stripped_line.startswith('with ') and stripped_line.endswith(':'):
            stack.append((indentation, 'with', ''))
        
        result_lines.append(line)
    
    # Close any remaining blocks
    while stack:
        indent_level, block_type, block_name = stack.pop()
        
        # Add the end block comment to the last line
        if block_type == 'def' or block_type == 'class':
            end_comment = f"# end {block_type} {block_name}"
        else:
            end_comment = f"# end {block_type}"
        
        # If the last line is not empty, add the comment
        if result_lines[-1].strip():
            result_lines[-1] = result_lines[-1] + "  " + end_comment
        else:
            # If the last line is empty, add the comment with proper indentation
            result_lines[-1] = ' ' * indent_level + end_comment
    
    return '\n'.join(result_lines)
# end def add_end_block_comments


def process_file(file_path: str) -> None:
    """
    Process a single Python file to add license header, docstrings, and end block comments.
    
    Args:
        file_path (str): Path to the Python file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add license header and file docstring
        updated_content = add_license_and_docstring(content, file_path)
        
        # Add end block comments
        updated_content = add_end_block_comments(updated_content)
        
        # Write the updated content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"Processed: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
# end def process_file


def main() -> None:
    """
    Main function to process all Python files in the boardGPT repository.
    """
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find the boardGPT directory (assuming this script is in the repository root)
    boardgpt_dir = os.path.join(script_dir, 'boardGPT')
    
    if not os.path.isdir(boardgpt_dir):
        print(f"Error: boardGPT directory not found at {boardgpt_dir}")
        sys.exit(1)
    
    # Find all Python files in the boardGPT directory
    python_files = find_python_files(boardgpt_dir)
    
    print(f"Found {len(python_files)} Python files to process")
    
    # Process each file
    for file_path in python_files:
        process_file(file_path)
    
    print("Done!")
# end def main


if __name__ == "__main__":
    main()
# end if