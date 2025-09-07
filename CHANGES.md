# Changes Made to othello.py

## Issue Description
In othello.py, when extract image command, I want the images to be squared. So before saving, the image must be cropped to the smaller dimension to have a squared image.

## Changes Implemented

1. Added PIL Image import to handle image processing:
   ```python
   from PIL import Image
   ```

2. Created a new helper function `crop_to_square` that takes an image path, opens the image, crops it to make it square based on the smaller dimension, and saves it back to the same path:
   ```python
   def crop_to_square(image_path: str) -> None:
       """
       Crop an image to make it square based on the smaller dimension.
       
       Args:
           image_path (str): Path to the image file
       """
       # Open the image
       img = Image.open(image_path)
       
       # Get image dimensions
       width, height = img.size
       
       # Determine the smaller dimension
       min_dim = min(width, height)
       
       # Calculate cropping box (centered)
       left = (width - min_dim) // 2
       top = (height - min_dim) // 2
       right = left + min_dim
       bottom = top + min_dim
       
       # Crop the image
       cropped_img = img.crop((left, top, right, bottom))
       
       # Save the cropped image back to the same path
       cropped_img.save(image_path)
   ```

3. Modified the image saving process in the `extract_game_as_images` function to use the `crop_to_square` helper function after saving the images with matplotlib:
   - For the initial board state:
     ```python
     # Save the initial board
     initial_board_path = f"{output_dir}/board_initial.png"
     plt.savefig(initial_board_path, bbox_inches='tight', pad_inches=0)
     plt.close(fig)
     
     # Crop the image to make it square
     crop_to_square(initial_board_path)
     ```
   
   - For each move's board state:
     ```python
     # Save the board state
     move_board_path = f"{output_dir}/board_move_{move_idx+1:03d}.png"
     plt.savefig(move_board_path, bbox_inches='tight', pad_inches=0)
     plt.close(fig)
     
     # Crop the image to make it square
     crop_to_square(move_board_path)
     ```

4. Updated the docstring of the `extract_game_as_images` function to reflect that the images will be cropped to be square:
   ```python
   def extract_game_as_images(game_file: str, game_index: int, output_dir: str, image_size: int = 8) -> None:
       """
       Extract each state of an Othello game as images without borders and title.
       The images are cropped to be square based on the smaller dimension.
       
       Args:
           game_file (str): Path to the binary file containing games
           game_index (int): Index of the game to extract
           output_dir (str): Directory to save the images
           image_size (int): Size of the output images in inches (default: 8)
       """
   ```

## Testing

A test script was created to verify that the `crop_to_square` function works correctly. The test creates images with different dimensions (wide, tall, and square) and applies the `crop_to_square` function to each image. The test confirms that all images are square after cropping.

The test results show that:
1. A wide image (initially 620x462) becomes 462x462 (cropped width to match height)
2. A tall image (initially 465x616) becomes 465x465 (cropped height to match width)
3. A square image (initially 542x539) becomes 539x539 (cropped to make it perfectly square)

This confirms that the implementation correctly crops images to make them square based on the smaller dimension, which satisfies the requirements specified in the issue description.