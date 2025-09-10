#!/usr/bin/env python3
"""
Test script for the crop_to_square function in othello.py
"""

import os
import sys
import matplotlib.pyplot as plt
from simulators.othello import crop_to_square
from PIL import Image

def test_crop_to_square():
    """
    Test the crop_to_square function to ensure it produces square images.
    """
    # Parameters for the test
    output_dir = "test_images"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating test images with different dimensions...")
    
    # Create a set of test images with different dimensions
    test_images = [
        {"name": "wide_image.png", "width": 800, "height": 600},
        {"name": "tall_image.png", "width": 600, "height": 800},
        {"name": "square_image.png", "width": 700, "height": 700}
    ]
    
    for img_info in test_images:
        # Create a figure with the specified dimensions
        fig, ax = plt.subplots(figsize=(img_info["width"]/100, img_info["height"]/100))
        
        # Draw a green background
        ax.add_patch(plt.Rectangle((0, 0), 10, 10, color='green'))
        
        # Draw some grid lines
        for i in range(11):
            ax.plot([i, i], [0, 10], 'k-', lw=1)
            ax.plot([0, 10], [i, i], 'k-', lw=1)
        
        # Remove borders, ticks, and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Save the image
        img_path = os.path.join(output_dir, img_info["name"])
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        print(f"Created {img_info['name']} with dimensions {img_info['width']}x{img_info['height']}")
    
    print("\nApplying crop_to_square to test images...")
    
    # Apply crop_to_square to each test image
    for img_info in test_images:
        img_path = os.path.join(output_dir, img_info["name"])
        
        # Get original dimensions
        original_img = Image.open(img_path)
        original_width, original_height = original_img.size
        print(f"{img_info['name']} before cropping: {original_width}x{original_height}")
        
        # Apply crop_to_square
        crop_to_square(img_path)
        
        # Get new dimensions
        cropped_img = Image.open(img_path)
        cropped_width, cropped_height = cropped_img.size
        print(f"{img_info['name']} after cropping: {cropped_width}x{cropped_height}")
        
        # Verify that the image is now square
        if cropped_width == cropped_height:
            print(f"{img_info['name']} is now square ✓")
        else:
            print(f"{img_info['name']} is not square ✗")
    
    print("\nTest completed.")

if __name__ == "__main__":
    test_crop_to_square()