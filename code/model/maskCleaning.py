"""
maskCleaning.py

This module is utilized to clean up the
building masks produced by our DeepLearningV3
model. The way it is cleaned up is through a
graph-traversal problem, where we try to find the
largest group of interconnected 1s in a 2D
binary array, where a 1 represents a building
pixel. The DFS implementation is at worst
quadratic time, or O(nm) where n is num of
cols and m is num of rows.

Author: Pablo Silva
Date: 07-27-2025
Last Updated: 07-27-2025
"""

import os
from PIL import Image
import numpy as np
import sys
from tqdm import tqdm 

recursion_limit = 5000000
sys.setrecursionlimit(recursion_limit)

def dfs_search(binary_array):
    """
    Implementation that first checks for
    rows and columns, then uses DFS to find
    largest group. After largest group is found,
    any other sub-group is converted to 0s.
    """
    rows = len(binary_array)
    cols = len(binary_array[0])

    # Keep original for debugging
    binary_array = [row[:] for row in binary_array]

    # Visited cells during DFS
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    largest_group_size = 0
    largest_group = []

    def dfs(r, c, current_group):
        """
        Implementation of depth first search,
        where r is number of rows and c is number
        of columns. It first starts by detecting a 1,
        then traversing (or backtracking) recursively
        for other nearby (horizontally/vertically) connected
        nodes. It returns through recursion by calling the parent
        node. Also, its important that it uses a stack
        (LIFO) to keep track of visited nodes, thus
        avoiding cycles aka infinite loops.
        """
        if (r < 0 or r >= rows or c < 0 or c >= cols or
                visited[r][c] or binary_array[r][c] == 0):
            return 0

        visited[r][c] = True
        current_group.append((r, c))
        count = 1  # Count current node

        # Explore neighbor nodes
        count += dfs(r + 1, c, current_group) # Down
        count += dfs(r - 1, c, current_group) # Up
        count += dfs(r, c + 1, current_group) # Right
        count += dfs(r, c - 1, current_group) # Left

        return count

    # Comparison, filter through largest group.
    for r in range(rows):
        for c in range(cols):
            if binary_array[r][c] == 1 and not visited[r][c]:
                current_group = []
                current_group_size = dfs(r, c, current_group)

                if current_group_size > largest_group_size:
                    largest_group_size = current_group_size
                    largest_group = current_group

                # If multiple groups are the same size, first one found is kept
                elif current_group_size == largest_group_size:
                    pass

    # Create output array, set non-largest to 0s
    output_array = [[0 for _ in range(cols)] for _ in range(rows)]
    for r, c in largest_group:
        output_array[r][c] = 1

    return output_array

def png_conversion(binary_array, output_path="output_mask.png"):
    """
    Converts a 2D binary array (0s and 1s) into a black and white PNG image.
    Note that the binary_array MUST be normalized to 0 and 1, and the output
    path can be editted if needed. Also assumes input validation.
    """

    height = len(binary_array)
    width = len(binary_array[0])

    # Creates 1-bit image
    img = Image.new('1', (width, height))
    pixels = img.load() # Get a pixel access object

    # Iterate through the binary array and set pixel values
    for y in range(height):
        for x in range(width):
            # If value = 1, set to white
            # If value = 0, set to black
            pixels[x, y] = binary_array[y][x] * 255

    # Save the image
    try:
        img.save(output_path)
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")

def process_masks(input_dir, output_dir):
    """
    Processes all PNG mask files in an input directory, applies DFS cleaning,
    and saves the cleaned masks to an output directory.
    """
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    mask_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    if not mask_files:
        print(f"No PNG mask files found in {input_dir}")
        return 0

    processed_count = 0
    for filename in tqdm(mask_files, desc="Cleaning masks"):
        input_path = os.path.join(input_dir, filename)
        
        # Append '_clean' to the original filename before the extension
        output_filename = os.path.splitext(filename)[0] + '_clean.png'
        output_path = os.path.join(output_dir, output_filename)

        try:
            img = Image.open(input_path).convert('L') # Convert to grayscale
            np_img = np.array(img)
            binary_array = (np_img > 128).astype(int) # Assuming masks are binary (0 or 255)

            result_array = dfs_search(binary_array)
            png_conversion(result_array, output_path)
            processed_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    print(f"Successfully cleaned {processed_count} masks and saved to {output_dir}")
    return processed_count

def main(INPUT_DIR, OUTPUT_DIR):
    process_masks(INPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()
