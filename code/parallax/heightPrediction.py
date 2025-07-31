import numpy as np
from PIL import Image

def get_coordinates(image_path):
    """Get highest and lowest (x,y) coordinates of white pixels"""
    mask_img = Image.open(image_path)

    if mask_img.mode != 'L':
        mask_img = mask_img.convert('L')

    mask = np.array(mask_img)
    white_pixels = np.where(mask > 127)

    if len(white_pixels[0]) == 0:
        return None

    white_rows = white_pixels[0]
    white_cols = white_pixels[1]

    top_row = np.min(white_rows)
    bottom_row = np.max(white_rows)

    top_index = np.where(white_rows == top_row)[0][0]
    bottom_index = np.where(white_rows == bottom_row)[0][0]

    top_coords = (white_rows[top_index])
    bottom_coords = (white_rows[bottom_index])

    return top_coords, bottom_coords

def compare_two_images(image1_path, image2_path):
    """Compare two images and print coordinates"""
    result1 = get_coordinates(image1_path)
    result2 = get_coordinates(image2_path)

    return result1[0], result2[0], result1[1], result2[1]