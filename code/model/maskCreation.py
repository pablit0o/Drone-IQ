"""
maskCreation.py

Author: Arjun Maganti
Date: 07-17-2025
Last Updated: 07-28-2025
"""

import os
import json
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import glob
from PIL import Image
import matplotlib.pyplot as plt
import re

def main(argmodel, argVIA_JSON_PATH, argIMAGES_DIR, argMASKS_OUTPUT_DIR, arginput_images_folder, argoutput_masks_folder, argannotated_masks, argpredicted_masks, argcomparisons_dir):
    model = argmodel
    VIA_JSON_PATH = argVIA_JSON_PATH
    IMAGES_DIR = argIMAGES_DIR
    MASKS_OUTPUT_DIR = argMASKS_OUTPUT_DIR
    input_images_folder = arginput_images_folder
    output_masks_folder = argoutput_masks_folder
    annotated_masks = argannotated_masks
    predicted_masks = argpredicted_masks
    comparisons_dir = argcomparisons_dir
    
    # Load the JSON
    with open(VIA_JSON_PATH) as f:
        data = json.load(f)

    metadata = data['_via_img_metadata']  # This is the main data

    # Iterate through each annotation
    for key, ann in tqdm(metadata.items()):
        filename = ann['filename']
        regions = ann['regions']
        img_path = os.path.join(IMAGES_DIR, filename)

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found: {filename}")
            continue

        height, width = img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw polygons on mask
        for region in regions:
            shape = region['shape_attributes']
            if shape['name'] != 'polygon':
                continue  # Skip non-polygon shapes
            x_points = shape['all_points_x']
            y_points = shape['all_points_y']
            points = np.array(list(zip(x_points, y_points)), dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

        # Save mask
        mask_filename = os.path.splitext(filename)[0] + '_mask.png'
        cv2.imwrite(os.path.join(MASKS_OUTPUT_DIR, mask_filename), mask)
        
    def save_predicted_masks(model, input_dir, output_dir, target_size=(512, 512)):
        """
        Generate and save predicted masks for all images in a directory

        Args:
            model: Your loaded DeepLab model
            input_dir: Directory containing input images
            output_dir: Directory to save predicted masks
            target_size: Model input size (height, width) - should match training size

        Returns:
            Number of successfully processed images
        """

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Find all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp', '*.PNG', '*.JPG', '*.JPEG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))

        if not image_files:
            print(f"No image files found in {input_dir}")
            return 0

        image_files.sort()
        print(f"Found {len(image_files)} images to process")

        successful_count = 0
        failed_files = []

        for image_path in tqdm(image_files, desc="Generating masks"):
            try:
                filename = os.path.basename(image_path)
                name_without_ext = os.path.splitext(filename)[0]

                # Load and preprocess image
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                original_shape = image_rgb.shape

                # Resize to model input size
                resized_image = cv2.resize(image_rgb, target_size)

                # Normalize and add batch dimension
                normalized_image = resized_image.astype(np.float32) / 255.0
                batched_image = np.expand_dims(normalized_image, axis=0)

                # Predict
                prediction = model.predict(batched_image, verbose=0)

                # Convert prediction to binary mask
                pred = prediction[0]  # Remove batch dimension
                class_mask = np.argmax(pred, axis=-1)  # Get class predictions
                binary_mask = (class_mask == 1).astype(np.uint8)  # Buildings = class 1

                # Resize back to original image size
                original_height, original_width = original_shape[:2]
                resized_mask = cv2.resize(binary_mask, (original_width, original_height),
                                        interpolation=cv2.INTER_NEAREST)

                # Convert to 0-255 range for saving
                mask_image = (resized_mask * 255).astype(np.uint8)

                # Save mask
                mask_filename = f"{name_without_ext}_mask.png"
                mask_path = os.path.join(output_dir, mask_filename)

                # Save using PIL
                mask_pil = Image.fromarray(mask_image, mode='L')
                mask_pil.save(mask_path)

                successful_count += 1

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                failed_files.append(filename)
                continue

        # Print summary
        print(f"\n=== MASK GENERATION COMPLETE ===")
        print(f"Successfully processed: {successful_count}/{len(image_files)} images")
        print(f"Masks saved to: {output_dir}")

        if failed_files:
            print(f"Failed to process: {failed_files}")

        return successful_count

    # Simple usage function
    def generate_all_masks(model, input_folder, output_folder):
        """
        Simplified function to generate all masks

        Args:
            model: Your loaded DeepLab model
            input_folder: Path to folder with input images
            output_folder: Path to folder where masks will be saved
        """

        return save_predicted_masks(
            model=model,
            input_dir=input_folder,
            output_dir=output_folder,
            target_size=(512, 512)
        )

    # Generate and save all masks
    count = generate_all_masks(model, input_images_folder, output_masks_folder)
    if count > 0:
        return 0
    print(f"Generated {count} masks!")

    def find_matching_files(original_dir, annotated_dir, predicted_dir):
        """
        Find matching files across three directories

        Args:
            original_dir: Directory with original images
            annotated_dir: Directory with annotated masks
            predicted_dir: Directory with predicted masks

        Returns:
            List of tuples (original_path, annotated_path, predicted_path)
        """
        # Get all files from each directory
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp', '*.PNG', '*.JPG', '*.JPEG']

        # Find original images
        original_files = []
        for ext in image_extensions:
            original_files.extend(glob.glob(os.path.join(original_dir, ext)))

        # Find annotated masks
        annotated_files = []
        for ext in image_extensions:
            annotated_files.extend(glob.glob(os.path.join(annotated_dir, ext)))

        # Find predicted masks
        predicted_files = []
        for ext in image_extensions:
            predicted_files.extend(glob.glob(os.path.join(predicted_dir, ext)))

        # Create dictionaries for easier matching
        original_dict = {}
        for file_path in original_files:
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0]
            original_dict[base_name] = file_path

        annotated_dict = {}
        for file_path in annotated_files:
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0]
            # Remove common suffixes like "_mask", "_annotated", etc.
            base_name = re.sub(r'(_mask|_annotated|_gt|_groundtruth)$', '', base_name)
            annotated_dict[base_name] = file_path

        predicted_dict = {}
        for file_path in predicted_files:
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0]
            # Remove common suffixes like "_mask", "_predicted", etc.
            base_name = re.sub(r'(_mask|_predicted|_pred)$', '', base_name)
            predicted_dict[base_name] = file_path

        # Find matching triplets
        matches = []
        for base_name in original_dict.keys():
            if base_name in annotated_dict and base_name in predicted_dict:
                matches.append((
                    original_dict[base_name],
                    annotated_dict[base_name],
                    predicted_dict[base_name]
                ))

        return matches

    def load_and_prepare_image(image_path, is_mask=False):
        """
        Load and prepare image for display

        Args:
            image_path: Path to image file
            is_mask: Whether this is a mask (binary) image

        Returns:
            Processed image array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            # Try with PIL if cv2 fails
            image = np.array(Image.open(image_path))

        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            if not is_mask:  # Only convert color images, not masks
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Handle mask images
        if is_mask:
            if len(image.shape) == 3:
                # Convert to grayscale if it's a color mask
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Ensure binary mask is in 0-1 range for display
            image = image.astype(np.float32) / 255.0
        else:
            # Ensure color image is in 0-1 range for display
            image = image.astype(np.float32) / 255.0

        return image

    def create_comparison_plot(original_path, annotated_path, predicted_path, save_path=None, show_plot=True):
        """
        Create side-by-side comparison plot

        Args:
            original_path: Path to original image
            annotated_path: Path to annotated mask
            predicted_path: Path to predicted mask
            save_path: Path to save the comparison (optional)
            show_plot: Whether to display the plot

        Returns:
            Figure object
        """
        # Load images
        original_img = load_and_prepare_image(original_path, is_mask=False)
        annotated_mask = load_and_prepare_image(annotated_path, is_mask=True)
        predicted_mask = load_and_prepare_image(predicted_path, is_mask=True)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title(f'Original Image\n{os.path.basename(original_path)}', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Annotated mask
        axes[1].imshow(annotated_mask, cmap='gray')
        axes[1].set_title(f'Ground Truth Mask\n{os.path.basename(annotated_path)}', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Predicted mask
        axes[2].imshow(predicted_mask, cmap='gray')
        axes[2].set_title(f'Predicted Mask\n{os.path.basename(predicted_path)}', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison saved to: {save_path}")

        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def create_overlay_comparison(original_path, annotated_path, predicted_path, save_path=None, show_plot=True):
        """
        Create comparison with masks overlaid on original image

        Args:
            original_path: Path to original image
            annotated_path: Path to annotated mask
            predicted_path: Path to predicted mask
            save_path: Path to save the comparison (optional)
            show_plot: Whether to display the plot
        """
        # Load images
        original_img = load_and_prepare_image(original_path, is_mask=False)
        annotated_mask = load_and_prepare_image(annotated_path, is_mask=True)
        predicted_mask = load_and_prepare_image(predicted_path, is_mask=True)

        # Resize masks to match original image if needed
        if original_img.shape[:2] != annotated_mask.shape[:2]:
            annotated_mask = cv2.resize(annotated_mask, (original_img.shape[1], original_img.shape[0]))
        if original_img.shape[:2] != predicted_mask.shape[:2]:
            predicted_mask = cv2.resize(predicted_mask, (original_img.shape[1], original_img.shape[0]))

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Original image
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # Ground truth overlay
        axes[0, 1].imshow(original_img)
        axes[0, 1].imshow(annotated_mask, cmap='Reds', alpha=0.5)
        axes[0, 1].set_title('Original + Ground Truth (Red)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        # Predicted overlay
        axes[1, 0].imshow(original_img)
        axes[1, 0].imshow(predicted_mask, cmap='Blues', alpha=0.5)
        axes[1, 0].set_title('Original + Predicted (Blue)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        # Both overlays
        axes[1, 1].imshow(original_img)
        axes[1, 1].imshow(annotated_mask, cmap='Reds', alpha=0.4)
        axes[1, 1].imshow(predicted_mask, cmap='Blues', alpha=0.4)
        axes[1, 1].set_title('Ground Truth (Red) + Predicted (Blue)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Overlay comparison saved to: {save_path}")

        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def compare_all_images(original_dir, annotated_dir, predicted_dir, output_dir=None,
                        max_images=None, comparison_type='side_by_side', show_plots=True):
        """
        Create comparisons for all matching images

        Args:
            original_dir: Directory with original images
            annotated_dir: Directory with annotated masks
            predicted_dir: Directory with predicted masks
            output_dir: Directory to save comparisons (optional)
            max_images: Maximum number of images to process (None for all)
            comparison_type: 'side_by_side' or 'overlay'
            show_plots: Whether to display plots

        Returns:
            Number of comparisons created
        """
        # Find matching files
        matches = find_matching_files(original_dir, annotated_dir, predicted_dir)

        if not matches:
            print("No matching files found across all three directories!")
            print(f"Original dir: {len(glob.glob(os.path.join(original_dir, '*')))} files")
            print(f"Annotated dir: {len(glob.glob(os.path.join(annotated_dir, '*')))} files")
            print(f"Predicted dir: {len(glob.glob(os.path.join(predicted_dir, '*')))} files")
            return 0

        print(f"Found {len(matches)} matching image sets")

        # Limit number of images if specified
        if max_images:
            matches = matches[:max_images]
            print(f"Processing first {len(matches)} images")

        # Create output directory if saving
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Process each match
        for i, (original_path, annotated_path, predicted_path) in enumerate(matches):
            try:
                base_name = os.path.splitext(os.path.basename(original_path))[0]

                # Determine save path
                save_path = None
                if output_dir:
                    save_filename = f"comparison_{comparison_type}_{base_name}.png"
                    save_path = os.path.join(output_dir, save_filename)

                # Create comparison
                if comparison_type == 'overlay':
                    create_overlay_comparison(original_path, annotated_path, predicted_path,
                                            save_path, show_plots)
                else:  # side_by_side
                    create_comparison_plot(original_path, annotated_path, predicted_path,
                                        save_path, show_plots)

                print(f"Processed {i+1}/{len(matches)}: {base_name}")

            except Exception as e:
                print(f"Error processing {base_name}: {str(e)}")
                continue

        print(f"\nCompleted processing {len(matches)} image comparisons")
        if output_dir:
            print(f"Saved comparisons to: {output_dir}")

        return len(matches)

    # Example usage function
    def main_comparison():
        """
        Main function to run comparisons - update paths as needed
        """
        # UPDATE THESE PATHS TO YOUR ACTUAL DIRECTORIES
        original_images_dir = argIMAGES_DIR
        annotated_masks_dir = annotated_masks
        predicted_masks_dir = predicted_masks
        output_comparisons_dir = comparisons_dir

        # Option 1: Side-by-side comparison
        print("Creating side-by-side comparisons...")
        compare_all_images(
            original_dir=original_images_dir,
            annotated_dir=annotated_masks_dir,
            predicted_dir=predicted_masks_dir,
            output_dir=output_comparisons_dir,
            max_images=None,  # Process first 5 images, set to None for all
            comparison_type='side_by_side',
            show_plots=True
        )


    if __name__ == "__main__":
        # Update the paths in main_comparison() function above, then run:
        main_comparison()
        #print("Update the directory paths in main_comparison() and run the function!")