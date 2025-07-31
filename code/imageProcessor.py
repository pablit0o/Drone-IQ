"""
imageProcessor.py

This module is used to load in SD cards into a
separate folder automatically. The only data that
it will extract from SD cards is JPG/JPEG-formatted 
images to use as input for our AI model. It is 
important to load in the SD card while the program 
is running, not before or after program execution.
Additionally, it lets the user select which files from
the overall SD card they want to transfer onto the
Valid Data folder.

Author: Pablo Silva
Date: 07-24-2025
Last Updated: 07-25-2025
"""
import os
import shutil
import wmi

def get_mounted_drives():
    """Returns a list of mounted drive letters on Windows. (ex. D:\\ or G:\\)"""
    c = wmi.WMI()
    drives = []
    # Check for each folder/path in Windows OS
    for drive in c.Win32_LogicalDisk():
        if drive.DriveType == 2: # Removable Disk
            drives.append(drive.DeviceID + "\\")
    return drives

def process_sd_card(sd_path, DESTINATION_FOLDER):
    """
    Processes a single SD card, finding drone photos (JPG/JPEG) and copying them
    to the DESTINATION_FOLDER.
    Returns True if any files were copied, False otherwise.
    """
    print(f"SD Card detected at: {sd_path}")
    drone_photo_path = None

    # Find the DCIM folder inside the SD card, may need to search further subdirectories
    for root, dirs, files in os.walk(sd_path):
        if "DCIM" in dirs:
            dcim_path = os.path.join(root, "DCIM")
            # Iterate through subdirectories within DCIM
            for sub_dir_name in os.listdir(dcim_path):
                full_sub_dir_path = os.path.join(dcim_path, sub_dir_name)
                if os.path.isdir(full_sub_dir_path):
                    # Helps find if this is the actual folder where images are stored
                    if any(f.lower().endswith(('.jpg', '.jpeg')) for f in os.listdir(full_sub_dir_path)):
                        drone_photo_path = full_sub_dir_path
                        print(f"Identified drone photo folder: {drone_photo_path}")
                        break # Found the media folder, no need to search more sub-dirs
            if drone_photo_path:
                break # Found the main DCIM path, stop os.walk

    if not drone_photo_path:
        print("No drone photo directory found.")
        return False
    
    copied_count = 0
    # Iterate through all files in the identified drone photo path
    for filename in os.listdir(drone_photo_path):
        source_file = os.path.join(drone_photo_path, filename)
        destination_file = os.path.join(DESTINATION_FOLDER, filename)

        # Check if it is an acceptable extension
        if os.path.isfile(source_file) and filename.lower().endswith(('.jpg', '.jpeg')):
            if os.path.exists(destination_file):
                print(f"Overwriting existing file: {filename}")
            shutil.copy2(source_file, destination_file)
            print(f"Copied: {filename}")
            copied_count += 1
            
    print(f"Finished copying from SD card. Copied {copied_count} files.")
    return True

def list_jpg_files(folder_path):
    """
    Lists all JPG/JPEG files in the given folder.
    Returns a list of their full paths.
    """
    jpg_files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            jpg_files.append(os.path.join(folder_path, filename))
    return jpg_files

def transfer_images(source_folder, valid_data_folder):
    """
    Allows the user to select N JPG images from source_folder and transfer them
    to valid_data_folder. Handles user input for single, multiple, or range selections.
    """
    image_paths = list_jpg_files(source_folder)
    if not image_paths:
        print(f"No JPG images found in folder.")
        return
    
    print("Available JPG images:")
    for i, img_path in enumerate(image_paths):
        # Index starting at 1 for user
        print(f"{i+1}. {os.path.basename(img_path)}") 
    
    selected_count = 0
    while True:
        selection_input = input("\nEnter numbers of images to transfer ('1 3 5' or '1-5') 'all' for all, or 'done' to finish: ").lower().strip()

        if selection_input == 'done':
            break
        
        elif selection_input == 'all':
            selected_indices = list(range(len(image_paths)))
       
        else:
            selected_indices = []
            
            # Input slicing by comma, based on user input
            parts = selection_input.replace(',', ' ').split()
            valid_input = True
            
            for part in parts:
                try:
                    if '-' in part: # Range selection
                        start, end = map(int, part.split('-'))
                        # Change index to 0-based
                        if 1 <= start <= end <= len(image_paths):
                            selected_indices.extend(range(start - 1, end))
                        else:
                            print(f"Invalid range: {part}. Numbers must be within 1 and {len(image_paths)}.")
                            valid_input = False
                            break
                    else: # Single number selection
                        idx = int(part) - 1 # Adjust again to 0-based
                        if 0 <= idx < len(image_paths):
                            selected_indices.append(idx)
                        else:
                            print(f"Invalid number: {part}. Please enter a number between 1 and {len(image_paths)}.")
                            valid_input = False
                            break
                except ValueError: # Non-integer inputs
                    print(f"Invalid input format: '{part}'. Please enter numbers, ranges, 'all', or 'done'.")
                    valid_input = False
                    break
            if not valid_input:
                continue # Mock catch statement in case something went wrong

        # Remove duplicates and sort the indices to process in order
        selected_indices = sorted(list(set(selected_indices)))

        if not selected_indices:
            print("No images selected. Try again.")
            continue

        print("\nTransferring images...")
        current_batch_copied = 0
        for idx in selected_indices:
            source_file = image_paths[idx]
            destination_file = os.path.join(valid_data_folder, os.path.basename(source_file))
            
            # In case duplicates
            if os.path.exists(destination_file):
                print(f"Overwriting existing file: {os.path.basename(destination_file)}")
            shutil.copy2(source_file, destination_file)
            print(f"Copied: {os.path.basename(source_file)}")
            current_batch_copied += 1
            selected_count += 1

        if current_batch_copied > 0:
            print(f"Successfully copied {current_batch_copied} images in this batch.")
        
        # Ask user to select more or finish
        if selection_input != 'all': 
            more_selection = input("Do you want to select more images from this folder? (y/n): ").lower()
            if more_selection != 'y':
                break # Exit loop

    print(f"\nImage selection complete. Transferred {selected_count} images to {valid_data_folder}.")
