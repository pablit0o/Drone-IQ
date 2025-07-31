"""
main.py

The main module has the overall control of the
entire program, below listed is a uncomprehensive
overview of the actions that main.py controls.

1) SD Card Insertion
2) Valid Data Transfer
3) DeepLabV3 Model
4) DFS Implementation
5) Height Prediction
6) Parallax Calculations
7) Data Visualization

Author: Pablo Silva
Date: 07-28-2025
Last Updated: 07-28-2025
"""
# Global imports
import time 
import os

# Local imports
import imageProcessor
import loadModel
import maskCreation
import maskCleaning
import heightPrediction
import parallax
#import dataVisualization

# Initialization
SD_CARD = True
DESTINATION_FOLDER = 'C:/Users/pbsil/Drone-IQ/droneData'
VALID_DATA_FOLDER = 'C:/Users/pbsil/Drone-IQ/validDroneData'
MODEL_PATH = 'C:/Users/pbsil/Drone-IQ/code/deeplabv3_building_model.h5'
VIA_JSON_PATH = 'C:/Users/pbsil/Drone-IQ/code/training_data_207.json'
IMAGES_DIR = 'C:/Users/pbsil/Drone-IQ/trainingData'
MASKS_OUTPUT_DIR = 'C:/Users/pbsil/Drone-IQ/masks'
INPUT_IMAGES_DIR = 'C:/Users/pbsil/Drone-IQ/validDroneData'
OUTPUT_MASKS_DIR = 'C:/Users/pbsil/Drone-IQ/droneMasks'
ANNOTATED_MASKS_DIR = 'C:/Users/pbsil/Drone-IQ/annotatedMasks'
PREDICTED_MASKS_DIR = 'C:/Users/pbsil/Drone-IQ/predictedMasks'
COMPARISONS_DIR = 'C:/Users/pbsil/Drone-IQ/comparisons'

# Identify later insertions
known_drives = imageProcessor.get_mounted_drives()
print(f"Currently connected drives: {known_drives}")

choice = input("Do you wish to skip the loading and transfering images? (y/n) ").lower()

if choice == 'y':
    SD_CARD = False

while SD_CARD:
    # 1) SD Card Insertion
    print("Waiting for SD card insertion...")
    SD_processed = False
    # Monitor for SD card insertion
    while True:
        current_drives = imageProcessor.get_mounted_drives()
        new_drives = [d for d in current_drives if d not in known_drives]
        if new_drives:
            print(f"New drive(s) detected: {', '.join(new_drives)}")
            time.sleep(2) # Cooldown
            for new_drive_path in new_drives:
                if imageProcessor.process_sd_card(new_drive_path, DESTINATION_FOLDER):
                    sd_card_processed_this_cycle = True 
            # Update drives
            known_drives = current_drives
            if sd_card_processed_this_cycle:
                break 
        else:
            print("No new SD card detected. Waiting...")
            time.sleep(3) # Cooldown
            # Lets user quit program
            user_input = input("Press 'q' to quit, otherwise refresh for SD cards: ").lower()
            if user_input == 'q':
                print("Exiting SD card monitoring.")
                break
        known_drives = current_drives
    if not sd_card_processed_this_cycle:
        print("No new images were copied from an SD card in this cycle.")
        proceed_selection = input("Do you want to manually select images from the existing data in the destination folder? (y/n): ").lower()
        if proceed_selection != 'y':
            print("Skipping image selection for this cycle.")
            # If no SD card insertion and user doesn't want to transfer images, ask for a restart of the cycle
            continue_overall = input("Do you want to start another program cycle (monitor for SD card, then select)? (y/n): ").lower()
            if continue_overall != 'y':
                SD_CARD = False # Terminate program loop
            continue 
    
    # 2) Valid Data Transfer
    imageProcessor.transfer_images(DESTINATION_FOLDER, VALID_DATA_FOLDER)

    # Check if they want to repeat process
    continue_program = input("\nDo you want to start another program cycle (monitor for SD card, then select)? (y/n): ").lower()
    if continue_program != 'y':
        SD_CARD = False # Terminate program


# 3) DeepLabV3 Model
model = loadModel.main(MODEL_PATH)
maskCreation.main(model, VIA_JSON_PATH, IMAGES_DIR, MASKS_OUTPUT_DIR, INPUT_IMAGES_DIR, OUTPUT_MASKS_DIR, ANNOTATED_MASKS_DIR, PREDICTED_MASKS_DIR, COMPARISONS_DIR)

# 4) DFS Implementation
maskCleaning.main(OUTPUT_MASKS_DIR, OUTPUT_MASKS_DIR) # No need to make an entirely new folder

# 5) Height Prediction
print(f"Scanning '{OUTPUT_MASKS_DIR}' for files...")
png1, png2 = None, None
        
for filename in os.listdir(OUTPUT_MASKS_DIR):
    # Builds full path
    full_path = os.path.join(OUTPUT_MASKS_DIR, filename)
    if os.path.isfile(full_path) and filename.lower().endswith('clean.png') and png1 == None:
        png1 = full_path
        print(f"First image successfully labeled as {full_path}")
    elif os.path.isfile(full_path) and filename.lower().endswith('clean.png') and png1 != None:
        png2 = full_path
        print(f"Second image successfully labeled as {full_path}")

y1, y2, y3, y4 = heightPrediction.compare_two_images(png1, png2)

# 6) Parallax Calculations
if y1 == y2:
    y2 += 1
if y3 == y4:
    y4 += 1
print(f"y1 = {y1}, y2 = {y2}, y3 = {y3}, y4 = {y4}")
height_meters = parallax.main(y1, y2, y3, y4)
height_feet = height_meters * 3.28084

print(f"Your building is {height_meters:,.2f} meters tall ({height_feet:,.2f} ft).")

# 7) Data Visualization


# Reset files
del_choice = input("Delete all files? (y/n) ").lower()
if del_choice == 'y':
    
    for filename in os.listdir(VALID_DATA_FOLDER):
        if filename.endswith('.JPG'):
            file_path = os.path.join(VALID_DATA_FOLDER, filename)
            os.remove(file_path)
            print(f"Deleted: {filename}")
    
    for filename in os.listdir(OUTPUT_MASKS_DIR):
        if filename.endswith('.png'):
            file_path = os.path.join(OUTPUT_MASKS_DIR, filename)
            os.remove(file_path)
            print(f"Deleted: {filename}")
            
# 25 26 --> 1.8
# 25 27 --> 5.8
# 26 27 --> 4