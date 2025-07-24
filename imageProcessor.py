import os
import shutil
import time
import wmi

DESTINATION_FOLDER = "C:/Users/pbsil/Drone-IQ/droneData" # Change if needed

def get_mounted_drives():
    """Returns a list of mounted drive letters on Windows. (ex. D:\\ or G:\\)"""
    c = wmi.WMI()
    drives = []
    # Check for each folder/path in Windows OS
    for drive in c.Win32_LogicalDisk():
        if drive.DriveType == 2: # Removable Disk
            drives.append(drive.DeviceID + "\\")
    return drives

def process_sd_card(sd_path):
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
                    # Helps find if this is the actual fodlder where images are stored
                    if any(f.lower().endswith(('.jpg', '.jpeg')) for f in os.listdir(full_sub_dir_path)):
                        drone_photo_path = full_sub_dir_path
                        print(f"Identified drone media folder: {drone_photo_path}")
                        break # Found the DCIM folder, no need to search more
            if drone_photo_path:
                break # Found the main DCIM path, stop os.walk

    if not drone_photo_path:
        print("Could not find a recognized drone photo/video directory (e.g., DCIM/100MEDIA) on SD card.")
        return

    # Ensure destination folder exists
    if not os.path.exists(DESTINATION_FOLDER):
        os.makedirs(DESTINATION_FOLDER)
        print(f"Created destination folder: {DESTINATION_FOLDER}")

    copied_count = 0
    # Iterate through all files in the identified drone photo path
    for filename in os.listdir(drone_photo_path):
        source_file = os.path.join(drone_photo_path, filename)
        destination_file = os.path.join(DESTINATION_FOLDER, filename)

        # Check if it is an acceptable extension
        if os.path.isfile(source_file) and filename.lower().endswith(('.jpg', '.jpeg')): 
            try:
                # Option 1: Overwrite existing files (Simplest)
                if os.path.exists(destination_file):
                    print(f"Overwriting existing file: {filename}")
                shutil.copy2(source_file, destination_file)
                print(f"Copied: {filename}")
                copied_count += 1

            except Exception as e:
                print(f"Error copying {filename}: {e}")

    if copied_count == 0:
        print("No media files found or copied from the SD card.")
    else:
        print(f"Finished copying from SD card. Copied {copied_count} files.")


if __name__ == "__main__":
    print("Monitoring for SD card insertion...")
    known_drives = get_mounted_drives() # Get initial drives
    print(f"Currently connected drives: {known_drives}")

    while True:
        current_drives = get_mounted_drives()
        new_drives = [d for d in current_drives if d not in known_drives]

        for new_drive_path in new_drives:
            # Short cooldown for OS to recuperate
            print(f"New drive detected: {new_drive_path}")
            time.sleep(2)
            process_sd_card(new_drive_path)

        # Update known_drives to reflect the current state
        known_drives = current_drives

        time.sleep(5) # Update drivers every 5 seconds