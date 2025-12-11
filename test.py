import os
from pathlib import Path

# Define the folder path
folder_path = r"images\image_backgrounds"

# Get all files in the folder
files = sorted(os.listdir(folder_path))

# Counter for renaming
counter = 1

for file in files:
    file_path = os.path.join(folder_path, file)
    
    # Skip if it's a directory
    if os.path.isdir(file_path):
        continue
    
    # Get file extension
    _, ext = os.path.splitext(file)
    
    # Create new filename
    new_name = f"{counter}{ext}"
    new_path = os.path.join(folder_path, new_name)
    
    # Rename the file
    os.rename(file_path, new_path)
    print(f"Renamed: {file} -> {new_name}")
    
    counter += 1

print("Done!")