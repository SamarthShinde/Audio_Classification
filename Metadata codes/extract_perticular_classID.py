import pandas as pd
import shutil
import os

# Define the path to the CSV file and the base directory containing the audio files
csv_path = '/Volumes/DATA_on_server/AUDIO_DATA/UrbanSound8K/metadata/UrbanSound8K.csv'
audio_base_dir = '/Volumes/DATA_on_server/AUDIO_DATA/UrbanSound8K/audio'
target_dir = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/music'

# Create the target directory if it does not exist
os.makedirs(target_dir, exist_ok=True)

# Read the CSV file
metadata = pd.read_csv(csv_path)

# Filter the metadata for classID 9
class_9_metadata = metadata[metadata['classID'] == 9]

# Copy the audio files to the target directory
for index, row in class_9_metadata.iterrows():
    fold = row['fold']
    file_name = row['slice_file_name']
    source_path = os.path.join(audio_base_dir, f'fold{fold}', file_name)
    target_path = os.path.join(target_dir, file_name)

    # Copy the file
    shutil.copy(source_path, target_path)

print(f"Copied {len(class_9_metadata)} files to {target_dir}")