import pandas as pd
import shutil
import os
from pydub import AudioSegment

# Define the path to the CSV file and the base directory containing the audio files
csv_path = '/Users/samarthshinde/Desktop/Backup/ready_codes/Audio_Classification_ML/to_record/metadata_test.csv'
audio_base_dir = '/Volumes/T7_Shield/Audio/to_record/audiorec'
target_dir = '/Users/samarthshinde/Desktop/mixtures'

# Create the target directory if it does not exist
os.makedirs(target_dir, exist_ok=True)

# Read the CSV file
metadata = pd.read_csv(csv_path)

# Filter the metadata for classID 1 (assuming 1 corresponds to Male and Female classes)
class_9_metadata = metadata[metadata['classID'] == 1]


# Function to check if the audio is mono and convert it to stereo if necessary
def convert_to_stereo(audio):
    if audio.channels == 1:
        return audio.set_channels(2)
    return audio


# Process and copy the audio files to the target directory
for index, row in class_9_metadata.iterrows():
    fold = row['fold']
    file_name = row['slice_file_name']
    source_path = os.path.join(audio_base_dir, f'fold{fold}', file_name)
    target_mp3_path = os.path.join(target_dir, f"{os.path.splitext(file_name)[0]}.mp3")

    # Load the audio file and convert it to MP3
    if os.path.exists(source_path):
        audio = AudioSegment.from_wav(source_path)

        # Convert to stereo if the audio is mono
        audio_stereo = convert_to_stereo(audio)

        # Export the audio to MP3 format
        audio_stereo.export(target_mp3_path, format="mp3")

print(f"Copied and converted {len(class_9_metadata)} files to {target_dir}")