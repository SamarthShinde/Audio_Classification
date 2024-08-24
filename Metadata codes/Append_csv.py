import os
import csv
import random

def append_to_existing_csv(input_csv, existing_csv, classID, class_name, audio_folders):
    """
    Append information from the input CSV to the existing CSV file.
    """
    with open(existing_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        with open(input_csv, 'r', newline='') as inputfile:
            reader = csv.DictReader(inputfile)
            for row in reader:
                folder = row['Folder']
                audio_name = row['Path']
                audio_path = find_audio_path(audio_folders, audio_name)
                if audio_path:
                    fsID = random.randint(2000, 30000)
                    start_time, end_time = get_audio_duration(audio_path)
                    writer.writerow([audio_name, fsID, start_time, end_time, 1, folder, classID, class_name])

def find_audio_path(audio_folders, audio_name):
    """
    Find the full path of the audio file based on the folders and audio file name.
    """
    for folder in audio_folders:
        for root, dirs, files in os.walk(folder):
            if audio_name in files:
                return os.path.join(root, audio_name)
    return None

def get_audio_duration(audio_path):
    """
    Get the start and end time of the audio file.
    """
    # Replace this with your implementation to get the duration of the audio file
    # For example, you can use librosa or other libraries to get the audio duration
    start_time = 0  # Placeholder value for start time
    end_time = 10  # Placeholder value for end time
    return start_time, end_time

# Example usage
input_csv = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Metadata codes/metadata/male_surprise.csv'  # Path to the input CSV containing class, folder, and audio file names
existing_csv = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Audio_datasets/metadata.csv'  # Path to the existing CSV file
classID = 1 # Set the class ID
class_name = 'Male'  # Set the class name
audio_folders = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Speech_emotion_rcog/Crema'  # Paths to the audio folders

append_to_existing_csv(input_csv, existing_csv, classID, class_name, audio_folders)
