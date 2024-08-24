import os
import csv
import random
import librosa

def extract_human_voices(audio_folder, existing_csv):
    """
    Extracts Desired audio data from audio files in a folder and appends the results to an existing CSV file.
    """
    # Process each audio file in the folder and append data to the existing CSV file
    with open(existing_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Process each audio file in the folder
        for audio_file in os.listdir(audio_folder):
            if audio_file.endswith('.wav') or audio_file.endswith('.mp3'):
                audio_path = os.path.join(audio_folder, audio_file)
                audio_duration = librosa.get_duration(path=audio_path)

                # Generate a random fsID within the specified range
                fsID = random.randint(300000, 400000)

                # Append new data to the existing CSV file
                writer.writerow([audio_file, fsID, 0, audio_duration, 0, 'music', 5, 'Music','street music from urban sound 8K'])

# Example usage
audio_folder = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/music' # Replace with the path to your audio folder
existing_csv = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/metadata_test.csv'  # Path to the existing CSV file
extract_human_voices(audio_folder, existing_csv)
