import os
from pydub import AudioSegment
import shutil

def convert_m4a_to_wav(input_folder, output_folder):
    """
    Convert M4A audio files to WAV format and copy existing WAV files to the output folder.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each audio file in the input folder
    for audio_file in os.listdir(input_folder):
        input_path = os.path.join(input_folder, audio_file)
        output_path = os.path.join(output_folder, os.path.splitext(audio_file)[0] + '.wav')

        # Check if the file is an M4A file
        if audio_file.endswith('.m4a'):
            # Load the M4A audio file
            audio = AudioSegment.from_file(input_path, format="m4a")

            # Export the audio to WAV format
            audio.export(output_path, format="wav")

            print(f"Converted '{input_path}' to '{output_path}'")
        # Check if the file is a WAV file and copy it to the output folder
        elif audio_file.endswith('.wav') and os.path.isfile(input_path):
            shutil.copy(input_path, output_folder)
            print(f"Copied '{input_path}' to '{output_folder}'")

# Example usage
input_folder = r'C:\Users\Admin\Desktop\Audio_class\7'  # Path to the input folder containing M4A and WAV audio files
output_folder = r"C:\Users\Admin\Desktop\Audio_class\7_new"  # Path to the output folder to save the WAV files

convert_m4a_to_wav(input_folder, output_folder)
