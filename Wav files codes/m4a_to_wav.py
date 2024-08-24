import os
from pydub import AudioSegment

def convert_m4a_to_wav(input_folder, output_folder):
    """
    Convert M4A audio files to WAV format and save them in the output folder.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each audio file in the input folder
    for audio_file in os.listdir(input_folder):
        if audio_file.endswith('.m4a'):
            input_path = os.path.join(input_folder, audio_file)
            output_path = os.path.join(output_folder, os.path.splitext(audio_file)[0] + '.wav')

            # Load the M4A audio file
            audio = AudioSegment.from_file(input_path, format="m4a")

            # Export the audio to WAV format
            audio.export(output_path, format="wav")

            print(f"Converted '{input_path}' to '{output_path}'")

# Example usage
input_folder = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/no_sound'  # Path to the input folder containing M4A audio files
output_folder = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/untitled folder'  # Path to the output folder to save the WAV files

convert_m4a_to_wav(input_folder, output_folder)