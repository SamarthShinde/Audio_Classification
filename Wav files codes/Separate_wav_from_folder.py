import os
import shutil

def separate_wav_files(input_folder, output_folder):
    """
    Separate WAV files from a folder containing JSON and CSV files and store them in a desired location.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each file in the input folder
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        # Check if the file is a WAV file
        if file_name.endswith('.wav') and os.path.isfile(file_path):
            # Move the WAV file to the output folder
            shutil.move(file_path, os.path.join(output_folder, file_name))
            print(f"Moved '{file_name}' to '{output_folder}'")

# Example usage
input_folder = r'C:\Users\Admin\Desktop\Audio\UrbanSound\data\air_conditioner'  # Replace with the path to your folder containing JSON, CSV, and WAV files
output_folder = r'C:\Users\Admin\Desktop\Audio_class\air conditioner'  # Replace with the path to your desired location for WAV files
separate_wav_files(input_folder, output_folder)
