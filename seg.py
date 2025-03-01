from pydub import AudioSegment
import os

# Function to split the audio file into 30-second segments
def split_audio_to_segments(file_path, output_directory, segment_duration=30):
    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    # Calculate the total duration of the audio in milliseconds
    total_duration = len(audio)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Split audio into segments
    for i in range(0, total_duration, segment_duration * 1000):
        # Extract 30-second segment
        segment = audio[i:i + segment_duration * 1000]
        # Export segment to output directory
        segment_name = f"segment_{i // 1000}_{(i + segment_duration * 1000) // 1000}.wav"
        segment.export(os.path.join(output_directory, segment_name), format="wav")
        print(f"Exported {segment_name}")

# Example usage
file_path = "/Users/samarthshinde/Desktop/Audio_car/htdemucs_ft/1720510309-6662474_ha_4079_cc_m_22_wg_00441_0000-1114_03_0000005/vocals.wav"  # Path to your input audio file
output_directory = "/Users/samarthshinde/Desktop/Audio_car/htdemucs_ft"  # Directory where segments will be saved
split_audio_to_segments(file_path, output_directory)