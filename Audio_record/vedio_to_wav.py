import os
from moviepy.editor import VideoFileClip


def extract_audio_from_video(video_file, output_dir):
    """
    Extract audio from a video file and save it as a WAV file in the specified directory with the name of the video folder.

    :param video_file: Path to the input video file.
    :param output_dir: Directory where the output audio file should be saved.
    """
    # Get the directory name of the video file
    video_folder_name = os.path.basename(os.path.dirname(video_file))

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the output audio file path
    audio_file = os.path.join(output_dir, f"{video_folder_name}.wav")

    # Load the video file
    video = VideoFileClip(video_file)

    # Extract the audio
    audio = video.audio

    # Write the audio to a WAV file
    audio.write_audiofile(audio_file, codec='pcm_s16le')


# Example usage
video_file = '/path/to/your/video/input_video.mp4'  # Path to your input video file
output_dir = '/path/to/your/output/directory'  # Directory to save the extracted audio file

extract_audio_from_video(video_file, output_dir)
