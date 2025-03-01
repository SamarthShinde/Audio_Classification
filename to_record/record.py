import pyaudio
from pydub import AudioSegment
import time
import os
import io

# Parameters for audio recording
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Stereo audio
RATE = 44100  # Sample rate
CHUNK = 1024  # Chunk size for buffering
RECORD_SECONDS = 20  # 2-minute recording

# Desired output folder to save audio files
output_folder = "/Users/samarthshinde/Desktop/instruments"

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize PyAudio
audio = pyaudio.PyAudio()


def record_audio(output_folder, file_count):
    # Initialize audio stream
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)

    print(f"Recording audio fragment {file_count} for {RECORD_SECONDS // 60} minute(s)...")
    frames = []

    # Record the audio in chunks
    try:
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
    except Exception as e:
        print(f"Error during recording: {e}")
        return None

    # Combine all frames into a single byte stream
    audio_data = b''.join(frames)

    # Convert the raw audio data to stereo and export as MP3
    audio_segment = AudioSegment(
        data=audio_data,
        sample_width=audio.get_sample_size(FORMAT),
        frame_rate=RATE,
        channels=CHANNELS
    )

    # Save the recorded audio as MP3
    mp3_output_filename = os.path.join(output_folder, f"audio_fragment_{file_count}.wav")
    try:
        audio_segment.export(mp3_output_filename, format="wav")
        print(f"Saved: {mp3_output_filename}")
    except Exception as e:
        print(f"Error saving MP3: {e}")


def main():
    file_count = 1  # Counter for saved files

    try:
        while True:
            # Record 2-minute audio and save it
            record_audio(output_folder, file_count)
            file_count += 1  # Increment the file count
            time.sleep(1)  # Small delay between recordings (optional)
    except KeyboardInterrupt:
        # Stop recording on keyboard interrupt
        print("\nRecording stopped by user.")
    finally:
        # Clean up the stream and PyAudio
        audio.terminate()


if __name__ == "__main__":
    main()