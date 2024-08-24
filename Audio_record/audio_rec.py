import pyaudio
import wave

def record_audio(duration_seconds, filename):
    # Set parameters for recording
    chunk = 1024  # Number of frames per buffer
    sample_format = pyaudio.paInt16  # 16-bit resolution
    channels = 1  # Mono
    fs = 44100  # Sample rate
    seconds = duration_seconds  # Duration of recording

    # Initialize PyAudio object
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    print("Recording...")

    frames = []

    # Record audio in chunks
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PyAudio object
    p.terminate()

    print("Finished recording.")

    # Save the recorded audio as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio recorded successfully and saved as '{filename}'")

# Define the duration of the recording in seconds
duration_seconds = 7

# Specify the filename to save the recorded audio
filename = ("utpal.wav")

# Record audio for the specified duration and save it to the specified filename
record_audio(duration_seconds, filename)
