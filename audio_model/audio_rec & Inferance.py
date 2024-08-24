import pyaudio
import wave
import os
import numpy as np
import librosa
from keras.models import load_model
import pickle

# Function to extract MFCC features from audio file
def extract_features(file_path):
    waveform, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=64)
    return np.mean(mfccs, axis=1)

# Function to perform inference using the trained model
def predict_class(model, file_path, encoder):
    # Extract MFCC features
    features = extract_features(file_path)
    # Reshape features for model input
    features = features.reshape(1, -1)
    # Perform prediction
    predictions = model.predict(features)
    # Get predicted class
    predicted_class = np.argmax(predictions)
    # Decode class label
    predicted_label = encoder.inverse_transform([predicted_class])[0]
    return predicted_label

# Function to list predictions for all audio files in a directory
def predict_directory(model, directory_path, encoder):
    # Iterate through each file in the directory
    for file_name in os.listdir(directory_path):
        # Check if the file is an audio file
        if file_name.endswith('.wav'):
            # Construct full file path
            file_path = os.path.join(directory_path, file_name)
            # Perform inference for the current file
            predicted_class = predict_class(model, file_path, encoder)
            print("File:", file_name, "Predicted Class:", predicted_class)

# Load the trained model
model_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/H_file/Audio2.h5'
model = load_model(model_path)

# Load label encoder used for encoding classes during training
label_encoder_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Lable_encoder/label_encoder3.pkl'
with open(label_encoder_path, 'rb') as f:
    encoder = pickle.load(f)

# Directory containing audio files for inference
audio_directory_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Audio_record'

def record_and_predict_audio(duration_seconds, filename):
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

    # Perform inference for the recorded audio
    predict_directory(model, os.path.dirname(filename), encoder)

# Define the duration of the recording in seconds
duration_seconds = 7

# Specify the filename to save the recorded audio
filename = "/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Audio_record/test7.wav"

# Record audio for the specified duration and save it to the specified filename
record_and_predict_audio(duration_seconds, filename)
