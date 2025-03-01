import os
import subprocess
import wave
import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model
import pickle
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Paths for model and encoder
model_path = '/Users/samarthshinde/Desktop/Backup/ready_codes/Audio_Classification_ML/H_file/audio_test7.h5'
label_encoder_path = '/Users/samarthshinde/Desktop/Backup/ready_codes/Audio_Classification_ML/Lable_encoder/audio_test7.pkl'

# Load the trained model and label encoder
model = load_model(model_path)
with open(label_encoder_path, 'rb') as f:
    encoder = pickle.load(f)

labels = encoder.classes_

# Function to extract MFCC features from audio
def extract_features(audio_data, sample_rate):
    if len(audio_data) < 2048:
        audio_data = np.pad(audio_data, (0, max(0, 2048 - len(audio_data))), 'constant')
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=64)
    return np.mean(mfccs, axis=1)

# Function to predict class probabilities
def predict_class(audio_data, sample_rate):
    features = extract_features(audio_data, sample_rate)
    features = features.reshape(1, -1)
    probabilities = model.predict(features, verbose=0)
    return probabilities[0]

# Setup audio recording parameters
sample_rate = 22050
buffer_size = sample_rate * 6
audio_buffer = np.zeros(buffer_size, dtype=np.float32)
output_dir = "recordings"
os.makedirs(output_dir, exist_ok=True)
chunk_index = 0
chunk_duration = 40
chunk_size = sample_rate * chunk_duration
chunk_buffer = np.zeros(chunk_size, dtype=np.float32)

def audio_callback(indata, frames, time, status):
    global audio_buffer, chunk_buffer, chunk_index
    if status:
        print(status, file=sys.stderr)
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]
    chunk_buffer = np.roll(chunk_buffer, -frames)
    chunk_buffer[-frames:] = indata[:, 0]
    if np.all(chunk_buffer != 0):
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_index}.wav")
        with wave.open(chunk_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((chunk_buffer * 32767).astype(np.int16).tobytes())
        print(f"Saved chunk: {chunk_path}")
        chunk_index += 1
        chunk_buffer[:] = 0

# Function to activate Conda and run Demucs
def process_with_demucs():
    print("Activating Conda environment and processing audio...")
    for file in os.listdir(output_dir):
        if file.endswith(".wav"):
            file_path = os.path.join(output_dir, file)
            command = f"conda activate demucs_env && demucs -d cpu {file_path}"
            try:
                subprocess.run(command, shell=True, check=True)
                print(f"Processed file: {file_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {file}: {e}")

# Real-time prediction and audio recording
try:
    print("Starting real-time prediction and recording...")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=int(sample_rate * 0.1)):
        ani = FuncAnimation(plt.figure(), lambda frame: None, blit=False)
        plt.show()
except KeyboardInterrupt:
    print("Stopping real-time prediction and recording.")
    process_with_demucs()
except Exception as e:
    print(f"An error occurred: {e}")
    process_with_demucs()
    sys.exit(-1)