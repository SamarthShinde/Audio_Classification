import os
import numpy as np
import sounddevice as sd
import librosa
import librosa.display
from tensorflow.keras.models import load_model
import pickle
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import torch
from demucs import pretrained
from demucs.apply import apply_model

# Load the trained model
model_path = '/Users/samarthshinde/Desktop/Backup/ready_codes/Audio_Classification_ML/H_file/audio_test7.h5'
model = load_model(model_path)

# Load the label encoder
label_encoder_path = '/Users/samarthshinde/Desktop/Backup/ready_codes/Audio_Classification_ML/Lable_encoder/audio_test7.pkl'
with open(label_encoder_path, 'rb') as f:
    encoder = pickle.load(f)

# Extract labels from encoder
labels = encoder.classes_

# Load the Demucs model for vocal separation
demucs_model = pretrained.get_model('htdemucs')
demucs_model.to('cpu')  # Use 'cuda' if GPU is available

# Function to apply Demucs for vocal separation
def separate_vocals(audio_data, sample_rate):
    # Convert mono to stereo by duplicating the channel
    if len(audio_data.shape) == 1:  # Check if input is mono
        audio_data = np.stack([audio_data, audio_data], axis=0)  # Create stereo input

    # Convert to PyTorch tensor
    audio_tensor = torch.tensor(audio_data).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Apply Demucs
        sources = apply_model(demucs_model, audio_tensor, split=True, overlap=0.25)
    vocals = sources[0, 0].cpu().numpy()
    return vocals

# Function to extract MFCC features from audio data
def extract_features(audio_data, sample_rate):
    if len(audio_data) < 2048:
        audio_data = np.pad(audio_data, (0, max(0, 2048 - len(audio_data))), 'constant')
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=64)
    return np.mean(mfccs, axis=1)

# Function to predict probabilities of classes
def predict_class(audio_data, sample_rate):
    features = extract_features(audio_data, sample_rate)
    features = features.reshape(1, -1)
    probabilities = model.predict(features, verbose=0)
    return probabilities[0]

# Set the sample rate and duration of the recording
sample_rate = 44100  # Demucs requires 44.1 kHz
buffer_size = sample_rate * 5  # 5 seconds buffer
audio_buffer = np.zeros(buffer_size, dtype=np.float32)

# Timing variables
analysis_time = 0
prediction_time = 0
frame_count = 0

# Callback for InputStream to process microphone audio
def audio_callback(indata, frames, time, status):
    global audio_buffer, output_stream
    if status:
        print(status, file=sys.stderr)
    # Update circular buffer
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]

    # Apply Demucs for vocal separation
    vocals = separate_vocals(audio_buffer, sample_rate)

    # Play isolated vocals through the speaker
    output_stream.write(vocals.reshape(-1, 1))

# Function to update the pie chart
def update_pie_chart(frame):
    global audio_buffer, analysis_time, prediction_time, frame_count
    start_time = time.time()

    # Process the audio buffer
    vocals = separate_vocals(audio_buffer, sample_rate)
    probabilities = predict_class(vocals, sample_rate)

    max_index = np.argmax(probabilities)
    predicted_class = labels[max_index]
    confidence = probabilities[max_index]

    analysis_time += (time.time() - start_time)

    ax.clear()

    confidence_threshold = 0.5
    if confidence > confidence_threshold:
        wedges, texts, autotexts = ax.pie(probabilities, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        ax.text(0, -1.2, f'Predicted Class: {predicted_class}', fontsize=14, ha='center')
    else:
        ax.text(0.5, 0.5, 'Confidence too low to display prediction', fontsize=14, ha='center', va='center')

    frame_count += 1

# Create figure and axis for the pie chart
fig, ax = plt.subplots()

def print_timings():
    global analysis_time, prediction_time, frame_count
    if frame_count > 0:
        print(f'Average analysis time per frame: {analysis_time / frame_count:.4f} seconds')
        print(f'Total frames processed: {frame_count}')

try:
    print("Starting real-time vocal separation, prediction, and playback...")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=int(sample_rate * 0.1)) as input_stream, \
         sd.OutputStream(channels=1, samplerate=sample_rate) as output_stream:
        ani = FuncAnimation(fig, update_pie_chart, blit=False)
        plt.show()
except KeyboardInterrupt:
    print("Stopping real-time prediction.")
    print_timings()
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(-1)