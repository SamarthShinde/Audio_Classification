import os
import wave
import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model
import pickle
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Load the trained model
model_path = '/Users/samarthshinde/Desktop/Backup/ready_codes/Audio_Classification_ML/H_file/audio_test7.h5'
model = load_model(model_path)

# Load the label encoder
label_encoder_path = '/Users/samarthshinde/Desktop/Backup/ready_codes/Audio_Classification_ML/Lable_encoder/audio_test7.pkl'
with open(label_encoder_path, 'rb') as f:
    encoder = pickle.load(f)

# Extract labels from encoder
labels = encoder.classes_

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
sample_rate = 22050  # or the sample rate you used during training
buffer_size = sample_rate * 6  # 6 seconds of audio to enable overlapping segments
audio_buffer = np.zeros(buffer_size, dtype=np.float32)
overlap_size = sample_rate * 3  # 3 seconds overlap

# Audio recording and chunk setup
output_dir = "recordings"
os.makedirs(output_dir, exist_ok=True)
chunk_duration = 40  # 40 seconds
chunk_size = sample_rate * chunk_duration
chunk_buffer = np.zeros(chunk_size, dtype=np.float32)
chunk_index = 0

# Callback function for sounddevice
def audio_callback(indata, frames, time, status):
    global audio_buffer, chunk_buffer, chunk_index
    if status:
        print(status, file=sys.stderr)

    # Update circular buffer
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]

    # Update chunk buffer
    chunk_buffer = np.roll(chunk_buffer, -frames)
    chunk_buffer[-frames:] = indata[:, 0]

    # Save the chunk if it's full
    if np.all(chunk_buffer != 0):  # Check if the buffer is filled
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_index}.wav")
        with wave.open(chunk_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((chunk_buffer * 32767).astype(np.int16).tobytes())
        print(f"Saved chunk: {chunk_path}")
        chunk_index += 1
        chunk_buffer[:] = 0  # Reset chunk buffer

# Function to update the pie chart
def update_pie_chart(frame):
    global audio_buffer
    segments = []
    for i in range(0, buffer_size - overlap_size + 1, overlap_size):
        segment_start = i
        segment_end = segment_start + overlap_size
        audio_data = audio_buffer[segment_start:segment_end]
        segments.append(predict_class(audio_data, sample_rate))

    avg_probabilities = np.mean(segments, axis=0)
    max_index = np.argmax(avg_probabilities)
    predicted_class = labels[max_index]
    confidence = avg_probabilities[max_index]

    ax.clear()
    confidence_threshold = 0.5
    if confidence > confidence_threshold:
        wedges, texts, autotexts = ax.pie(avg_probabilities, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        ax.legend(wedges, labels, loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.text(0, -1.2, f'Predicted Class: {predicted_class}', fontsize=14, ha='center')
    else:
        ax.text(0.5, 0.5, 'Confidence too low to display prediction', fontsize=14, ha='center', va='center')

# Create a figure and axis for the pie chart
fig, ax = plt.subplots()

try:
    print("Starting real-time prediction and recording...")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=int(sample_rate * 0.1)):
        ani = FuncAnimation(fig, update_pie_chart, blit=False)
        plt.show()
except KeyboardInterrupt:
    print("Stopping real-time prediction and recording.")
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(-1)