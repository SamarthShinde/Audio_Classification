import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model
import pickle
import queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import time

# Load the trained models
engine_rev_model_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Combine_Logic/engine_rev_model.h5'
second_level_model_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Combine_Logic/second_level_model.h5'
engine_rev_model = load_model(engine_rev_model_path)
second_level_model = load_model(second_level_model_path)

# Load the label encoders
engine_rev_encoder_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Combine_Logic/engine_rev_label_encoder.pkl'
second_level_encoder_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Combine_Logic/second_level_label_encoder.pkl'
with open(engine_rev_encoder_path, 'rb') as f:
    engine_rev_encoder = pickle.load(f)
with open(second_level_encoder_path, 'rb') as f:
    second_level_encoder = pickle.load(f)

engine_rev_labels = engine_rev_encoder.classes_
second_level_labels = second_level_encoder.classes_

# Function to extract MFCC features
def extract_features(audio_data, sample_rate):
    if len(audio_data) < 2048:
        audio_data = np.pad(audio_data, (0, max(0, 2048 - len(audio_data))), 'constant')
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=64)
    return np.mean(mfccs, axis=1)

# Function to predict engine_rev
def predict_engine_rev(audio_data, sample_rate):
    features = extract_features(audio_data, sample_rate)
    features = features.reshape(1, -1)
    probabilities = engine_rev_model.predict(features, verbose=0)
    return probabilities[0]

# Function to predict second level
def predict_second_level(audio_data, sample_rate):
    features = extract_features(audio_data, sample_rate)
    features = features.reshape(1, -1)
    probabilities = second_level_model.predict(features, verbose=0)
    return probabilities[0]

# Parameters
sample_rate = 22050
buffer_size = sample_rate * 3  # 3 seconds of audio
audio_buffer = np.zeros(buffer_size, dtype=np.float32)
audio_queue = queue.Queue()

# Buffers for averaging
engine_rev_buffer = []
second_level_buffer = []
last_engine_rev_text = ''  # To store the last Engine Rev prediction text
new_engine_rev_text = ''  # To store the new Engine Rev prediction text
last_update_time = 0  # To track the last update time
start_time = time.time()  # To track the start time for the timer

# Callback function to be called by sounddevice
def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

# Function to update the detection
def update_detection(frame):
    global audio_buffer, engine_rev_buffer, second_level_buffer, last_engine_rev_text, new_engine_rev_text, last_update_time, start_time

    if not audio_queue.empty():
        audio_data = audio_queue.get()
        audio_data = audio_data.flatten()

        # Update the buffer with new audio data
        audio_buffer = np.roll(audio_buffer, -len(audio_data))
        audio_buffer[-len(audio_data):] = audio_data

        # Predict engine_rev
        engine_rev_probabilities = predict_engine_rev(audio_buffer, sample_rate)
        engine_rev_buffer.append(engine_rev_probabilities)

        # If buffer size exceeds 3, remove the oldest entry
        if len(engine_rev_buffer) > 3:
            engine_rev_buffer.pop(0)

        # Average engine_rev predictions
        avg_engine_rev_probabilities = np.mean(engine_rev_buffer, axis=0)
        engine_rev_detected = avg_engine_rev_probabilities[np.argmax(avg_engine_rev_probabilities)] > 0.5
        engine_rev_class = engine_rev_labels[np.argmax(avg_engine_rev_probabilities)]
        new_engine_rev_text = f'New Prediction: {engine_rev_class} ({avg_engine_rev_probabilities[np.argmax(avg_engine_rev_probabilities)]:.2f})'

        ax.clear()
        if engine_rev_detected:
            # If engine_rev is detected, predict second level
            second_level_probabilities = predict_second_level(audio_buffer, sample_rate)
            second_level_buffer.append(second_level_probabilities)

            # If buffer size exceeds 3, remove the oldest entry
            if len(second_level_buffer) > 3:
                second_level_buffer.pop(0)

            # Average second level predictions
            avg_second_level_probabilities = np.mean(second_level_buffer, axis=0)
            second_level_class = second_level_labels[np.argmax(avg_second_level_probabilities)]

            # Plot the second level detection results
            wedges, texts, autotexts = ax.pie(avg_second_level_probabilities, labels=second_level_labels, autopct='%1.1f%%', startangle=140)
            ax.set_title(f'Second Level Detection: {second_level_class}')
        else:
            ax.text(0.5, 0.5, 'Engine Rev Not Detected', fontsize=14, ha='center')

        # Plot the first level detection results every 30 seconds
        current_time = time.time()
        if current_time - last_update_time >= 30:
            last_engine_rev_text = new_engine_rev_text
            last_update_time = current_time

        # Add text box for the last and new Engine Rev predictions
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(-1.5, 0.5, f'1st Level Prediction: {last_engine_rev_text}', fontsize=10, verticalalignment='center', bbox=props)
        ax.text(-1.5, 0.3, f'1st Level Old Prediction: {new_engine_rev_text}', fontsize=10, verticalalignment='center', bbox=props)

        # Add timer text
        elapsed_time = current_time - start_time
        timer_text = f'Timer: {30 - int(elapsed_time % 30)}'
        ax.text(0.5, 1.1, timer_text, fontsize=12, ha='center', transform=ax.transAxes)

        return ax

# Create a figure and axis for the plot
fig, ax = plt.subplots()

try:
    print("Starting hierarchical sound detection...")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=int(sample_rate * 0.1)):
        ani = FuncAnimation(fig, update_detection, blit=False)
        plt.show()
except KeyboardInterrupt:
    print("Stopping hierarchical sound detection.")
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(-1)