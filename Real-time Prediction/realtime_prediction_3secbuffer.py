import os
import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model
import pickle
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the trained model
model_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/H_file/audio_test6.h5'
model = load_model(model_path)

# Load the label encoder
label_encoder_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Lable_encoder/audio_test6.pkl'
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

# Circular buffer to hold audio data
buffer_size = sample_rate * 3  # 3 seconds of audio
audio_buffer = np.zeros(buffer_size, dtype=np.float32)

# Callback function to be called by sounddevice
def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(status, file=sys.stderr)
    # Update the circular buffer with new audio data
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]

# Function to update the pie chart
def update_pie_chart(frame):
    global audio_buffer
    # Collect 3 segments of 3 seconds each
    segments = []
    for i in range(3):
        segment_start = int(i * buffer_size / 3)
        segment_end = segment_start + buffer_size // 3
        audio_data = audio_buffer[segment_start:segment_end]
        segments.append(predict_class(audio_data, sample_rate))

    # Average the probabilities
    avg_probabilities = np.mean(segments, axis=0)
    max_index = np.argmax(avg_probabilities)
    predicted_class = labels[max_index]

    ax.clear()

    # Plot pie chart
    wedges, texts, autotexts = ax.pie(avg_probabilities, labels=labels, autopct='%1.1f%%', startangle=140)

    # Plot the dot for the class with highest probability
    max_angle = (sum(avg_probabilities[:max_index]) + avg_probabilities[max_index] / 2) * 360
    ax.plot([0, 0], [0, 0], 'ko')  # Center dot
    ax.plot([0.5 * np.cos(np.radians(max_angle))], [0.5 * np.sin(np.radians(max_angle))], 'ro')  # Dot on pie chart

    # Ensure the aspect ratio is equal to make the pie circular
    ax.axis('equal')

    # Add legend in the top right corner
    ax.legend(wedges, labels, loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # Add a box in the bottom right corner for percentage display
    textstr = '\n'.join([f'{label}: {prob * 100:.2f}%' for label, prob in zip(labels, avg_probabilities)])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(1.3, -1.0, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

    # Display the highest predicted class below the pie chart
    ax.text(0, -1.2, f'Predicted Class: {predicted_class}', fontsize=14, ha='center')

    return wedges, texts, autotexts

# Create a figure and axis for the pie chart
fig, ax = plt.subplots()

try:
    print("Starting real-time prediction...")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=int(sample_rate * 0.1)):
        ani = FuncAnimation(fig, update_pie_chart, blit=False)
        plt.show()
except KeyboardInterrupt:
    print("Stopping real-time prediction.")
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(-1)