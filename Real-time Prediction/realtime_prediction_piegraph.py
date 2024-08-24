import os
import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model
import pickle
import queue
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

# Queue to hold audio data
audio_queue = queue.Queue()

# Callback function to be called by sounddevice
def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

# Set the sample rate and duration of the recording
sample_rate = 22050  # or the sample rate you used during training

# Function to update the pie chart
def update_pie_chart(frame):
    if not audio_queue.empty():
        audio_data = audio_queue.get()
        audio_data = audio_data.flatten()
        probabilities = predict_class(audio_data, sample_rate)

        ax.clear()

        # Plot pie chart
        wedges, texts, autotexts = ax.pie(probabilities, labels=labels, autopct='%1.1f%%', startangle=140)

        # Find the label with the highest probability
        max_index = np.argmax(probabilities)
        max_angle = (sum(probabilities[:max_index]) + probabilities[max_index] / 2) * 360

        # Plot the dot
        ax.plot([0, 0], [0, 0], 'ko')  # Center dot
        ax.plot([0.5 * np.cos(np.radians(max_angle))], [0.5 * np.sin(np.radians(max_angle))], 'ro')  # Dot on pie chart

        # Ensure the aspect ratio is equal to make the pie circular
        ax.axis('equal')

        # Add legend in the top right corner
        ax.legend(wedges, labels, loc='upper right', bbox_to_anchor=(1.3, 1.0))

        # Add a box in the bottom right corner for percentage display
        textstr = '\n'.join([f'{label}: {prob * 100:.2f}%' for label, prob in zip(labels, probabilities)])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.3, -1.0, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)

    return wedges, texts, autotexts

# Create a figure and axis for the pie chart
fig, ax = plt.subplots()

try:
    print("Starting real-time prediction...")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate):
        ani = FuncAnimation(fig, update_pie_chart, blit=False)
        plt.show()
except KeyboardInterrupt:
    print("Stopping real-time prediction.")
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(-1)


"In this real time prediction, make changes such that instead of predicting the sound realtime, first analyse the sound for 3 sec 3 times, then for that analysed 3 sec sound what would be the highest predicted class, would be displayed below the pie chart & in the pie chart it will show the avg percentage of 3 sec analysed sound 3 times & display it over a pie chart, so indirectly the pie chart will have the 3,3,3 sec 3 audio analysed avg percentage displayed"