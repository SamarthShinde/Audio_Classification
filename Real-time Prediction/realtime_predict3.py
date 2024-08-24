import os
import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model
import pickle
import queue
import sys
import pandas as pd
from tabulate import tabulate

# Load the trained model
model_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/H_file/audio_test4.h5'
model = load_model(model_path)

# Load the label encoder
label_encoder_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Lable_encoder/audio_test4.pkl'
with open(label_encoder_path, 'rb') as f:
    encoder = pickle.load(f)

# Extract labels from encoder
labels = encoder.classes_

# Initialize table with labels and empty probabilities
df = pd.DataFrame(columns=labels)
df.loc[0] = [0] * len(labels)


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

try:
    print("Starting real-time prediction...")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate):
        while True:
            if not audio_queue.empty():
                audio_data = audio_queue.get()
                audio_data = audio_data.flatten()
                probabilities = predict_class(audio_data, sample_rate)

                # Reset the dataframe row to 0
                df.loc[0] = [0] * len(labels)

                # Update the dataframe with the new probabilities in percentage
                for i, prob in enumerate(probabilities):
                    df.loc[0, labels[i]] = f"{prob * 100:.2f}%"

                # Highlight the column with the highest probability
                max_index = np.argmax(probabilities)
                max_label = labels[max_index]

                # Create a new DataFrame for display with highlighting
                display_df = df.copy()
                display_df[max_label] = display_df[max_label].apply(lambda x: f"\033[1;31m{x}\033[0m")

                # Clear the console and print the updated table
                print("\033c", end="")  # Clear the console

                print("Class Predictions Table\n")
                print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
except KeyboardInterrupt:
    print("Stopping real-time prediction.")
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(-1)