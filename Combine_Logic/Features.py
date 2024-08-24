import pandas as pd
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
import pickle

# Load the metadata
data = pd.read_csv('/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/metadata_test.csv')

# Define paths
audio_path = '/Volumes/T7_Shield/Audio/to_record/audiorec'  # Replace with the path to your audio files

# Function to extract MFCC features from an audio file
def extract_features(file_path):
    waveform, sample_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=64)
    return np.mean(mfccs, axis=1)

# Extract features and labels
features = []
labels = []

for index, row in data.iterrows():
    file_path = os.path.join(audio_path, row['slice_file_name'])
    if os.path.exists(file_path):
        mfccs = extract_features(file_path)
        features.append(mfccs)
        labels.append(row['class'])

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Save the processed data for further use
np.save('features.npy', features)
np.save('labels.npy', labels)