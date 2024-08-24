import numpy as np
import pandas as pd
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
import pickle

# Define the paths
metadata_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Combine_Logic/engine_rev_metadata.csv'
audio_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Combine_Logic/audiorec'  # Replace with the path to your audio files
model_save_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Combine_Logic/engine_rev_model.h5'
encoder_save_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Combine_Logic/engine_rev_label_encoder.pkl'

# Function to extract MFCC features
def extract_features(file_path):
    waveform, sample_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=64)
    return np.mean(mfccs, axis=1)

# Load the engine_rev metadata
engine_rev_metadata = pd.read_csv(metadata_path)

# Verify filtering
print("Engine Rev Metadata:")
print(engine_rev_metadata.head())

# Extract features and labels for engine_rev detection
features = []
labels = []

for index, row in engine_rev_metadata.iterrows():
    file_path = os.path.join(audio_path, row['slice_file_name'])
    print(f"Checking file: {file_path}")  # Print the file path being checked
    if os.path.exists(file_path):
        mfccs = extract_features(file_path)
        features.append(mfccs)
        labels.append(row['class'])
    else:
        print(f"File not found: {file_path}")  # Print if the file is not found

# Check if features and labels are extracted correctly
if not features or not labels:
    raise ValueError("No data found for engine_rev detection. Please check your metadata and class names.")

# Encode labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
encoded_labels = to_categorical(encoded_labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(np.array(features), encoded_labels, test_size=0.2, random_state=42)

# Define the model
engine_rev_model = Sequential([
    Dense(1024, activation='relu', input_shape=(64,)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(np.unique(labels)), activation='softmax')
])

# Compile the model
engine_rev_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
engine_rev_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=32)

# Save the model and label encoder
engine_rev_model.save(model_save_path)
with open(encoder_save_path, 'wb') as f:
    pickle.dump(encoder, f)