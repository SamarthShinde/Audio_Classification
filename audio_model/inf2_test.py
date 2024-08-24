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
def predict_class(model, file_path, encoder, top_k=6):
    # Extract MFCC features
    features = extract_features(file_path)
    # Reshape features for model input
    features = features.reshape(1, -1)
    # Perform prediction
    probabilities = model.predict(features)
    # Get top k predicted classes and their probabilities
    top_k_indices = np.argsort(probabilities[0])[::-1][:top_k]
    predicted_classes = encoder.inverse_transform(top_k_indices)
    predicted_probabilities = probabilities[0][top_k_indices]
    # Handle previously unseen labels
    predicted_classes_filtered = []
    for cls in predicted_classes:
        if cls in encoder.classes_:
            predicted_classes_filtered.append(cls)
        else:
            predicted_classes_filtered.append('Unknown')
    return predicted_classes_filtered, predicted_probabilities


# Function to list predictions for all audio files in a directory
def predict_directory(model, directory_path, encoder, top_k=6):
    # Iterate through each file in the directory
    for file_name in os.listdir(directory_path):
        # Check if the file is an audio file
        if file_name.endswith('.wav'):
            # Construct full file path
            file_path = os.path.join(directory_path, file_name)
            # Perform inference for the current file
            predicted_classes, predicted_probabilities = predict_class(model, file_path, encoder, top_k)
            print("File:", file_name)
            for cls, prob in zip(predicted_classes, predicted_probabilities):
                print("Predicted Class:", cls, "Probability:", prob)
            print()

# Load the trained model
model_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/H_file/audio_test6.h5'
model = load_model(model_path)

# Load label encoder used for encoding classes during training
label_encoder_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Lable_encoder/audio_test6.pkl'
with open(label_encoder_path, 'rb') as f:
    encoder = pickle.load(f)

# Directory containing audio files for inference
audio_directory_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/2024-07-22/male'

# Perform predictions for all audio files in the directory
predict_directory(model, audio_directory_path, encoder, top_k=5)
