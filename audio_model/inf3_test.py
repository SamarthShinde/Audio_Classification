import os
import numpy as np
import librosa
from keras.models import load_model
import pickle
import pandas as pd

# Function to extract MFCC features from audio file
def extract_features(file_path):
    waveform, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=64)
    return np.mean(mfccs, axis=1)

# Function to perform inference using the trained model
def predict_class(model, file_path, encoder):
    # Extract MFCC features
    features = extract_features(file_path)
    # Reshape features for model input
    features = features.reshape(1, -1)
    # Perform prediction
    predictions = model.predict(features)
    # Get predicted class
    predicted_class = encoder.inverse_transform(np.argsort(predictions)[0][-6:])[::-1]
    # Get probabilities
    probabilities = np.sort(predictions)[0][-6:][::-1]
    return predicted_class, probabilities

# Function to list predictions for all audio files in a directory
def predict_directory(model, directory_path, encoder, metadata_df):
    new_rows = []
    # Iterate through each file in the directory
    for file_name in os.listdir(directory_path):
        # Check if the file is an audio file
        if file_name.endswith('.wav'):
            # Construct full file path
            file_path = os.path.join(directory_path, file_name)
            # Perform inference for the current file
            predicted_classes, predicted_probabilities = predict_class(model, file_path, encoder)
            print(f"File: {file_name}")
            for predicted_class, probability in zip(predicted_classes, predicted_probabilities):
                print(f"Predicted Class: {predicted_class} Probability: {probability:.2f}")
            # Prompt user for confirmation
            user_input = input("Is the predicted class correct? (yes/no): ")
            if user_input.lower() == "no":
                new_class = input("Enter the correct class: ")
            else:
                new_class = predicted_classes[0]  # Assume first prediction as correct
            # Move audio file to the corresponding folder
            if new_class in encoder.classes_:
                folder_index = encoder.transform([new_class])[0]
                if folder_index < 8:
                    folder_name = f"fold{folder_index}"
                else:
                    folder_name = f"fold{folder_index % 8 + 8}"
                new_folder_path = os.path.join(directory_path, folder_name)
                os.makedirs(new_folder_path, exist_ok=True)
                os.rename(file_path, os.path.join(new_folder_path, file_name))
                print("Audio file moved to folder:", folder_name)
                # Append new row to DataFrame
                new_rows.append({'slice_file_name': file_name,
                                 'classID': new_class,
                                 'fold': folder_index})
            else:
                print("Class not found in encoder classes.")
    # Concatenate new rows to the existing metadata DataFrame
    new_metadata_df = pd.concat([metadata_df, pd.DataFrame(new_rows)], ignore_index=True)
    return new_metadata_df

# Load the trained model
model_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/H_file/audio_test2.h5'
model = load_model(model_path)

# Load label encoder used for encoding classes during training
label_encoder_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Lable_encoder/audio_test2.pkl'
with open(label_encoder_path, 'rb') as f:
    encoder = pickle.load(f)

# Load existing metadata CSV file
metadata_csv_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/metadata_test.csv'
metadata_df = pd.read_csv(metadata_csv_path)

# Directory containing audio files for inference
audio_directory_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Audio_record'

# Perform predictions for all audio files in the directory
updated_metadata = predict_directory(model, audio_directory_path, encoder, metadata_df)

# Save updated metadata to CSV
updated_metadata.to_csv(metadata_csv_path, index=False)

print("Inference and metadata update completed.")
