import os
import numpy as np
import pandas as pd
import librosa
from keras.models import load_model
import pickle
import shutil

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
    predicted_class = np.argmax(predictions)
    # Decode class label
    predicted_label = encoder.inverse_transform([predicted_class])[0]
    return predicted_label

# Function to map class to fold
def get_fold_for_class(predicted_label):
    class_to_fold = {
        'car_horn': 0,
        'air_condition': 1,
        'engine_idling': 2,
        'Car': 5,
        'Motorcycle': 6,
        'Truck': 7,
        'female_angry': 8, 'male_angry': 8,
        'female_fear': 9, 'male_fear': 9,
        'female_happy': 10, 'male_happy': 10,
        'female_neutral': 11, 'male_neutral': 11,
        'female_sad': 12, 'male_sad': 12,
        'female_surprise': 13, 'male_surprise': 13
    }
    return class_to_fold.get(predicted_label, 11)

# Function to list predictions for all audio files in a directory
def predict_directory(model, directory_path, encoder, metadata_df):
    # Iterate through each file in the directory
    for file_name in os.listdir(directory_path):
        # Check if the file is an audio file
        if file_name.endswith('.wav'):
            # Construct full file path
            file_path = os.path.join(directory_path, file_name)
            # Perform inference for the current file
            predicted_label = predict_class(model, file_path, encoder)
            print("File:", file_name, "Predicted Class:", predicted_label)
            is_correct = input("Is the predicted class correct? (yes/no): ").strip().lower()
            if is_correct == 'no':
                predicted_label = input("Enter the correct class: ").strip()

            # Get the fold for the predicted class
            fold = get_fold_for_class(predicted_label)
            # Destination folder
            destination_folder = os.path.join(
                '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Audio_datasets/Audio', f'fold{fold}')
            # Copy the audio file to the correct folder
            destination_path = os.path.join(destination_folder, file_name)
            shutil.copy(file_path, destination_path)
            print(f"Audio file copied to folder: fold{fold}")

            # Update metadata
            fsID = np.random.randint(1, 1000001)
            start = 0
            end = librosa.get_duration(filename=destination_path)
            salience = 1
            classID = list(encoder.classes_).index(predicted_label) + 1

            new_metadata = {
                'slice_file_name': file_name,
                'fsID': fsID,
                'start': start,
                'end': end,
                'salience': salience,
                'fold': fold,
                'classID': classID
            }

            # Append or update metadata
            if file_name in metadata_df['slice_file_name'].values:
                metadata_df.loc[metadata_df['slice_file_name'] == file_name, 'classID'] = classID
            else:
                metadata_df = metadata_df.append(new_metadata, ignore_index=True)

    return metadata_df

# Load the trained model
model_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/H_file/Audio6.h5'
model = load_model(model_path)

# Load label encoder used for encoding classes during training
label_encoder_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Lable_encoder/label_encoder5.pkl'
with open(label_encoder_path, 'rb') as f:
    encoder = pickle.load(f)

# Directory containing audio files for inference
audio_directory_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Audio_record/Test_audioi/New Folder With Items'

# Path to the metadata CSV file
metadata_csv_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Audio_datasets/Urban8k_&_UrbanSound.csv'

# Load or create metadata DataFrame
if os.path.exists(metadata_csv_path):
    metadata_df = pd.read_csv(metadata_csv_path)
else:
    metadata_df = pd.DataFrame(columns=['slice_file_name', 'fsID', 'start', 'end', 'salience', 'fold', 'classID'])

# Perform predictions for all audio files in the directory
updated_metadata = predict_directory(model, audio_directory_path, encoder, metadata_df)

# Save the updated metadata to the CSV file
updated_metadata.to_csv(metadata_csv_path, index=False)

"this code now copies the audio files to the desired folder instead of moving them"
