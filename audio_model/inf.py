import os
import numpy as np
import librosa
import pandas as pd
from keras.models import load_model
import pickle
from tabulate import tabulate  # For displaying results in a formatted table


# Function to extract MFCC features from a segment of the audio
def extract_features_segment(waveform, sample_rate, start_time, duration=5):
    # Extract a segment of the waveform
    start_sample = int(start_time * sample_rate)
    end_sample = int((start_time + duration) * sample_rate)
    segment = waveform[start_sample:end_sample]

    # Check if the segment has enough samples
    if len(segment) == 0:
        return None

    # Extract MFCC features from the segment
    mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=64)
    return np.mean(mfccs, axis=1)


# Function to perform inference on each 5-second segment
def predict_class_segment(model, features, encoder):
    # Reshape features for model input
    features = features.reshape(1, -1)
    # Perform prediction
    predictions = model.predict(features)
    # Get predicted class probabilities
    predicted_class = np.argmax(predictions)
    predicted_label = encoder.inverse_transform([predicted_class])[0]
    # Return class label and probabilities
    return predicted_label, predictions[0]


# Function to process an entire audio file in 5-second segments
def predict_audio_by_segments(model, file_path, encoder, duration=5):
    # Load the audio file
    waveform, sample_rate = librosa.load(file_path, sr=None)
    total_duration = librosa.get_duration(y=waveform, sr=sample_rate)

    # Prepare the table for storing results
    results = []

    # Iterate through the audio in 5-second segments
    for start_time in range(0, int(total_duration), duration):
        features = extract_features_segment(waveform, sample_rate, start_time, duration)

        if features is None:
            continue

        # Perform inference on the segment
        predicted_label, predicted_probabilities = predict_class_segment(model, features, encoder)

        # Append the results to the list
        results.append([
            os.path.basename(file_path),  # Audio name
            f'{start_time}-{start_time + duration} sec',  # Time frame
            predicted_label,  # Predicted class
            f'{predicted_probabilities[0] * 100:.2f}%',  # Engine_rev %
            f'{predicted_probabilities[1] * 100:.2f}%',  # Female %
            f'{predicted_probabilities[2] * 100:.2f}%',  # Male %
            f'{predicted_probabilities[3] * 100:.2f}%',  # Music %
            f'{predicted_probabilities[4] * 100:.2f}%'  # No_sound %
        ])

    # Convert results to DataFrame for better visualization
    return results


# Load the trained model
model_path = '/Users/samarthshinde/Desktop/Backup/ready_codes/Audio_Classification_ML/H_file/audio_test7.h5'
model = load_model(model_path)

# Load label encoder used for encoding classes during training
label_encoder_path = '/Users/samarthshinde/Desktop/Backup/ready_codes/Audio_Classification_ML/Lable_encoder/audio_test7.pkl'
with open(label_encoder_path, 'rb') as f:
    encoder = pickle.load(f)

# Directory containing the audio file for inference
audio_file_path = '/Users/samarthshinde/Desktop/Audio_car/htdemucs_ft/1720510309-6662474_ha_4079_cc_m_22_wg_00441_0000-1114_03_0000005/vocals.wav'

# Perform predictions for the entire audio file in 5-second segments
prediction_results = predict_audio_by_segments(model, audio_file_path, encoder)

# Display the prediction table with formatting
table_headers = ['Audio Name', 'Time Frame', 'Predicted Class', 'Engine_rev %', 'Female %', 'Male %', 'Music %', 'No_sound %']
print(tabulate(prediction_results, headers=table_headers, tablefmt="grid"))

# Optionally save the table to a CSV file
prediction_df = pd.DataFrame(prediction_results, columns=table_headers)
prediction_df.to_csv('/Users/samarthshinde/Desktop/audio_predictions.csv', index=False)