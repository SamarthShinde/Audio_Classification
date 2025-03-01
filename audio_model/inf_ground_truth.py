import os
import numpy as np
import librosa
import pandas as pd
from keras.models import load_model
import pickle
from tabulate import tabulate

# Function to extract MFCC features from a segment of the audio
def extract_features_segment(waveform, sample_rate, start_time, duration=5):
    start_sample = int(start_time * sample_rate)
    end_sample = int((start_time + duration) * sample_rate)
    segment = waveform[start_sample:end_sample]
    if len(segment) == 0:
        return None
    mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=64)
    return np.mean(mfccs, axis=1)

# Function to perform inference on each 5-second segment and return confidence scores
def predict_class_segment(model, features, encoder):
    features = features.reshape(1, -1)
    predictions = model.predict(features)[0]  # Get the probability for each class
    predicted_class = np.argmax(predictions)
    predicted_label = encoder.inverse_transform([predicted_class])[0]
    class_confidences = {encoder.inverse_transform([i])[0]: f"{pred * 100:.2f}%" for i, pred in enumerate(predictions)}
    return predicted_label, class_confidences

# Main function to process each audio file and match ground truth
def process_audio_files(model, audio_dir, encoder, ground_truth_csv):
    # Load ground truth data
    ground_truth_df = pd.read_csv(ground_truth_csv)

    results = []
    for audio_file in os.listdir(audio_dir):
        if not audio_file.endswith(".wav"):
            continue
        file_path = os.path.join(audio_dir, audio_file)
        waveform, sample_rate = librosa.load(file_path, sr=None)
        total_duration = librosa.get_duration(y=waveform, sr=sample_rate)

        # Iterate through 5-second segments
        for start_time in range(0, int(total_duration), 5):
            features = extract_features_segment(waveform, sample_rate, start_time)
            if features is None:
                continue

            predicted_label, class_confidences = predict_class_segment(model, features, encoder)

            # Search for ground truth in CSV based on audio name and segment time
            audio_name = os.path.basename(audio_file)
            segment_df = ground_truth_df[(ground_truth_df['Audio Name'] == audio_name) &
                                         (ground_truth_df['Segment Time'] == f"{start_time}-{start_time + 5}")]

            if not segment_df.empty:
                ground_truth_labels = segment_df['Class Name'].values[0]
                # Append to results
                results.append([
                    audio_name,  # Audio name
                    start_time,  # Segment start
                    f"{start_time}-{start_time + 5} sec",  # Time frame
                    predicted_label,  # Predicted class
                    ground_truth_labels,  # Ground truth
                    class_confidences  # Confidence for each class
                ])

    # Save results to CSV
    result_df = pd.DataFrame(results, columns=['Audio Name', 'Segment', 'Time Frame', 'Predicted Inference', 'Ground Truth Inference', 'Class Confidence'])
    result_df.to_csv(os.path.join(audio_dir, "inference_results_with_confidence.csv"), index=False)
    print("Inference results saved as 'inference_results_with_confidence.csv'")

# Load the trained model and encoder
model_path = '/Users/samarthshinde/Desktop/Backup/ready_codes/Audio_Classification_ML/H_file/audio_test7.h5'
model = load_model(model_path)

label_encoder_path = '/Users/samarthshinde/Desktop/Backup/ready_codes/Audio_Classification_ML/Lable_encoder/audio_test7.pkl'
with open(label_encoder_path, 'rb') as f:
    encoder = pickle.load(f)

# Directory containing audio files and path to ground truth CSV
audio_dir = '/Users/samarthshinde/Desktop/Audio_drive/2024-07-11'
ground_truth_csv = '/Users/samarthshinde/Desktop/CSV/combined_file.csv'

# Process audio files for inference
process_audio_files(model, audio_dir, encoder, ground_truth_csv)