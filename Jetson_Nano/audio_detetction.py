import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import pickle
import soundfile as sf

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

# Function to save audio fragments in the predicted class folder
def save_audio_fragments(audio_path, sample_rate=22050, fragment_duration=10):
    audio_data, sr = librosa.load(audio_path, sr=sample_rate)
    total_duration = len(audio_data) // sample_rate
    fragment_samples = fragment_duration * sample_rate
    fragment_count = total_duration // fragment_duration

    for i in range(fragment_count):
        start_sample = i * fragment_samples
        end_sample = start_sample + fragment_samples
        fragment_data = audio_data[start_sample:end_sample]

        probabilities = predict_class(fragment_data, sample_rate)
        max_index = np.argmax(probabilities)
        predicted_class = labels[max_index]
        confidence = probabilities[max_index]

        output_folder = os.path.join("predicted_fragments", predicted_class)
        os.makedirs(output_folder, exist_ok=True)

        fragment_filename = os.path.join(output_folder, f"fragment_{i + 1}.wav")
        sf.write(fragment_filename, fragment_data, sample_rate)

        print(f"Fragment {i + 1} saved in folder '{predicted_class}' with confidence {confidence:.2f}")

# Example usage
audio_file_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/untitled folder/#ANI Podcast with Smita Prakash  EP-17  Palki Sharma, Managing Editor, Network 18.wav'
save_audio_fragments(audio_file_path)