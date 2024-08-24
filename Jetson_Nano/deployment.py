import numpy as np
from tensorflow.keras.models import load_model
import librosa
import pickle
import os

# Load the saved model
model_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/H_file/Audio7.h5'
model = load_model(model_path)

# Load the label encoder
encoder_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Lable_encoder/label_encoder6.pkl'
with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)

# Function to preprocess audio input
def preprocess_audio(file):
    waveform, sampleRate = librosa.load(file, sr=16000)
    features = librosa.feature.mfcc(y=waveform, sr=sampleRate, n_mfcc=64)
    return np.mean(features, axis=1)

# Function for real-time inference
def predict(audio_file):
    input_data = preprocess_audio(audio_file)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    predictions = model.predict(input_data)
    predicted_class = encoder.inverse_transform(np.argmax(predictions, axis=1))[0]
    return predicted_class

# Example usage with a single audio file
audio_file = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/audiorec/fold2024-06-20/avishrant_driving_on_open_5_154337_segment_6.wav'
predicted_class = predict(audio_file)
print(f"Predicted class for '{audio_file}': {predicted_class}")