import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import warnings
import random

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# -----------------------------
# Configuration Parameters
# -----------------------------
CSV_FILE = '/Users/samarthshinde/Desktop/Backup/ready_codes/Audio_Classification_ML/to_record/metadata_test.csv'          # Path to your CSV file
AUDIO_DIR = '/Volumes/T7_Shield/Audio/to_record/audiorec'             # Root directory of audio files
SAMPLE_RATE = 16000                    # Sampling rate for audio files
N_MFCC = 40                            # Number of MFCC features
TEST_SIZE = 0.2                        # Proportion of dataset for testing
RANDOM_STATE = 42                      # Random state for reproducibility
EPOCHS_SPEECH = 50                     # Number of epochs for speech detection
EPOCHS_GENDER = 50                     # Number of epochs for gender classification
BATCH_SIZE = 32                        # Batch size for training

# -----------------------------
# 1. Load and Organize the Data
# -----------------------------
print("Loading CSV file...")
data = pd.read_csv(CSV_FILE)

# Display first few rows
print("\nFirst few rows of the dataset:")
print(data.head())

# -----------------------------
# 2. Data Filtering and Labeling
# -----------------------------
print("\nFiltering data for noisy conditions (salience == 1)...")
noisy_data = data[data['salience'] == 0].reset_index(drop=True)

# Count the number of speech samples in noisy_data
num_speech_noisy = noisy_data[noisy_data['classID'].isin([1, 2])].shape[0]
num_total_noisy = noisy_data.shape[0]
print(f"\nNumber of speech samples in noisy conditions: {num_speech_noisy} out of {num_total_noisy}")

# Create speech detection labels
# 1: Speech (Male or Female), 0: Non-Speech (Engine_rev, No_sound, Music)
noisy_data['speech_label'] = noisy_data['classID'].apply(lambda x: 1 if x in [1, 2] else 0)

# -----------------------------
# 3. Feature Extraction
# -----------------------------
print("\nExtracting features from audio files...")
features = []
speech_labels = []
gender_labels = []

for idx, row in tqdm(noisy_data.iterrows(), total=noisy_data.shape[0]):
    file_path = os.path.join(AUDIO_DIR, row['fold'], row['slice_file_name'])
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfccs_mean = np.mean(mfccs.T, axis=0)  # Mean over time frames

        # Append features and labels
        features.append(mfccs_mean)
        speech_labels.append(row['speech_label'])

        # For gender labels, set to 1 (Female) or 0 (Male), None for non-speech
        if row['classID'] == 1:
            gender_labels.append(0)  # Male
        elif row['classID'] == 2:
            gender_labels.append(1)  # Female
        else:
            gender_labels.append(None)  # Non-speech
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Convert lists to NumPy arrays
X = np.array(features)
y_speech = np.array(speech_labels)
y_gender = np.array(gender_labels)

# -----------------------------
# 4. Prepare Data for Gender Classification
# -----------------------------
# Indices where gender labels are not None (i.e., speech samples)
speech_indices = [i for i, label in enumerate(y_gender) if label is not None]

print(f"\nNumber of speech samples for gender classification: {len(speech_indices)}")

# Prepare data for gender classification
X_gender = X[speech_indices]
y_gender = y_gender[speech_indices]

# Encode gender labels (0: Male, 1: Female)
le_gender = LabelEncoder()
y_gender_encoded = le_gender.fit_transform(y_gender)

# -----------------------------
# 5. Split Data into Training and Testing Sets
# -----------------------------
print("\nSplitting data into training and testing sets...")

# For Speech Detection
X_train_speech, X_test_speech, y_train_speech, y_test_speech = train_test_split(
    X, y_speech, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_speech)

# For Gender Classification
if len(X_gender) > 0:
    X_train_gender, X_test_gender, y_train_gender, y_test_gender = train_test_split(
        X_gender, y_gender_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_gender_encoded)
else:
    print("\nNo speech samples available in noisy conditions for gender classification.")
    X_train_gender, X_test_gender, y_train_gender, y_test_gender = None, None, None, None

# -----------------------------
# 6. Build and Train Speech Detection Model
# -----------------------------
print("\nBuilding and training Speech Detection model...")

speech_model = Sequential([
    Dense(256, input_shape=(N_MFCC,), activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

speech_model.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

speech_model.summary()

# Train the model
speech_history = speech_model.fit(
    X_train_speech, y_train_speech,
    epochs=EPOCHS_SPEECH,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_speech, y_test_speech),
    verbose=1
)

# -----------------------------
# 7. Evaluate Speech Detection Model
# -----------------------------
print("\nEvaluating Speech Detection model...")
speech_loss, speech_acc = speech_model.evaluate(X_test_speech, y_test_speech, verbose=0)
print(f"Speech Detection Accuracy: {speech_acc * 100:.2f}%")

# Classification Report
y_pred_speech = (speech_model.predict(X_test_speech) > 0.5).astype("int32")
print("\nSpeech Detection Classification Report:")
print(classification_report(y_test_speech, y_pred_speech))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test_speech, y_pred_speech))

# -----------------------------
# 8. Build and Train Gender Classification Model
# -----------------------------
if X_train_gender is not None:
    print("\nBuilding and training Gender Classification model...")

    gender_model = Sequential([
        Dense(256, input_shape=(N_MFCC,), activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    gender_model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])

    gender_model.summary()

    # Train the model
    gender_history = gender_model.fit(
        X_train_gender, y_train_gender,
        epochs=EPOCHS_GENDER,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_gender, y_test_gender),
        verbose=1
    )

    # -----------------------------
    # 9. Evaluate Gender Classification Model
    # -----------------------------
    print("\nEvaluating Gender Classification model...")
    gender_loss, gender_acc = gender_model.evaluate(X_test_gender, y_test_gender, verbose=0)
    print(f"Gender Classification Accuracy: {gender_acc * 100:.2f}%")

    # Classification Report
    y_pred_gender = (gender_model.predict(X_test_gender) > 0.5).astype("int32")
    print("\nGender Classification Classification Report:")
    print(classification_report(y_test_gender, y_pred_gender, target_names=['Male', 'Female']))

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_gender, y_pred_gender))
else:
    print("\nSkipping Gender Classification as there are no speech samples in noisy conditions.")