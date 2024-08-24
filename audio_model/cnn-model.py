import pandas as pd
import numpy as np
import os
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras.utils import to_categorical
from keras import layers, Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from warnings import filterwarnings

# Suppressing warnings
filterwarnings('ignore')

# Read metadata
data = pd.read_csv('/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/metadata_test.csv')
audioPath = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/audiorec'

# Plot count distribution of classes
plt.figure(figsize=(10, 4))
sns.countplot(y=data['class'], palette='viridis')
plt.title('Distribution of Classes', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Class', fontsize=14)
plt.show()

# Function to augment data
def augmentData(waveform):
    # Applying pre-emphasis
    waveform = librosa.effects.preemphasis(waveform)
    # Time stretching
    if np.random.rand() < 0.5:
        rate = np.random.uniform(0.8, 1.2)
        waveform = librosa.effects.time_stretch(waveform, rate=rate)
    # Adding noise
    waveform += 0.01 * np.random.normal(size=waveform.shape)
    return waveform

# Function to extract MFCC features
def mfccExtract(file):
    waveform, sampleRate = librosa.load(file)
    # waveform = augmentData(waveform) # If needed, you can apply augmentation here
    features = librosa.feature.mfcc(y=waveform, sr=sampleRate, n_mfcc=64)
    return np.mean(features, axis=1)

# List to store extracted features
extractAll = []

# Iterate through each row in the metadata
for index, row in tqdm(data.iterrows()):
    # Construct file path
    audioFile = os.path.join(audioPath, 'fold' + str(row['fold']), row['slice_file_name'])

    # Check if the file exists
    if not os.path.exists(audioFile):
        print(f"File '{audioFile}' not found. Skipping...")
        continue

    # Proceed with extracting features and appending them
    features = mfccExtract(audioFile)
    if features is None:
        print(f"No features extracted for file '{audioFile}'. Skipping...")
        continue

    extractAll.append([features, row['class']])

# Convert features into numpy array
if extractAll:
    featuresDf = pd.DataFrame(extractAll, columns=['Features', 'Class'])
    x = np.array(featuresDf['Features'].tolist())

    # Encoding classes
    encoder = LabelEncoder()
    y = encoder.fit_transform(featuresDf['Class'])
    y = to_categorical(y, num_classes=5)

    # Train-validation split
    trainX, testX, trainY, testY = train_test_split(x, y, stratify=y, test_size=0.2, random_state=0)
else:
    print("No data extracted. Check your data loading and feature extraction process.")

# Define the model
model = Sequential([
    layers.Dense(1024, activation='relu', input_shape=(64,)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Define callbacks
earlyStopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=5e-4,
    patience=10,
    restore_best_weights=True
)

reduceLR = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=1e-5
)

# Train the model
history = model.fit(
    trainX, trainY,
    validation_data=(testX, testY),
    epochs=30,
    callbacks=[earlyStopping, reduceLR]
)

# Plot training and validation loss
historyDf = pd.DataFrame(history.history)
historyDf.loc[:, ['loss', 'val_loss']].plot()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot training and validation accuracy
historyDf.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Evaluate the model
score = model.evaluate(testX, testY)[1] * 100
print(f'Validation accuracy of model : {score:.2f}%')

# Plot confusion matrix
pred = np.argmax(model.predict(testX), axis=1)
true = np.argmax(testY, axis=1)

matrix = confusion_matrix(true, pred)

plt.figure(figsize=(12, 6))
sns.heatmap(matrix, annot=True, cbar=False, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Class', fontsize=14)
plt.xticks(rotation=30)
plt.ylabel('True Class', fontsize=14)
plt.show()

# Save the model
model_save_path = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/H_file/audio_test7.h5'
model.save(model_save_path)
print(f"Model saved at {model_save_path}")
