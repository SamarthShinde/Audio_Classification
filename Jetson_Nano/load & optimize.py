import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Load the Keras model
model = load_model('/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/H_file/audio_test1.h5')

# Compile the model with optimizations for deployment
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Optionally, you can further optimize or simplify the model architecture if neededpip