import os
import pickle
from sklearn.preprocessing import LabelEncoder

# Assuming 'classes' is a list containing your class labels
classes = ['Male','Female','Engine_rev','No_sound','Music']

# Initialize LabelEncoder
encoder = LabelEncoder()

# Fit LabelEncoder
encoder.fit(classes)

# Define the directory path where you want to save the file
save_directory = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Lable_encoder'

# Ensure that the directory exists, create it if it doesn't
os.makedirs(save_directory, exist_ok=True)

# Save LabelEncoder to file
file_path = os.path.join(save_directory, 'audio_test7.pkl')
try:
    with open(file_path, 'wb') as f:
        pickle.dump(encoder, f)
    print("Label encoder saved successfully at:", file_path)
except Exception as e:
    print("An error occurred while saving the label encoder:", e)
