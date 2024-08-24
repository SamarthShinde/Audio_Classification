import pandas as pd

# Load the provided metadata
metadata = pd.read_csv('/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/metadata_test.csv')

# Define the classes for each task
engine_rev_classes = ['Engine_rev', 'No_sound']  # Use 'No_sound' for 'engine_off'
second_level_classes = ['Male', 'Female', 'No_sound']  # Adjust these class names according to your actual classes

# Filter the metadata for engine_rev detection
engine_rev_metadata = metadata[metadata['class'].isin(engine_rev_classes)].copy()

# Rename 'No_sound' to 'No_engine_rev' for the engine_rev_metadata but keep the original class names
engine_rev_metadata['class'] = engine_rev_metadata['class'].replace({'No_sound': 'No_engine_rev'})

# Create new class IDs for engine_rev_metadata starting from 1
engine_rev_metadata['classID'] = engine_rev_metadata['class'].factorize()[0] + 1

# Filter the metadata for second level detection
second_level_metadata = metadata[metadata['class'].isin(second_level_classes)].copy()

# Create new class IDs for second_level_metadata starting from 1
second_level_metadata['classID'] = second_level_metadata['class'].factorize()[0] + 1

# Save the filtered metadata to new CSV files
engine_rev_metadata.to_csv('/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/engine_rev_metadata.csv', index=False)
second_level_metadata.to_csv('/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/second_level_metadata.csv', index=False)

print("Metadata split and saved successfully.")

# Verify the new metadata files
engine_rev_metadata = pd.read_csv('/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/engine_rev_metadata.csv')
second_level_metadata = pd.read_csv('/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/second_level_metadata.csv')

print("Engine Rev Metadata:")
print(engine_rev_metadata.head())

print("\nSecond Level Metadata:")
print(second_level_metadata.head())