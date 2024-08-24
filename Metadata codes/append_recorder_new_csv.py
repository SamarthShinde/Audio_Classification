import pandas as pd

# Load the CSV files
metadata_test1 = pd.read_csv('/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/metadata_test1.csv')
metadata_test = pd.read_csv('/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/metadata_test.csv')

# Add the 'comment' column with 'none' as the default value to metadata_test1
metadata_test1['comment'] = 'none'

# Concatenate the two DataFrames
combined_metadata = pd.concat([metadata_test1, metadata_test], ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_metadata.to_csv('/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/combined_metadata.csv', index=False)

print("CSV files have been combined and saved as combined_metadata.csv")