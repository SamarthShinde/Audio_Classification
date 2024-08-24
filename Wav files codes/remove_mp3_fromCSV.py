import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(r'C:\Users\Admin\Desktop\Audio_class\UrbanSoundMetadata.csv')

# Filter out rows containing '.mp3' in the 'slice_file_name' column
filtered_df = df[~df['slice_file_name'].str.contains('.mp3')]

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv('filtered_file.csv', index=False)

print("Filtered data saved to 'filtered_file.csv'")