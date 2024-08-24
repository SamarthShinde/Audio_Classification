import pandas as pd

# Load the CSV file
file_path = "/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/to_record/metadata_test.csv"
df = pd.read_csv(file_path)

# Convert all float numbers to integers
df = df.applymap(lambda x: int(x) if isinstance(x, float) else x)

# Save the changes to the original CSV file
df.to_csv(file_path, index=False)

print(f"Updated CSV file saved to {file_path}")