import pandas as pd

# Paths of the four CSV files to combine
csv_file1 = '/Users/samarthshinde/Desktop/CSV/annotations.csv'
csv_file2 = '/Users/samarthshinde/Desktop/CSV/annotations_2.csv'
csv_file3 = '/Users/samarthshinde/Desktop/CSV/annotations_3.csv'
csv_file4 = '/Users/samarthshinde/Desktop/CSV/annotations_4.csv'

# Load each CSV file into a DataFrame
df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)
df3 = pd.read_csv(csv_file3)
df4 = pd.read_csv(csv_file4)

# Concatenate the DataFrames
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('/Users/samarthshinde/Desktop/CSV/combined_file.csv', index=False)

print("CSV files successfully combined and saved as 'combined_file.csv'")