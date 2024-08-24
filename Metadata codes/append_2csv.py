import csv

def append_csv_files(csv_file1, csv_file2, output_csv):
    """
    Append the rows of two CSV files with the same structure and columns.
    """
    # Read data from the first CSV file
    with open(csv_file1, 'r', newline='') as file1:
        reader1 = csv.DictReader(file1)
        rows1 = list(reader1)

    # Read data from the second CSV file
    with open(csv_file2, 'r', newline='') as file2:
        reader2 = csv.DictReader(file2)
        rows2 = list(reader2)

    # Append rows from the second CSV file to the rows from the first CSV file
    combined_rows = rows1 + rows2

    # Write the combined rows to the output CSV file
    with open(output_csv, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=reader1.fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)

# Example usage
csv_file1 = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Audio_datasets/UrbanSound_filter565.csv'  # Replace with the path to the first CSV file
csv_file2 = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Audio_datasets/metadata_W_female.csv'  # Replace with the path to the second CSV file
output_csv = 'combined_file.csv'  # Replace with the path to the output combined CSV file
append_csv_files(csv_file1, csv_file2, output_csv)
