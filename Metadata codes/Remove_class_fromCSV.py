import csv
import os

def filter_classes(input_csv, output_csv, classes_to_remove):
    """
    Filters out rows associated with specified classes from the CSV file.
    """
    # Create a temporary CSV file to store filtered data
    temp_csv = 'temp.csv'

    # Open input CSV file for reading and output CSV file for writing
    with open(input_csv, 'r', newline='') as input_file, open(temp_csv, 'w', newline='') as temp_file:
        reader = csv.reader(input_file)
        writer = csv.writer(temp_file)

        # Write header to the temporary CSV file
        writer.writerow(next(reader))

        # Copy rows to the temporary CSV file, excluding rows associated with specified classes
        for row in reader:
            if row[-1] not in classes_to_remove:
                writer.writerow(row)

    # Rename the temporary CSV file to the output CSV file
    os.rename(temp_csv, output_csv)

# Example usage
input_csv = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Metadata codes/metadata_New_test.csv'  # Path to the combined CSV file
output_csv = 'metadata_test.csv'  # Path to the filtered CSV file
classes_to_remove = ["Music_in_cabin"]

filter_classes(input_csv, output_csv, classes_to_remove)
