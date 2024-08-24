import csv
import random

def filter_csv(csv_file, output_csv, sample_size=350):
    """
    Filter a CSV file to keep only a random sample of each class with size <= sample_size.
    """
    # Dictionary to store data for each class
    class_data = {}

    # Read the CSV file and organize data by class
    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            class_label = row['class']
            if class_label not in class_data:
                class_data[class_label] = []
            class_data[class_label].append(row)

    # Randomly sample each class
    sampled_data = []
    for class_label, data in class_data.items():
        if len(data) <= sample_size:
            sampled_data.extend(data)
        else:
            sampled_data.extend(random.sample(data, sample_size))

    # Write the sampled data to the output CSV file
    with open(output_csv, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(sampled_data)

# Example usage
csv_file = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Metadata codes/combined_file.csv'  # Replace with the path to your CSV file
output_csv = 'sampled_data.csv'  # Replace with the path to the output CSV file
filter_csv(csv_file, output_csv, sample_size=350)
