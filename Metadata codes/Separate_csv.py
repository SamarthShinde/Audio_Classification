import os
import csv

def extract_classes(csv_file):
    """
    Extract unique classes from the provided CSV file.
    """
    classes = set()
    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            labels = row['labels'].split(',')
            classes.update(labels)
    return classes

def create_class_csv(csv_file, classes):
    """
    Create separate CSV files for each class with class, folder, and audio file name.
    """
    for class_name in classes:
        class_csv = class_name + '.csv'
        with open(class_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Class', 'Folder', 'Path'])
            with open(csv_file, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    labels = row['labels'].split(',')
                    if class_name in labels:
                        folder = row['Folder']
                        audio_path = os.path.basename(row['AudioPath'])
                        writer.writerow([class_name, folder, audio_path])
        print(f"Created {class_csv}")

# Example usage
input_csv = '/Users/samarthshinde/Downloads/Data_path.csv'  # Specify the input CSV file name
classes = extract_classes(input_csv)
create_class_csv(input_csv, classes)
