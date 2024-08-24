import csv

def modify_csv(csv_file):
    """
    Modify the 'fold', 'class', and 'classID' columns in the CSV file based on class labels.
    """
    # Define the mapping of class labels to fold, class name, and classID
    class_mapping = {
        'engine_idling': {'fold': 2, 'class': 'engine idling', 'classID': 3},
        'air_conditioner': {'fold': 1, 'class': 'air conditioner', 'classID': 2},
        'car_horn': {'fold': 0, 'class': 'car horn', 'classID': 1}
    }

    # Read the CSV file and modify the specified columns
    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            class_label = row['class']
            if class_label in class_mapping:
                row['fold'] = class_mapping[class_label]['fold']
                row['class'] = class_mapping[class_label]['class']
                row['classID'] = class_mapping[class_label]['classID']
            rows.append(row)

    # Write the modified data back to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# Example usage
csv_file = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Audio_datasets/UrbanSound_filter565.csv'  # Replace with the path to your CSV file
modify_csv(csv_file)
