import csv

def modify_fold_values(csv_file):
    """
    Modify fold values in the CSV file.
    """
    modified_rows = []
    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Modify fold values
            if row['fold'] == 'CREMA':
                row['fold'] = '8'
            elif row['fold'] == 'RAVDESS':
                row['fold'] = '9'
            elif row['fold'] == 'SAVEE':
                row['fold'] = '10'
            elif row['fold'] == 'TESS':
                row['fold'] = '11'
            modified_rows.append(row)

    # Rewrite the CSV file with modified rows
    with open(csv_file, 'w', newline='') as file:
        fieldnames = modified_rows[0].keys() if modified_rows else []
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(modified_rows)

# Example usage
csv_file = '/Users/samarthshinde/Desktop/ready_codes/Audio_Classification_ML/Audio_datasets/metadata.csv'  # Path to the CSV file
modify_fold_values(csv_file)
