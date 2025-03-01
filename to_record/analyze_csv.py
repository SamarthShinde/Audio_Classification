import pandas as pd


def analyze_csv(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Calculate duration for each segment using the 'Segment Time' column
    # Split the time range (e.g., "0-5") and calculate the duration
    data[['Start', 'End']] = data['Segment Time'].str.split('-', expand=True).astype(float)
    data['Duration'] = data['End'] - data['Start']

    # Explode 'Class Name' column in case multiple classes are assigned to a segment
    data['Class Name'] = data['Class Name'].str.split(',')
    data = data.explode('Class Name').reset_index(drop=True)

    # Calculate total time per class
    total_time_per_class = data.groupby('Class Name')['Duration'].sum()

    # Calculate total number of segments per class
    total_segments_per_class = data['Class Name'].value_counts()

    # Combine both results into a single DataFrame for easy viewing
    result = pd.DataFrame({
        'Total Time': total_time_per_class,
        'Total Number of Segments': total_segments_per_class
    })

    return result


# Call the function and display the result
result = analyze_csv('/Users/samarthshinde/Desktop/CSV/combined_file.csv')
print(result)