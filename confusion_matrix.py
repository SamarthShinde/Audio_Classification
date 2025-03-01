import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the inference results CSV file
file_path = '/Users/samarthshinde/Desktop/CSV/inference_results_with_confidence.csv'
inference_df = pd.read_csv(file_path)

# Drop rows with NaN values in 'Predicted Inference' or 'Ground Truth Inference'
inference_df = inference_df.dropna(subset=['Predicted Inference', 'Ground Truth Inference'])

# Convert to string to avoid mixed data type issues
y_pred = inference_df['Predicted Inference'].astype(str)
y_true = inference_df['Ground Truth Inference'].astype(str)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred, labels=y_true.unique())

# Customize the display of the confusion matrix
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size to match the example
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=y_true.unique())
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')

# Add labels, title, and improve style
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Class", fontsize=14)
plt.ylabel("True Class", fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.yticks(rotation=0)    # Keep y-axis labels horizontal

# Show plot
plt.tight_layout()
plt.show()