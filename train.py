# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Load the dataset from a CSV file
drug_df = pd.read_csv("Data/drug200.csv")

# Shuffle the dataset to ensure randomness
drug_df = drug_df.sample(frac=1)

# Display the first 3 rows of the dataset
drug_df.head(3)

# Separate the features (X) and the target variable (y)
X = drug_df.drop("Drug", axis=1).values
y = drug_df.Drug.values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

# Define the categorical and numerical columns
cat_col = [1, 2, 3]
num_col = [0, 4]

# Create a ColumnTransformer to apply different preprocessing steps to different columns
transform = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), cat_col),  # Encode categorical columns
        ("num_imputer", SimpleImputer(strategy="median"), num_col),  # Impute missing values in numerical columns
        ("num_scaler", StandardScaler(), num_col),  # Scale numerical columns
    ]
)

# Create a pipeline that first preprocesses the data and then fits a RandomForestClassifier
pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
    ]
)

# Fit the pipeline on the training data
pipe.fit(X_train, y_train)

# Import metrics to evaluate the model
from sklearn.metrics import accuracy_score, f1_score

# Make predictions on the test set
predictions = pipe.predict(X_test)

# Calculate accuracy and F1 score
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

# Print the accuracy and F1 score
print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))

# Save the metrics to a text file
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}.")

# Import libraries for plotting the confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Generate the confusion matrix
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)

# Plot the confusion matrix
disp.plot()

# Save the confusion matrix plot as an image file
plt.savefig("Results/model_results.png", dpi=120)

# Import skops for model serialization
import skops.io as sio

# Save the trained pipeline to a file
sio.dump(pipe, "Model/drug_pipeline.skops")

# Load the trained pipeline from the file with trusted types
sio.load("Model/drug_pipeline.skops", trusted='numpy.dtype')
