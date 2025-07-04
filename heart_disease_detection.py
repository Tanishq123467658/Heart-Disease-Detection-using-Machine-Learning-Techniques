import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# Set visual style for seaborn
sns.set(style="whitegrid")


# 1. Data Loading and Exploration

heart_data = pd.read_csv('data.csv')

# Print basic dataset information
print("Dataset Preview:")
print(heart_data.head())
print("\nDataset Shape:", heart_data.shape)
print("\nDataset Info:")
heart_data.info()
print("\nMissing Values per Column:")
print(heart_data.isnull().sum())
print("\nStatistical Summary:")
print(heart_data.describe())
print("\nTarget Variable Distribution:")
print(heart_data['target'].value_counts())


# 2. Data Visualization

# Plot distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=heart_data, palette='viridis')
plt.title('Distribution of Heart Disease Target Variable')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.savefig('target_distribution.png', bbox_inches='tight')
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
corr = heart_data.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png', bbox_inches='tight')
plt.show()

# Visualize age distribution by target
plt.figure(figsize=(10, 4))
sns.histplot(data=heart_data, x='age', hue='target', kde=True, palette='viridis', bins=20)
plt.title('Age Distribution by Heart Disease Status')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('age_distribution.png', bbox_inches='tight')
plt.show()

# Visualize cholesterol levels by target using a boxplot
plt.figure(figsize=(10, 4))
sns.boxplot(x='target', y='chol', data=heart_data, palette='viridis')
plt.title('Cholesterol Levels by Heart Disease Status')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Cholesterol')
plt.savefig('cholesterol_boxplot.png', bbox_inches='tight')
plt.show()


# 3. Data Preparation and Modeling

# Separate features and target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print("\nData split:")
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# 4. Model Evaluation

# Evaluate on training data
Y_train_pred = model.predict(X_train)
training_accuracy = accuracy_score(Y_train, Y_train_pred)
print("\nAccuracy on Training Data:", training_accuracy)

# Evaluate on test data
Y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_test_pred)
print("Accuracy on Test Data:", test_accuracy)

# Print confusion matrix and classification report
conf_matrix = confusion_matrix(Y_test, Y_test_pred)
class_report = classification_report(Y_test, Y_test_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.show()


# 5. Prediction on New Data (User Input)

# Define the feature order (update if necessary to match your CSV)
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                 'ca', 'thal']

print("\nEnter the following features as comma-separated values:")
print(", ".join(feature_names))
user_input = input("Input: ")

try:
    # Process the user input: convert each value to float
    input_data = [float(x.strip()) for x in user_input.split(',')]
    if len(input_data) != len(feature_names):
        raise ValueError("Incorrect number of features.")
except Exception as e:
    print("Error processing input:", e)
    exit()

# Convert to NumPy array and reshape for prediction
input_data_np = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_data_np)
print("\nPrediction for the input data:", prediction)

if prediction[0] == 0:
    print('The Person does not have Heart Disease')
else:
    print('The Person has Heart Disease')
