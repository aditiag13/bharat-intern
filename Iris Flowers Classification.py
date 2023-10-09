# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the Iris dataset (it's available in scikit-learn)
from sklearn.datasets import load_iris
iris = load_iris()

# Create a DataFrame from the Iris dataset
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Assuming you want to use only petal length and sepal length as features
X = iris_df[['petal length (cm)', 'sepal length (cm)']]
y = iris_df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# You can now use this trained model to predict the species of a flower sample
# For example, you can predict the species of a flower with certain petal and sepal lengths
new_data = np.array([[4.5, 5.0]])  # Replace with your own petal and sepal length values
predicted_species = model.predict(new_data)
print(f"Predicted Species: {iris.target_names[predicted_species[0]]}")
