# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your wine quality dataset (assuming it's in a CSV file)
data = pd.read_csv('wine_quality_data.csv')

# Assuming your dataset has features (e.g., 'fixed acidity', 'pH', 'alcohol', etc.) and the target 'quality'
# X contains the features, and y contains the target variable
X = data[['fixed acidity', 'pH', 'alcohol']]  # Add more features as needed
y = data['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error to evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# You can now use this trained model to predict the quality of wine samples
# For example, you can predict the quality of a wine sample with certain features
new_data = np.array([[7.0, 3.0, 12.0]])  # Replace with your own feature values
predicted_quality = model.predict(new_data)
print(f"Predicted Wine Quality: {predicted_quality[0]}")