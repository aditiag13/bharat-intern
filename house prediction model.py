# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset (assuming it's in a CSV file)
data = pd.read_csv('house_data.csv')

# Assuming your dataset has features (e.g., 'sqft', 'num_bedrooms', 'num_bathrooms', etc.) and the target 'price'
# X contains the features, and y contains the target variable
X = data[['sqft', 'num_bedrooms', 'num_bathrooms']]
y = data['price']

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

# Plot the actual vs. predicted prices for a visual inspection
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()

# You can now use this trained model to make predictions on new data
# For example, you can predict the price of a house with certain features
new_data = np.array([[2500, 3, 2]])  # Replace with your own feature values
predicted_price = model.predict(new_data)
print(f"Predicted Price: {predicted_price[0]}")