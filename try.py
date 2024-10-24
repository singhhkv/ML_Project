import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_csv('car_data.csv')

# Features and target variable
X = data[['Year', 'Mileage', 'Engine_Size']]
y = data['Selling_Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions for the test set
y_pred = model.predict(X_test)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='red', alpha=0.6, label='Actual Prices')
plt.scatter(range(len(y_pred)), y_pred, color='blue', alpha=0.6, label='Predicted Prices')
plt.plot(range(len(y_pred)), y_pred, color='blue', alpha=0.4)  # Connecting line for predicted prices
plt.xlabel('Test Sample Index')
plt.ylabel('Selling Price')
plt.title('Actual vs. Predicted Selling Price')
plt.legend()
plt.grid(True)
plt.show()

# Model evaluation
score = model.score(X_test, y_test)
print(f'R² score: {score}')

# User input for prediction
def predict_price():
    print("\nEnter car details for price prediction:")
    year = int(input("Year of manufacture (e.g., 2015): "))
    mileage = int(input("Mileage (e.g., 50000): "))
    engine_size = float(input("Engine size (e.g., 2.0): "))
    
    # Predicting based on user input
    input_data = pd.DataFrame([[year, mileage, engine_size]], columns=['Year', 'Mileage', 'Engine_Size'])
    predicted_price = model.predict(input_data)
    
    print(f"\nThe predicted selling price for the car is: ${predicted_price[0]:.2f}")

# Call the function to predict price based on user input
predict_price()
