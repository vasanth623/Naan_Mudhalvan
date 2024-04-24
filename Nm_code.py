import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Load historical pricing data into Pandas DataFrame
data = pd.read_csv('SM_Historical_data.csv')
data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' column to datetime format
print(data.head())
# Step 2: Add technical indicators as features
data['10_day_EMA'] = data['Close'].ewm(span=10, adjust=False).mean()

# Step 3: Train a simple linear regression model
X = data[['10_day_EMA']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Analyze the accuracy of the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)

# Step 5: Plot the results as a line plot
plt.figure(figsize=(12, 6))

# Plot time series
plt.plot(data['Date'], data['Close'], color='blue', label='Predicted Closing price')



plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Predicted Closing Prices')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
