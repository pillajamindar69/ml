import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample house dataset
data = {
    'Area': [1500, 1800, 2400, 3000, 3500],
    'Bedrooms': [3, 4, 3, 5, 4],
    'Age': [10, 15, 20, 18, 8],
    'Price': [400000, 500000, 600000, 650000, 620000]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# Split data (test size increased to avoid R² warning)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Outputs
print("✅ Coefficients:", model.coef_)
print("✅ Intercept:", model.intercept_)
print("✅ Mean Squared Error:", mse)
print("✅ R2 Score:", r2)