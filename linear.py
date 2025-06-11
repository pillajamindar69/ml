import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset and select one feature for simplicity
X, y = load_diabetes(return_X_y=True)
X = X[:, [2]]  # Use the third feature (e.g., BMI)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

# Plotting
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, model.predict(X_test), color='red')
plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Linear Regression on Diabetes Dataset")
plt.show()