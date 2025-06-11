from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Convert target into 3 classes: Low (0), Medium (1), High (2)
y_class = np.digitize(y, bins=[100, 200])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=4)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Evaluate
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))