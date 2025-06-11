from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
ds = load_diabetes()
X = ds.data
y = ds.target

# Convert regression target into classification labels
# Example: 3 classes - low (0), medium (1), high (2)
y_class = np.digitize(y, bins=[100, 200])  # Split at 100 and 200

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=1)

# Train model
clf = DecisionTreeClassifier(max_depth=3, criterion='entropy')
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# Visualize tree
plt.figure(figsize=(16, 10))
plot_tree(clf, feature_names=ds.feature_names, class_names=["Low", "Medium", "High"], filled=True)
plt.title("Decision Tree - Diabetes Dataset (Binned Classification)")
plt.show()