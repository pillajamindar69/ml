from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
diabetes = load_diabetes()
X = diabetes.data
df = pd.DataFrame(X, columns=diabetes.feature_names)

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
df['Cluster'] = kmeans.fit_predict(X)

# Visualize using two features: BMI vs. BP
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='bmi', y='bp', hue='Cluster', palette='Set2')
plt.title('K-Means Clustering on Diabetes Dataset (BMI vs. BP)')
plt.grid(True)
plt.show()