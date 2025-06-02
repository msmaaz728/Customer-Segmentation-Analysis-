import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
customer_data = pd.read_csv('customer_data.csv')
X = customer_data[['Age', 'Income', 'Spending_Score']].dropna()

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customer_data['Cluster'] = kmeans.fit_predict(X)

# Visualize clusters
plt.scatter(customer_data['Income'], customer_data['Spending_Score'],
            c=customer_data['Cluster'], cmap='viridis')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation Analysis')
plt.show()

# Cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Cluster counts
print("\nCluster Counts:")
print(customer_data['Cluster'].value_counts())

# Cluster averages
print("\nCluster Averages:")
print(customer_data.groupby('Cluster').mean(numeric_only=True))