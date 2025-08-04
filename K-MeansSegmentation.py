import pandas as pd
from utils import k_means_labeling
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt

rfm = pd.read_csv('rfm_scoring.csv')

# Feature Scaling
rfm_features = rfm[['Recency', 'Frequency', 'Monetary']].copy()
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

# Elbow Method
wcss = []
K_values = range(1,11)

for k in K_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 4))
plt.plot(K_values, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Optimal K
knee = KneeLocator(K_values, wcss, curve='convex', direction='decreasing')
k_optimal = knee.knee

print(f"\n Optimal number of clusters (k): {k_optimal}")

# K-means Clustering
kmeans = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42)
rfm['K-Means Cluster'] = kmeans.fit_predict(rfm_scaled)


# Quantile Ranging
r_33 = rfm['Recency'].quantile(0.33)
r_66 = rfm['Recency'].quantile(0.66)

f_33 = rfm['Frequency'].quantile(0.33)
f_66 = rfm['Frequency'].quantile(0.66)

m_33 = rfm['Frequency'].quantile(0.33)
m_66 = rfm['Frequency'].quantile(0.66)

# Cluster Analysis
cluster_summary = rfm.groupby('K-Means Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
}).round(1)

cluster_summary['Segment'] = cluster_summary.apply(
    lambda row: k_means_labeling(row, r_33, r_66, f_33, f_66, m_33, m_66),
    axis=1
)

print(cluster_summary)

# Map labels back to main RFM dataframe
cluster_map = cluster_summary['Segment'].to_dict()
rfm['K-Means Segment'] = rfm['K-Means Cluster'].map(cluster_map)

# Save final output
rfm.to_csv("rfm_with_segments.csv", index=False)
print("\n Cluster Labeling Complete!")

print(rfm.head(50))
# Visualization
segment_counts = rfm['K-Means Segment'].value_counts()
print(segment_counts)
plt.figure(figsize=(10, 6))
plt.bar(segment_counts.index, segment_counts.values, color='skyblue')

plt.title("Customer Count per Segment")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

