# 📊 Clustering Algorithms: KMeans, Hierarchical, and DBSCAN

## 📌 Overview
This project demonstrates three popular **unsupervised learning clustering techniques**:
- **KMeans Clustering**
- **Hierarchical Clustering**
- **DBSCAN**

We also use **PCA (Principal Component Analysis)** to reduce dimensions for visualization, and evaluate clusters using multiple metrics.

---

## ⚙️ KMeans Clustering

### 🔹 Explanation
- KMeans partitions data into **k clusters**.
- Each cluster is represented by its **centroid**.
- Points are assigned to the **nearest centroid** until convergence.

### 🔹 Elbow Method (Choosing k)
```python
wcss = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(2, 10), wcss, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method for Optimal k")
plt.show()
# 📊 Clustering Algorithms: Hierarchical and DBSCAN

## 📌 Overview
This project demonstrates two popular **unsupervised clustering techniques**:
- **Hierarchical Clustering**
- **DBSCAN**

We also use **PCA (Principal Component Analysis)** to reduce dimensions for visualization.


## ⚙️ Hierarchical Clustering

### 🔹 Explanation
- Builds a **tree-like structure (dendrogram)** of clusters.
- Starts with each data point as its own cluster.
- Iteratively merges the two closest clusters until all are grouped.
- You can decide the number of clusters by **cutting the dendrogram** at a chosen level.

### 🔹 Dendrogram with Ward’s Method

import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sch.dendrogram(sch.linkage(X_pca, method='ward'))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()
