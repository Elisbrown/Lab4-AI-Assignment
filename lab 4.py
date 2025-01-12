import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
import os
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
lfw_dataset = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
data = lfw_dataset.data
images = lfw_dataset.images
target = lfw_dataset.target
target_names = lfw_dataset.target_names
print(f"Data shape: {data.shape}")
print(f"Image shape: {images.shape}")
print(f"Number of classes:{len(target_names)}")

# Standerdization
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print(f"scaled data shape:{data_scaled.shape}")
# pca section
pca = PCA(n_components=50, random_state=42)
data_pca = pca.fit_transform(data_scaled)
explained_variance = pca.explained_variance_ratio_
print(f"explained variance:{sum(explained_variance)*100:.2f}%")
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5,metric='euclidean')
clusterer.fit(data_pca)
labels = clusterer.labels_
noise_points = sum(1 for label in labels if label == -1)
print(f"cluster labels:{labels}")
print(f"number of noise points:{noise_points}")
# Step 6: Analyze Noise Points
noise_points = data_pca[labels == -1] #Fixed this line.  cluster_labels was not defined
print(f"Number of noise points: {len(noise_points)}")

# Step 7: Test on New Data (Optional)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(data_pca,clusterer.labels_)
new_image = data_pca[0].reshape(1,-1)  # Example: First image
new_label = knn.predict(new_image)
print(f"Cluster assigned to the new image: {new_label}")
clusterer.fit(data_pca)
labels = clusterer.labels_
noise_points = sum(1 for label in labels if label == -1)
print(f"cluster labels:{labels}")
print(f"number of noise points:{noise_points}")
# Step 6: Analyze Noise Points
noise_points = data_pca[clusterer.labels_ == -1]
print(f"Number of noise points: {len(noise_points)}")

# Step 7: Test on New Data (Optional)
new_image = data_pca[0].reshape(1,-1)  # Example: First image
new_label = knn.predict(new_image)
print(f"Cluster assigned to the new image: {new_label}")
pca_2d = PCA(n_components=2, random_state=42)
data_pca_2d = pca_2d.fit_transform(data_scaled)
plt.scatter(data_pca_2d[:, 0], data_pca_2d[:, 1], c=labels, cmap='Spectral',s=10)
plt.colorbar(label='Cluster')
plt.title('HBDSCAN CLUSTERING ON LFW DATASET')
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 2')
plt.show()
