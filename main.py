import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import hdbscan
import warnings

# Suppress FutureWarnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Step 1: Load and Explore the Dataset
image_dir = '/content/lab4-AI-Assignment/img_align_celeba'  # Adjust path as needed
image_limit = 100  # Limit the dataset to 100 images for faster processing
images = []
filenames = []

# Loop through the directory and load a limited number of images
for i, filename in enumerate(os.listdir(image_dir)):
    if i >= image_limit:  # Stop after loading the specified number of images
        break
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            img = cv2.imread(os.path.join(image_dir, filename))
            if img is None:
                raise cv2.error(f"Error loading image: {filename}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            images.append(img)
            filenames.append(filename)
        except cv2.error as e:
            print(f"Error loading image: {filename} - {e}")

# Convert to NumPy array
images = np.array(images)
print(f"Loaded {images.shape[0]} images with shape {images.shape[1:]}.")

# Visualize sample images
plt.figure(figsize=(10, 5))
for i in range(min(5, len(images))):  # Show up to 5 sample images
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
plt.suptitle("Sample Images from Dataset")
plt.show()

# Step 2: Data Preprocessing
flattened_images = images.reshape(images.shape[0], -1)  # Flatten images
scaler = StandardScaler()
scaled_data = scaler.fit_transform(flattened_images)  # Standardize data

# Step 3: Dimensionality Reduction with PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
reduced_data = pca.fit_transform(scaled_data)
print(f"Reduced data shape after PCA: {reduced_data.shape}")

# Normalize data for cosine similarity
reduced_data = normalize(reduced_data)

# Step 4: Apply HDBSCAN Clustering
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=3,
    min_samples=1,
    metric='cosine',
    algorithm='generic'  # Use 'generic' algorithm for cosine metric
)
cluster_labels = clusterer.fit_predict(reduced_data)

# Analyze clusters and noise points
noise_points = np.sum(cluster_labels == -1)
print(f"Number of noise points: {noise_points}")

# Step 5: Experiment with Different Metrics
metrics = ['euclidean', 'manhattan', 'cosine']
metric_results = {}
for metric in metrics:
    temp_clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1, metric=metric, algorithm='generic')
    temp_labels = temp_clusterer.fit_predict(reduced_data)
    metric_results[metric] = temp_labels
    noise_count = np.sum(temp_labels == -1)
    print(f"Metric: {metric}, Noise Points: {noise_count}")

# Step 6: Visualize Results with t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_reduced_data = tsne.fit_transform(reduced_data)

plt.figure(figsize=(8, 6))
plt.scatter(tsne_reduced_data[:, 0], tsne_reduced_data[:, 1], c=cluster_labels, cmap='viridis', s=5)
plt.title("HDBSCAN Clustering of Facial Images")
plt.colorbar(label="Cluster Labels")
plt.show()

# Evaluate clustering quality
if len(set(cluster_labels)) > 1:  # Ensure at least 2 clusters exist
    silhouette_avg = silhouette_score(reduced_data, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.2f}")
else:
    print("Silhouette score not applicable: only one cluster detected.")

# Step 7: Representative Images for Clusters
unique_labels = np.unique(cluster_labels[cluster_labels != -1])  # Exclude noise
representative_images = {}
for label in unique_labels:
    cluster_indices = np.where(cluster_labels == label)[0]
    cluster_center = reduced_data[cluster_indices].mean(axis=0)  # Cluster center
    distances = np.linalg.norm(reduced_data[cluster_indices] - cluster_center, axis=1)
    closest_index = cluster_indices[np.argmin(distances)]
    representative_images[label] = images[closest_index]

# Visualize representative images for clusters
plt.figure(figsize=(10, len(unique_labels) * 2))
for i, (label, img) in enumerate(representative_images.items()):
    plt.subplot(len(unique_labels), 1, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Representative Image for Cluster {label}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Step 8: Test New Image on the Cluster Model
new_image = images[0].reshape(1, -1)  # Example: First image
new_image_scaled = scaler.transform(new_image)
new_image_pca = pca.transform(new_image_scaled)
new_image_pca_normalized = normalize(new_image_pca)

# Assign the new image to the closest cluster
if hasattr(clusterer, 'labels_') and len(clusterer.labels_) > 0:
    cluster_centers = []
    for label in unique_labels:
        cluster_indices = np.where(cluster_labels == label)[0]
        cluster_center = reduced_data[cluster_indices].mean(axis=0)
        cluster_centers.append(cluster_center)
    
    cluster_centers = np.array(cluster_centers)
    distances = np.linalg.norm(cluster_centers - new_image_pca_normalized, axis=1)
    closest_cluster = unique_labels[np.argmin(distances)]
    print(f"New image assigned to cluster: {closest_cluster}")
else:
    print("No clusters found to assign the new image.")

