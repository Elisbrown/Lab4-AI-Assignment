import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import hdbscan

# Specify the directory containing the images
image_dir = 'img_align_celeba'  # Adjust the path if necessary

# Create an empty list to store the images and filenames
images = []
filenames = []

# Loop through all files in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            # Load the image using OpenCV
            img = cv2.imread(os.path.join(image_dir, filename))
            if img is None:
                raise cv2.error(f"Error loading image: {filename}")  # Raise an exception if the image is None
            # Convert the image to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Append the image and filename to the lists
            images.append(img)
            filenames.append(filename)
        except cv2.error as e:
            print(f"Error loading image: {filename} - {e}")

# Convert the list of images to a NumPy array
images = np.array(images)

# Flatten the images
flattened_images = images.reshape(images.shape[0], -1)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(flattened_images)

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of the variance
reduced_data = pca.fit_transform(scaled_data)

# Apply HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
cluster_labels = clusterer.fit_predict(reduced_data)

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_reduced_data = tsne.fit_transform(reduced_data)

# Visualize the clusters
plt.scatter(tsne_reduced_data[:, 0], tsne_reduced_data[:, 1], c=cluster_labels)
plt.title("HDBSCAN Clustering of Facial Images")
plt.show()

# Analyze the results, identify noise points, etc.