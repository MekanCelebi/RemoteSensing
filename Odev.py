import numpy as np
from sklearn.cluster import KMeans
import rasterio
import matplotlib.pyplot as plt

# Load the multispectral image
with rasterio.open('multispectral.tif') as src:
    img = src.read()

# Extract the four bands and reshape
data = img[[0, 1, 2, 3], :, :].reshape((-1, 4))

# Initialize KMeans
kmeans = KMeans(n_clusters=5)

# Fit the model
kmeans.fit(data)

# Get the labels
labels = kmeans.labels_

# Reshape the labels
labels = labels.reshape(img.shape[1], img.shape[2])

# Save the classified image
with rasterio.open('classified_image.tif', 'w', driver='GTiff', height=labels.shape[0], 
                   width=labels.shape[1], count=1, dtype=labels.dtype) as dst:
    dst.write(labels, 1)

# Display the classified image
plt.imshow(labels, cmap='viridis')
plt.show()