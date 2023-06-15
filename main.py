import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.morphology import skeletonize, thin
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu

# Load an image from SOCOFing dataset
img = io.imread('SOCOFing/data/Fingerprints - Set A/101_1.tif', as_gray=True)

# Binarize the image using Otsu thresholding
threshold_value = threshold_otsu(img)
binary_image = img > threshold_value

# Apply skeletonization to the binary image
skeleton = skeletonize(binary_image)

# Thin the skeletonized image to remove redundant pixels
thinned = thin(skeleton)

# Find the coordinates of the ridge endings and bifurcations (minutiae)
minutiae_coordinates = peak_local_max(thinned, min_distance=10, exclude_border=False)

# Plot the image and minutiae for visualization
fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')
ax.plot(minutiae_coordinates[:, 1], minutiae_coordinates[:, 0], 'r.', markersize=10)
plt.show()
