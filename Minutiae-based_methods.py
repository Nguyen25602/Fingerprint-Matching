import cv2
import numpy as np
import os
from scipy.spatial.distance import cdist

# Function to extract minutiae from an image
def extract_minutiae(image):
    # Thinning the image
    thinned_image = thinning(image)

    # Calculate the crossing numbers
    crossing_numbers = calculate_crossing_numbers(thinned_image)

    # Calculate the ridge endings and bifurcations
    endings, bifurcations = calculate_endings_and_bifurcations(thinned_image, crossing_numbers)

    # Combine the minutiae into a single array
    minutiae = np.concatenate((endings, bifurcations), axis=0)

    return minutiae

def match_minutiae(minutiae1, minutiae2):
    # Calculate the pairwise distance matrix between the minutiae
    dist_matrix = cdist(minutiae1, minutiae2)

    # Calculate the minimum distance for each minutia in minutiae1
    min_dist = np.min(dist_matrix, axis=1)

    # Calculate the average minimum distance
    avg_min_dist = np.mean(min_dist)

    return avg_min_dist

def calculate_crossing_numbers(image):
    # Define the pattern matrix
    pattern = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=np.uint8)

    # Pad the image with zeros to avoid border issues
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant')

    # Perform the hit-miss transform
    hit_miss = cv2.morphologyEx(padded_image, cv2.MORPH_HITMISS, pattern)

    # Count the number of white pixels in the hit-miss output
    crossing_numbers = np.count_nonzero(hit_miss == 255)

    return crossing_numbers

def calculate_endings_and_bifurcations(thinned_image, crossing_numbers):
    # Initialize arrays to hold the locations of ridge endings and bifurcations
    endings = []
    bifurcations = []

    # Define the neighborhood matrix
    neighborhood = np.array([[1, 1, 1],
                             [1, 0, 1],
                             [1, 1, 1]])

    # Loop over each pixel in the thinned image
    for i in range(1, thinned_image.shape[0] - 1):
        for j in range(1, thinned_image.shape[1] - 1):
            if thinned_image[i, j] == 0:
                # Check if the current pixel has three or four neighbors
                neighbors = thinned_image[i - 1:i + 2, j - 1:j + 2]
                num_neighbors = np.sum(neighbors) - 1  # subtract 1 to exclude the current pixel

                if num_neighbors == 2:
                    # Check if the current pixel is a ridge ending or bifurcation
                    num_transitions = np.sum(neighbors * neighbor) - crossing_numbers[i, j]
                    if num_transitions == 1:
                        endings.append((i, j))
                    elif num_transitions == 3:
                        bifurcations.append((i, j))

    return endings, bifurcations

def thinning(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    ret, img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_not(img)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        dilated = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, dilated)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            break

    return skel


path = 'SOCOFing/Real/'

# Get list of file names in the directory
file_list = os.listdir(path)

# Load images from files
socofing_dataset = []
for filename in file_list:
    # Check that the file is a BMP image
    if filename.endswith('.BMP'):
        image = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        socofing_dataset.append(image)

# Load the query image
query_image = cv2.imread('SOCOFing/Altered/Altered-Hard/1__M_Left_little_finger_Obl.BMP', cv2.IMREAD_GRAYSCALE)

# Extract minutiae from the query image
query_minutiae = extract_minutiae(query_image)

# Match the minutiae with the SOCOFing dataset
best_match_index, best_match_score = -1, 0
for i, image in enumerate(socofing_dataset):
    # Thinning the dataset image
    image_matching = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)

    # Extract minutiae from the dataset image
    dataset_minutiae = extract_minutiae(image_matching)

    # Match the minutiae
    match_score = match_minutiae(query_minutiae, dataset_minutiae)

    # Keep track of the best match
    if match_score > best_match_score:
        best_match_score = match_score
        best_match_index = i

# Display the best match
best_match_image = socofing_dataset[best_match_index]
cv2.imshow('Best match', best_match_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
