import cv2
import numpy as np
import os
# Load the query image
query_image = cv2.imread('SOCOFing/Altered/Altered-Hard/40__F_Right_index_finger_Zcut.BMP', cv2.IMREAD_GRAYSCALE)

# Set path to the directory containing the SOCOFing dataset
path = 'SOCOFing/Real/'

# Get list of file names in the directory
file_list = os.listdir(path)

# Load images from files
socofing_dataset = []
for filename in file_list:
    # Check that the file is a BMP image
    if filename.endswith('.BMP'):
        image = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        socofing_dataset.append((filename, image))

# Find the best match in the SOCOFing dataset using correlation-based techniques
best_match_score = -1
best_match_index = -1

for i, (filename, image) in enumerate(socofing_dataset):
    result = cv2.matchTemplate(image, query_image, cv2.TM_CCORR_NORMED)
    match_score = np.max(result)
    if match_score > best_match_score:
        best_match_score = match_score
        best_match_index = i
        best_match_filename = filename  # store the filename of the best match

# Display the best match
best_match_image = socofing_dataset[best_match_index][1]
cv2.imshow('Best match', best_match_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the filename of the best match
print(f'Best match: {best_match_filename}')
print("Score: " + str(best_match_score))