import numpy as np
import pywt
import cv2
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import sys

# Global parameters
BLOCK_SIZE = 8       # Block size for DWT (Discrete Wavelet Transform)
ALPHA = 18.8         # Scaling factor for watermark embedding

THRESHOLD_TAU = 0.65    # Similarity threshold to determine if an attack was successful
WPSNR_THRESHOLD = 35    # Threshold for wPSNR (Weighted Peak Signal-to-Noise Ratio)

# Predefined spirals used for embedding and extracting the watermark
spiral1 =  [(64, 64), (48, 49), (66, 95), (87, 33), (20, 71), (106, 91), (50, 11), (37, 117), (124, 43), (2, 39), (94, 128), (143, 81), (53, 150), (133, 6), (132, 131), (0, 141), (166, 51), (87, 169), (168, 112), (24, 176), (171, 8), (132, 169), (195, 74), (64, 199), (176, 149), (205, 26), (116, 204), (212, 109), (31, 218), (169, 188), (231, 56), (87, 233)]
spiral2 =  [(192, 64), (176, 49), (194, 95), (215, 33), (148, 71), (234, 91), (178, 11), (165, 117), (130, 39), (222, 128), (125, 103), (181, 150), (99, 61), (128, 141), (105, 4), (215, 169), (85, 98), (152, 176), (73, 33), (90, 143), (192, 199), (54, 76), (115, 186), (244, 204), (54, 129), (159, 218), (32, 45), (76, 182), (215, 233), (23, 102), (119, 226), (22, 4)]
spiral3 =  [(64, 192), (48, 177), (66, 223), (87, 161), (20, 199), (106, 219), (50, 139), (37, 245), (124, 171), (2, 167), (86, 121), (143, 209), (16, 123), (133, 134), (60, 94), (166, 179), (119, 96), (168, 240), (19, 84), (171, 136), (85, 66), (195, 202), (156, 91), (39, 52), (205, 154), (121, 52), (212, 237), (194, 100), (70, 28), (231, 184), (161, 51), (10, 24)]
spiral4 =  [(192, 192), (176, 177), (194, 223), (215, 161), (148, 199), (234, 219), (178, 139), (165, 245), (130, 167), (214, 121), (125, 231), (144, 123), (99, 189), (188, 94), (105, 132), (247, 96), (85, 226), (147, 84), (73, 161), (213, 66), (102, 94), (54, 204), (167, 52), (60, 125), (113, 57), (32, 173), (198, 28), (62, 85), (23, 230), (138, 24), (22, 132), (237, 15)]
spiral5 =  [(128, 128), (112, 113), (130, 159), (151, 97), (84, 135), (170, 155), (114, 75), (101, 181), (188, 107), (66, 103), (158, 192), (150, 57), (61, 167), (207, 145), (80, 59), (117, 214), (197, 70), (35, 125), (196, 195), (124, 30), (64, 205), (230, 115), (41, 68), (151, 233), (183, 32), (21, 162), (232, 176), (83, 20), (88, 240), (235, 72), (9, 97), (196, 233)]

# List of spirals to be checked later for detection
spirals = [spiral1, spiral2, spiral3, spiral4, spiral5]

def extract_watermark(original_image, watermarked_image, coordinates):
    """
    Extracts the watermark from an image using DWT, SVD, and given spiral coordinates.
    
    Args:
    original_image: The original image before watermarking.
    watermarked_image: The watermarked image.
    coordinates: The coordinates of the spiral used for embedding.

    Returns:
    extracted_watermark: The extracted watermark as an array of singular values.
    """

    block_size = BLOCK_SIZE
    alpha = ALPHA

    # Initialize the watermarks
    watermark_1 = np.zeros(1024, dtype=np.float64)
    watermark_2 = np.zeros(1024, dtype=np.float64)

    # Apply wavelet to the original and watermarked block
    Coefficients_ori = pywt.dwt2(original_image, wavelet='haar')
    LL_ori, (LH_ori, HL_ori, HH_ori) = Coefficients_ori
    Coefficients_wm = pywt.dwt2(watermarked_image, wavelet='haar')
    LL_wm, (LH_wm, HL_wm, HH_wm) = Coefficients_wm    

    # Extract watermark 1 from the first 16 positions of the spiral
    for i, (x, y) in enumerate(coordinates[:16]):
        original_block_LH = LH_ori[x:x+block_size, y:y+block_size]
        original_block_HL = HL_ori[x:x+block_size, y:y+block_size]
        original_block_HH = HH_ori[x:x+block_size, y:y+block_size]

        watermarked_block_LH = LH_wm[x:x+block_size, y:y+block_size]
        watermarked_block_HL = HL_wm[x:x+block_size, y:y+block_size]
        watermarked_block_HH = HH_wm[x:x+block_size, y:y+block_size]        

        # Calculate the normalized difference for each sub-block
        diff_LH = (abs((watermarked_block_LH - original_block_LH) / alpha)).flatten()
        diff_HL = (abs((watermarked_block_HL - original_block_HL) / alpha)).flatten()
        diff_HH = (abs((watermarked_block_HH - original_block_HH) / alpha)).flatten()

        # Average the differences to obtain watermark 1
        watermark_1[i*64:(i+1)*64] = np.clip(np.mean([diff_LH, diff_HL, diff_HH], axis=0), 0, 1)

    # Extract watermark 2 from the last 16 positions of the spiral
    for i, (x, y) in enumerate(coordinates[16:32]):
        original_block_LH = LH_ori[x:x+block_size, y:y+block_size]
        original_block_HL = HL_ori[x:x+block_size, y:y+block_size]
        original_block_HH = HH_ori[x:x+block_size, y:y+block_size]

        watermarked_block_LH = LH_wm[x:x+block_size, y:y+block_size]
        watermarked_block_HL = HL_wm[x:x+block_size, y:y+block_size]
        watermarked_block_HH = HH_wm[x:x+block_size, y:y+block_size]        

        # Calculate the normalized difference for each sub-block
        diff_LH = (abs((watermarked_block_LH - original_block_LH) / alpha)).flatten()
        diff_HL = (abs((watermarked_block_HL - original_block_HL) / alpha)).flatten()
        diff_HH = (abs((watermarked_block_HH - original_block_HH) / alpha)).flatten()

        # Average the differences to obtain watermark 2
        watermark_2[i*64:(i+1)*64] = np.clip(np.mean([diff_LH, diff_HL, diff_HH], axis=0), 0, 1)

    # Average the two watermarks to obtain the final watermark
    extracted_watermark = (watermark_1 + watermark_2) / 2

    return extracted_watermark


def find_differences(image1, image2):
    """
    Finds the coordinates where two images differ.

    Args:
    image1: The first image.
    image2: The second image.

    Returns:
    differences: An array of coordinates where the two images differ.
    """
    differences = np.argwhere(image1 != image2)
    return differences

def check_spiral_for_differences(differences, spiral):
    """
    Return number of spiral point equal to the predefinited spiral
    
    Args:
    differences: Coordinates where the images differ.
    spiral: The predefined spiral coordinates.

    Returns:
    Number of matching coordinates
    """
    matching_coords = 0
    # Count how many of the difference points match spiral points
    for diff in differences:
        if tuple(diff) in spiral:
            matching_coords += 1
    
    return matching_coords

def wpsnr(img1, img2):
    """
    Calculates the Weighted Peak Signal-to-Noise Ratio (wPSNR) between two images.
    
    Args:
    img1: The first image.
    img2: The second image.

    Returns:
    The wPSNR value between the two images.
    """
    img1 = np.float32(img1) / 255.0  
    img2 = np.float32(img2) / 255.0  

    difference = img1 - img2
    
    # If there is no difference, return an arbitrarily high wPSNR value
    same = not np.any(difference)
    if same:
        return 9999999

    # Assuming a CSF (Contrast Sensitivity Function) is saved as a CSV file
    csf = np.genfromtxt('csf.csv', delimiter=',')
    ew = convolve2d(difference, np.rot90(csf, 2), mode='valid')

    decibels = 20.0 * np.log10(1.0 / np.sqrt(np.mean(np.mean(ew ** 2))))
    return decibels

def similarity(X, X_star):
    """
    Computes the similarity measure between the original and the new watermarks.
    Returns 0 if the denominator is zero to avoid NaN values.
    
    Args:
    X: The original watermark.
    X_star: The extracted watermark from the attacked image.

    Returns:
    The similarity score.
    """
    
    # Computes the similarity measure between the original and the new watermarks.
    norm_X = np.sqrt(np.sum(np.multiply(X, X)))
    norm_X_star = np.sqrt(np.sum(np.multiply(X_star, X_star)))

    if norm_X == 0 or norm_X_star == 0:
        return 0.0

    s = np.sum(np.multiply(X, X_star)) / (norm_X * norm_X_star)
    return s

def detection(original_image_path, watermarked_image_path, attacked_image_path):
    """
    Compares the extracted watermarks from the watermarked and attacked images.
    
    Args:
    original_image_path: Path to the original image.
    watermarked_image_path: Path to the watermarked image.
    attacked_image_path: Path to the attacked image.

    Returns:
    A tuple (output1, output2) indicating whether the attack was successful and the wPSNR value.
    """
    original_image = cv2.imread(original_image_path, 0)  # Load original image
    watermarked_image = cv2.imread(watermarked_image_path, 0)  # Load watermarked image
    attacked_image = cv2.imread(attacked_image_path, 0)  # Load attacked image

    Coefficients_ori = pywt.dwt2(original_image, wavelet='haar')
    LL_ori, (LH_ori, HL_ori, HH_ori) = Coefficients_ori
    Coefficients_wm = pywt.dwt2(watermarked_image, wavelet='haar')
    LL_wm, (LH_wm, HL_wm, HH_wm) = Coefficients_wm

    # Find the coordinates where the original and watermarked images differ
    differences = find_differences(LH_ori, LH_wm)

    # Identify which spiral has the maximum matching points based on the differences
    max_matching_points = 0
    spiral_index = None

    for idx, spiral in enumerate(spirals):
        matching_points = check_spiral_for_differences(differences, spiral)
        
        if matching_points > max_matching_points:
            max_matching_points = matching_points
            spiral_index = idx

    # Raise an error if no matching spiral is found
    if spiral_index is None:
        output1 = 0
        return output1, wpsnr(watermarked_image, attacked_image)
    
    used_spiral = spirals[spiral_index]

    # Extract watermark from the watermarked image using the detected spiral
    watermark_extracted_from_watermarked = extract_watermark(original_image, watermarked_image, used_spiral)
    
    #print("Similarity between original w and extracted w from watermarked: ", similarity(watermark_extracted_from_watermarked, np.load('polymer.npy')))

    # Extract watermark from the attacked image using the same spiral
    watermark_extracted_from_attacked = extract_watermark(original_image, attacked_image, used_spiral)

    # Calculate the similarity between the two extracted watermarks
    similarity_w = similarity(watermark_extracted_from_watermarked, watermark_extracted_from_attacked)
    #print("Similarity between original w and extracted w from attacked: ", similarity(watermark_extracted_from_attacked, watermark_extracted_from_watermarked))


    # Calculate the wPSNR between the watermarked and attacked images
    wpsnr_value = wpsnr(watermarked_image, attacked_image)

    # Determine if the attack was successful based on similarity and wPSNR

    if(similarity_w >= THRESHOLD_TAU):
        output1 = 1 # watermark found
    else:
        output1 = 0 # watermark not found

    return output1, wpsnr_value # Return result of detection and wPSNR
