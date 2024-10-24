import numpy as np
import pywt
import cv2
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

# Global parameters
BLOCK_SIZE = 4          # Block size for DWT (Discrete Wavelet Transform)
ALPHA = 4.11            # Scaling factor for watermark embedding
THRESHOLD_TAU = 0.75    # Similarity threshold to determine if an attack was successful
WPSNR_THRESHOLD = 35    # Threshold for wPSNR (Weighted Peak Signal-to-Noise Ratio)

# Predefined spirals used for embedding and extracting the watermark
spiral1 = [(128, 128), (95, 98), (133, 191), (175, 66), (39, 143), (213, 182), (100, 21), (73, 234), (248, 85), (3, 77), (188, 257), (287, 163), (106, 301), (266, 12), (264, 263), (333, 101), (175, 339), (337, 224), (47, 353), (343, 15), (264, 339), (390, 149), (129, 399), (353, 298), (410, 51), (232, 409), (424, 219), (62, 437), (339, 376), (463, 111), (175, 466), (432, 300)]
spiral2 = [(384, 128), (351, 98), (389, 191), (431, 66), (295, 143), (469, 182), (356, 21), (329, 234), (504, 85), (259, 77), (444, 257), (249, 206), (362, 301), (198, 121), (255, 283), (210, 7), (431, 339), (169, 196), (303, 353), (145, 65), (179, 287), (385, 399), (107, 153), (230, 373), (488, 409), (107, 259), (318, 437), (64, 89), (151, 364), (431, 466), (46, 205), (237, 452)]
spiral3 = [(128, 384), (95, 354), (133, 447), (175, 322), (39, 399), (213, 438), (100, 277), (73, 490), (248, 341), (3, 333), (172, 241), (287, 419), (31, 246), (266, 268), (119, 187), (333, 357), (238, 192), (337, 480), (38, 168), (343, 271), (171, 132), (390, 405), (312, 181), (77, 103), (410, 307), (242, 103), (424, 475), (388, 199), (140, 55), (463, 367), (323, 101), (19, 48)]
spiral4 = [(384, 384), (351, 354), (389, 447), (431, 322), (295, 399), (469, 438), (356, 277), (329, 490), (504, 341), (259, 333), (428, 241), (249, 462), (287, 246), (198, 377), (375, 187), (210, 263), (494, 192), (169, 452), (294, 168), (145, 321), (427, 132), (203, 188), (107, 409), (333, 103), (120, 249), (498, 103), (226, 114), (64, 345), (396, 55), (123, 169), (46, 461), (275, 48)]
spiral5 = [(256, 256), (223, 226), (261, 319), (303, 194), (167, 271), (341, 310), (228, 149), (201, 362), (376, 213), (131, 205), (316, 385), (300, 113), (121, 334), (415, 291), (159, 118), (234, 429), (394, 140), (70, 249), (392, 391), (247, 59), (127, 411), (461, 229), (82, 135), (303, 467), (366, 64), (41, 324), (465, 352), (166, 40), (175, 481), (471, 143), (17, 193), (392, 467)]

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
    extracted_watermark = np.zeros(1024)  # Preallocate an array for the extracted watermark

    # Loop through the coordinates in the spiral
    for i, (x, y) in enumerate(coordinates):

        # Extract blocks from the original and watermarked images
        original_block = original_image[x:x + block_size, y:y + block_size]
        watermarked_block = watermarked_image[x:x + block_size, y:y + block_size]

        # Apply Wavelet Transform (DWT) to both blocks
        Coeff_orig = pywt.wavedec2(original_block, wavelet='haar', level=1)
        Coeff_wm = pywt.wavedec2(watermarked_block, wavelet='haar', level=1)

        # Extract the LL sub-band (low-low) from the DWT coefficients
        LL_orig = Coeff_orig[0]
        LL_wm = Coeff_wm[0]

        # Perform SVD (Singular Value Decomposition) on both LL sub-bands
        U_orig, S_orig, V_orig = np.linalg.svd(LL_orig)
        U_wm, S_wm, V_wm = np.linalg.svd(LL_wm)

        # Extract the singular values (watermark) from the watermarked block
        Sw_extracted = (S_wm - S_orig) / ALPHA

        # Insert the extracted watermark into the final watermark array
        idx = (i * LL_orig.shape[0]) % 32  # Determine index in the watermark array
        extracted_watermark[idx:idx + Sw_extracted.shape[0]] = Sw_extracted[:min(block_size, 32 - idx)]
    
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
    Checks if the differences between two images align with a predefined spiral.
    
    Args:
    differences: Coordinates where the images differ.
    spiral: The predefined spiral coordinates.

    Returns:
    True if enough points match the spiral, False otherwise.
    """
    matching_coords = 0
   
    # Count how many of the difference points match spiral points
    for diff in differences:
        if tuple(diff) in spiral:
            matching_coords += 1
    
    # At least half of the spiral points need to match
    return matching_coords >= len(spiral) // 2  

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
    csf = np.genfromtxt('utilities/csf.csv', delimiter=',')
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
    # Calculate the denominator
    denominator = (np.sqrt(np.sum(np.multiply(X, X))) * np.sqrt(np.sum(np.multiply(X_star, X_star))))
    
    # If the denominator is zero, return 0 to avoid division by zero
    if denominator == 0:
        return 0
    
    # Calculate the similarity score
    s = np.sum(np.multiply(X, X_star)) / denominator
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

    # Find the coordinates where the original and watermarked images differ
    differences = find_differences(original_image, watermarked_image)
    
    # Identify which spiral was used based on the differences
    spiral_index = None
    for idx, spiral in enumerate(spirals):
        if check_spiral_for_differences(differences, spiral):
            spiral_index = idx
            break

    # If no matching spiral is found, raise an error
    if spiral_index is None:
        raise ValueError("No matching spiral found.")

    print(f"[SPIRAL CENTER DETECTED]: {spirals[spiral_index][0]}")
    
    used_spiral = spirals[spiral_index]

    # Extract watermark from the watermarked image using the detected spiral
    watermark_extracted_from_watermarked = extract_watermark(original_image, watermarked_image, used_spiral)

    # Extract watermark from the attacked image using the same spiral
    watermark_extracted_from_attacked = extract_watermark(original_image, attacked_image, used_spiral)

    # Calculate the similarity between the two extracted watermarks
    similarity_w = similarity(watermark_extracted_from_watermarked.flatten(), watermark_extracted_from_attacked.flatten())
    
    # Calculate the wPSNR between the watermarked and attacked images
    wpsnr_value = wpsnr(watermarked_image, attacked_image)

    print(f'[SIMILARITY]: {similarity_w:.2f}')

    # Determine if the attack was successful based on similarity and wPSNR
    if (similarity_w < THRESHOLD_TAU) and (wpsnr_value >= WPSNR_THRESHOLD):
        output1 = 0  # Attack failed
        output2 = wpsnr_value
    else:
        output1 = 1  # Attack successful
        output2 = wpsnr_value

    return output1, output2  # Return result of detection and wPSNR
