import numpy as np
import cv2
import pywt
from scipy.signal import convolve2d
from math import sqrt

BLOCK_SIZE = 4          # Size of the blocks used for embedding the watermark
ALPHA = 4.11            # Scaling factor for embedding the watermark (controls intensity)
THRESHOLD_TAU = 0.75    # Similarity threshold for watermark detection (τ)
WPSNR_THRESHOLD = 35    # WPSNR threshold in dB

# Spirals previously used in embedding
spiral1 = [(128, 128), (95, 98), (133, 191), (175, 66), (39, 143), (213, 182), (100, 21), (73, 234), (248, 85), (3, 77), (188, 257), (287, 163), (106, 301), (266, 12), (264, 263), (333, 101), (175, 339), (337, 224), (47, 353), (343, 15), (264, 339), (390, 149), (129, 399), (353, 298), (410, 51), (232, 409), (424, 219), (62, 437), (339, 376), (463, 111), (175, 466), (432, 300)]
spiral2 = [(384, 128), (351, 98), (389, 191), (431, 66), (295, 143), (469, 182), (356, 21), (329, 234), (504, 85), (259, 77), (444, 257), (249, 206), (362, 301), (198, 121), (255, 283), (210, 7), (431, 339), (169, 196), (303, 353), (145, 65), (179, 287), (385, 399), (107, 153), (230, 373), (488, 409), (107, 259), (318, 437), (64, 89), (151, 364), (431, 466), (46, 205), (237, 452)]
spiral3 = [(128, 384), (95, 354), (133, 447), (175, 322), (39, 399), (213, 438), (100, 277), (73, 490), (248, 341), (3, 333), (172, 241), (287, 419), (31, 246), (266, 268), (119, 187), (333, 357), (238, 192), (337, 480), (38, 168), (343, 271), (171, 132), (390, 405), (312, 181), (77, 103), (410, 307), (242, 103), (424, 475), (388, 199), (140, 55), (463, 367), (323, 101), (19, 48)]
spiral4 = [(384, 384), (351, 354), (389, 447), (431, 322), (295, 399), (469, 438), (356, 277), (329, 490), (504, 341), (259, 333), (428, 241), (249, 462), (287, 246), (198, 377), (375, 187), (210, 263), (494, 192), (169, 452), (294, 168), (145, 321), (427, 132), (203, 188), (107, 409), (333, 103), (120, 249), (498, 103), (226, 114), (64, 345), (396, 55), (123, 169), (46, 461), (275, 48)]
spiral5 = [(256, 256), (223, 226), (261, 319), (303, 194), (167, 271), (341, 310), (228, 149), (201, 362), (376, 213), (131, 205), (316, 385), (300, 113), (121, 334), (415, 291), (159, 118), (234, 429), (394, 140), (70, 249), (392, 391), (247, 59), (127, 411), (461, 229), (82, 135), (303, 467), (366, 64), (41, 324), (465, 352), (166, 40), (175, 481), (471, 143), (17, 193), (392, 467)]

SPIRALS = [spiral1, spiral2, spiral3, spiral4, spiral5]

def wpsnr(img1, img2):
    """
    Calculates the Weighted Peak Signal-to-Noise Ratio (wPSNR) between two images.
    :param img1: The first image (reference).
    :param img2: The second image (to compare).
    :return: The wPSNR value in dB.
    """
    img1 = np.float32(img1) / 255.0
    img2 = np.float32(img2) / 255.0

    difference = img1 - img2
    same = not np.any(difference)
    if same:
        return 9999999  # If the images are identical, return a very high wPSNR value

    csf = np.genfromtxt('utilities/csf.csv', delimiter=',')  # Contrast Sensitivity Function (CSF) from a CSV file
    ew = convolve2d(difference, np.rot90(csf, 2), mode='valid')

    decibels = 20.0 * np.log10(1.0 / sqrt(np.mean(np.mean(ew ** 2))))
    return decibels

def extract_watermark(image, spiral_points, block_size=BLOCK_SIZE, alpha=ALPHA):
    """
    Extracts the watermark from a given image using predefined spiral points.
    :param image: The watermarked or attacked image.
    :param spiral_points: The list of points where the watermark is embedded.
    :return: Extracted watermark as a 32x32 array.
    """
    watermark_extracted = np.zeros((32, 32))

    for i, (x, y) in enumerate(spiral_points):
        if x + block_size > image.shape[1] or y + block_size > image.shape[0]:
            continue

        block = image[x:x + block_size, y:y + block_size]
        Coefficients = pywt.wavedec2(block, wavelet='haar', level=1)
        LL_tmp = Coefficients[0]

        Uc, Sc, Vc = np.linalg.svd(LL_tmp)

        # Ensure the index is within bounds of Sc
        extracted_value = Sc[i % len(Sc)] / alpha  # Ensure index is within the size of Sc
        watermark_extracted[i // 32, i % 32] = extracted_value

    return watermark_extracted

def detection(input1, input2, input3):
    """
    Detects whether the attacked image contains the watermark and calculates the WPSNR.
    :param input1: The name of the original image.
    :param input2: The name of the watermarked image.
    :param input3: The name of the attacked image.
    :return: output1 (1 if watermark is detected, 0 if not), output2 (WPSNR value).
    """
    # Load images
    original_image = cv2.imread(input1, 0)
    watermarked_image = cv2.imread(input2, 0)
    attacked_image = cv2.imread(input3, 0)

    best_similarity = -1
    best_spiral = None

    # Iterate over all 5 possible spirals to find the best match
    for spiral_points in SPIRALS:
        # Extract watermarks from the watermarked and attacked images using the current spiral
        watermark_extracted_from_watermarked = extract_watermark(watermarked_image, spiral_points)
        watermark_extracted_from_attacked = extract_watermark(attacked_image, spiral_points)

        # Calculate the similarity (normalized correlation) between the two watermarks
        similarity = np.sum(np.multiply(watermark_extracted_from_watermarked, watermark_extracted_from_attacked)) / np.sqrt(
            np.sum(watermark_extracted_from_watermarked ** 2) * np.sum(watermark_extracted_from_attacked ** 2))

        # Check if this spiral gives a higher similarity
        if similarity > best_similarity:
            best_similarity = similarity
            best_spiral = spiral_points

    # Calculate the WPSNR between the watermarked and attacked images
    wpsnr_value = wpsnr(watermarked_image, attacked_image)

    # Determine if the attack was successful
    if best_similarity >= THRESHOLD_TAU:
        output1 = 1  # Watermark is detected
    else:
        output1 = 0  # Watermark is destroyed

    output2 = wpsnr_value

    # Return detection result and WPSNR value
    return output1, output2
