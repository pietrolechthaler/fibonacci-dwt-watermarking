import numpy as np
import cv2
import os
import pywt
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
from math import sqrt
from matplotlib import pyplot as plt
import sys

# Global parameters for the watermarking algorithm
BLOCK_SIZE = 8         # Size of the blocks used for embedding the watermark
ALPHA = 18.8             # Scaling factor for embedding the watermark (controls intensity)

BLOCKS_TO_EMBED = 32    # Number of blocks to embed the watermark in

# Predefined spirals used for embedding and extracting the watermark
spiral1  =  [(64, 64), (48, 49), (66, 95), (87, 33), (20, 71), (106, 91), (50, 11), (37, 117), (124, 43), (2, 39), (94, 128), (143, 81), (53, 150), (133, 6), (132, 131), (0, 141), (166, 51), (87, 169), (168, 112), (24, 176), (171, 8), (132, 169), (195, 74), (64, 199), (176, 149), (205, 26), (116, 204), (212, 109), (31, 218), (169, 188), (231, 56), (87, 233)]
spiral2  =  [(192, 64), (176, 49), (194, 95), (215, 33), (148, 71), (234, 91), (178, 11), (165, 117), (130, 39), (222, 128), (125, 103), (181, 150), (99, 61), (128, 141), (105, 4), (215, 169), (85, 98), (152, 176), (73, 33), (90, 143), (192, 199), (54, 76), (115, 186), (244, 204), (54, 129), (159, 218), (32, 45), (76, 182), (215, 233), (23, 102), (119, 226), (22, 4)]
spiral3  =  [(64, 192), (48, 177), (66, 223), (87, 161), (20, 199), (106, 219), (50, 139), (37, 245), (124, 171), (2, 167), (86, 121), (143, 209), (16, 123), (133, 134), (60, 94), (166, 179), (119, 96), (168, 240), (19, 84), (171, 136), (85, 66), (195, 202), (156, 91), (39, 52), (205, 154), (121, 52), (212, 237), (194, 100), (70, 28), (231, 184), (161, 51), (10, 24)]
spiral4  =  [(192, 192), (176, 177), (194, 223), (215, 161), (148, 199), (234, 219), (178, 139), (165, 245), (130, 167), (214, 121), (125, 231), (144, 123), (99, 189), (188, 94), (105, 132), (247, 96), (85, 226), (147, 84), (73, 161), (213, 66), (102, 94), (54, 204), (167, 52), (60, 125), (113, 57), (32, 173), (198, 28), (62, 85), (23, 230), (138, 24), (22, 132), (237, 15)]
spiral5  =  [(128, 128), (112, 113), (130, 159), (151, 97), (84, 135), (170, 155), (114, 75), (101, 181), (188, 107), (66, 103), (158, 192), (150, 57), (61, 167), (207, 145), (80, 59), (117, 214), (197, 70), (35, 125), (196, 195), (124, 30), (64, 205), (230, 115), (41, 68), (151, 233), (183, 32), (21, 162), (232, 176), (83, 20), (88, 240), (235, 72), (9, 97), (196, 233)]
# List of predefined spirals
spirals = [spiral1, spiral2, spiral3, spiral4, spiral5]

# Center coordinates for each predefined spiral
centers = [(64, 64), (192, 64), (64, 192), (192, 192), (128, 128)]

def wpsnr(img1, img2):
    """
    Calculates the Weighted Peak Signal-to-Noise Ratio (wPSNR) between two images,
    weighing the differences using a contrast sensitivity function (CSF).
    
    :param img1: The original image.
    :param img2: The image to compare (potentially attacked).
    :return: The wPSNR value.
    """
    img1 = np.float32(img1) / 255.0  
    img2 = np.float32(img2) / 255.0 
    difference = img1 - img2
    same = not np.any(difference)  # Check if images are identical

    if same:
        return 9999999  # Return a high wPSNR value if identical

    csf = np.genfromtxt('csf.csv', delimiter=',')  # Load CSF matrix
    ew = convolve2d(difference, np.rot90(csf, 2), mode='valid')
    
    # Calculate the wPSNR in decibels
    decibels = 20.0 * np.log10(1.0 / sqrt(np.mean(np.mean(ew ** 2))))
    return decibels

def get_fibonacci_spiral(n, center, img_shape):
    """
    Retrieves a predefined spiral if it matches the center; otherwise, 
    generates a new Fibonacci spiral centered at the specified point.
        
    :param n_points: Number of points to include in the generated spiral.
    :param center: The center point (x, y) for the desired spiral.
    :param img_shape: Shape of the image to define the bounds.
    :return: List of points (x, y) in the spiral.
    """
    # Check for a predefined spiral that matches the center
    for spiral in spirals:
        if spiral[0] == center:
            return spiral
    
    # Generate a new Fibonacci spiral if none matches
    return generate_fibonacci_spiral(n, center, img_shape)


def generate_fibonacci_spiral(n, center, img_shape):
    """
    Generates a Fibonacci spiral to select points for watermark embedding.
    
    :param n: Number of points to generate along the spiral.
    :param center: The center of the spiral (x, y).
    :param img_shape: The shape of the image (height, width).
    :return: List of coordinates (x, y) forming a Fibonacci spiral.
    """
    max_radius = min(img_shape[0], img_shape[1]) / 2  # Max radius for spiral
    fibonacci_points = []
    phi = (1 + sqrt(5)) / 2  # Golden ratio
    
    i = 0
    while len(fibonacci_points) < n:
        theta = i * (2 * np.pi / phi)  # Calculate angle
        r = sqrt(i / n) * max_radius  # Calculate radius
        x = int(r * np.cos(theta)) + center[0]
        y = int(r * np.sin(theta)) + center[1]
        
        # Add only points within the image bounds
        if 0 <= x + BLOCK_SIZE < img_shape[1] and 0 <= y + BLOCK_SIZE < img_shape[0]:
            fibonacci_points.append((x, y))
        
        i += 1

    return fibonacci_points

def embed_watermark(watermark_to_embed, original_image, fibonacci_spiral, block_size, alpha):
    """
    Embeds the watermark into the original image at positions defined by the Fibonacci spiral.
    
    :param watermark_to_embed: 1D array of watermark data.
    :param original_image: Grayscale original image where watermark is embedded.
    :param fibonacci_spiral: List of coordinates for watermark embedding.
    :param block_size: Size of each embedding block.
    :param alpha: Scaling factor for embedding strength.
    :return: The watermarked image.
    """
    # Copy of the original image
    watermarked_image = original_image.copy()

    # Split the watermark into blocks for embedding, each spiral point embeds two blocks
    num_blocks = 16
    
    # Linear dimension for two blocks
    block_length = block_size**2

    # List of 8x8 blocks for the watermark
    watermark_blocks = []
    for i in range(num_blocks):
        block_data = watermark_to_embed[i * block_length: (i + 1) * block_length]
        watermark_block = block_data.reshape(block_size, block_size)
        watermark_blocks.append(watermark_block)

    # Apply wavelet to the original image
    Coefficients = pywt.dwt2(original_image, wavelet='haar')
    LL, (LH, HL, HH) = Coefficients
   
    # Save the sign of the coefficients
    sign_LH, sign_HL, sign_HH = np.sign(LH), np.sign(HL), np.sign(HH)

    # Take the absolute value of the coefficients
    watermarked_LH, watermarked_HL, watermarked_HH = abs(LH), abs(HL), abs(HH)

    #Insert watermark in two blocks at a time
    for i, (x, y) in enumerate(fibonacci_spiral[:BLOCKS_TO_EMBED]):
        block_index = i % num_blocks
        watermark_block = watermark_blocks[block_index]
        watermarked_LH[x:x+block_size, y:y+block_size] += watermark_block * alpha        
        watermarked_HL[x:x+block_size, y:y+block_size] += watermark_block * alpha
        watermarked_HH[x:x+block_size, y:y+block_size] += watermark_block * alpha

    # Restore the sign of the coefficients
    watermarked_LH *= sign_LH
    watermarked_HL *= sign_HL
    watermarked_HH *= sign_HH

    # Inverse wavelet transform
    watermarked_image = pywt.idwt2((LL, (watermarked_LH, watermarked_HL, watermarked_HH)), 'haar')
   
    # Clip the values to the range 0-255
    watermarked_image = np.clip(watermarked_image, 0, 255).round().astype(np.uint8)

    #print('[WPSNR]', wpsnr(original_image, watermarked_image))
    return watermarked_image

def embedding(original_image_path, watermark_path):
    """
    Embeds a watermark into the original image using a Fibonacci spiral
    and wavelet transform techniques. Evaluates robustness by applying attacks 
    and selects the best version based on wPSNR.
    
    :param original_image_path: Path to the original image (grayscale).
    :param watermark_path: Path to the watermark file (in .npy format).
    :return: The best watermarked image based on robustness.
    """
    original_image = cv2.imread(original_image_path, 0)
    watermark_to_embed = np.load(watermark_path)

    # Calculate variance for each image quadrant
    quadrants = [
        (0, 0, 128, 128),       # Top-left
        (384, 0, 512, 128),     # Bottom-left
        (0, 384, 128, 512),     # Top-right
        (384, 384, 512, 512)    # Bottom-right  
    ]   
    variances = [np.var(original_image[q[0]:q[2], q[1]:q[3]]) for q in quadrants]

    # Select quadrant with lowest variance
    worst_quadrant_index = np.argmin(variances)
    opposite_corners = {
        0: centers[3],  # Opposite of top-left is bottom-right
        1: centers[2],  # Opposite of bottom-left is top-right
        2: centers[1],  # Opposite of top-right is bottom-left
        3: centers[0]   # Opposite of bottom-right is top-left
    }
    spiral_center = opposite_corners[worst_quadrant_index]

    # Select Fibonacci spiral based on quadrant
    fibonacci_spiral = get_fibonacci_spiral(BLOCKS_TO_EMBED, spiral_center, original_image.shape)

    # Embed watermark in selected quadrant and center
    watermarked_image_corner = embed_watermark(watermark_to_embed, original_image, fibonacci_spiral, BLOCK_SIZE, ALPHA)
    watermarked_image_center = embed_watermark(watermark_to_embed, original_image, spirals[4], BLOCK_SIZE, ALPHA)
    
    # Compare the two watermarked images using wPSNR
    if wpsnr(original_image, watermarked_image_corner) > wpsnr(original_image, watermarked_image_center):
        #print(wpsnr(original_image, watermarked_image_corner))
        return watermarked_image_corner
    else:
        #print(wpsnr(original_image, watermarked_image_center))
        return watermarked_image_center
