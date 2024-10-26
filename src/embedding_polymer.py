import numpy as np
import cv2
import os
import pywt
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
from math import sqrt
from matplotlib import pyplot as plt

# Global parameters for the watermarking algorithm
BLOCK_SIZE = 4          # Size of the blocks used for embedding the watermark
BLOCKS_TO_EMBED = 32    # Number of blocks to embed the watermark in
ALPHA = 5.70            # Scaling factor for embedding the watermark (controls intensity)

# Predefined spirals used for embedding and extracting the watermark
spiral1 = [(128, 128), (95, 98), (133, 191), (175, 66), (39, 143), (213, 182), (100, 21), (73, 234), (248, 85), (3, 77), (188, 257), (287, 163), (106, 301), (266, 12), (264, 263), (333, 101), (175, 339), (337, 224), (47, 353), (343, 15), (264, 339), (390, 149), (129, 399), (353, 298), (410, 51), (232, 409), (424, 219), (62, 437), (339, 376), (463, 111), (175, 466), (432, 300)]
spiral2 = [(384, 128), (351, 98), (389, 191), (431, 66), (295, 143), (469, 182), (356, 21), (329, 234), (504, 85), (259, 77), (444, 257), (249, 206), (362, 301), (198, 121), (255, 283), (210, 7), (431, 339), (169, 196), (303, 353), (145, 65), (179, 287), (385, 399), (107, 153), (230, 373), (488, 409), (107, 259), (318, 437), (64, 89), (151, 364), (431, 466), (46, 205), (237, 452)]
spiral3 = [(128, 384), (95, 354), (133, 447), (175, 322), (39, 399), (213, 438), (100, 277), (73, 490), (248, 341), (3, 333), (172, 241), (287, 419), (31, 246), (266, 268), (119, 187), (333, 357), (238, 192), (337, 480), (38, 168), (343, 271), (171, 132), (390, 405), (312, 181), (77, 103), (410, 307), (242, 103), (424, 475), (388, 199), (140, 55), (463, 367), (323, 101), (19, 48)]
spiral4 = [(384, 384), (351, 354), (389, 447), (431, 322), (295, 399), (469, 438), (356, 277), (329, 490), (504, 341), (259, 333), (428, 241), (249, 462), (287, 246), (198, 377), (375, 187), (210, 263), (494, 192), (169, 452), (294, 168), (145, 321), (427, 132), (203, 188), (107, 409), (333, 103), (120, 249), (498, 103), (226, 114), (64, 345), (396, 55), (123, 169), (46, 461), (275, 48)]
spiral5 = [(256, 256), (223, 226), (261, 319), (303, 194), (167, 271), (341, 310), (228, 149), (201, 362), (376, 213), (131, 205), (316, 385), (300, 113), (121, 334), (415, 291), (159, 118), (234, 429), (394, 140), (70, 249), (392, 391), (247, 59), (127, 411), (461, 229), (82, 135), (303, 467), (366, 64), (41, 324), (465, 352), (166, 40), (175, 481), (471, 143), (17, 193), (392, 467)]
# List of spirals to be checked later for detection
spirals = [spiral1, spiral2, spiral3, spiral4, spiral5]

centers = [(128, 128), (384, 128), (128, 384), (384, 384), (256, 256)]


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
    same = not np.any(difference)  # Check if the images are identical
    if same is True:
        return 9999999  # Return a high wPSNR value if they are identical

    csf = np.genfromtxt('utilities/csf.csv', delimiter=',')    
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
    # Check if a predefined spiral matches the given center
    for spiral in spirals:
        if spiral[0] == center:
            return spiral
    
    # If no predefined spiral matches, generate a new Fibonacci spiral
    return generate_fibonacci_spiral(n, center, img_shape)


def generate_fibonacci_spiral(n, center, img_shape):
    """
    Generates a Fibonacci spiral to select points for watermark embedding.
    
    :param n: Number of points to generate along the spiral.
    :param center: The center of the spiral (x, y).
    :param img_shape: The shape of the image (height, width).
    :return: List of coordinates (x, y) forming a Fibonacci spiral.
    """
    max_radius = min(img_shape[0], img_shape[1]) / 2  # Maximum radius of the spiral
    fibonacci_points = []
    phi = (1 + sqrt(5)) / 2  # Golden ratio
    
    i = 0
    while len(fibonacci_points) < n:
        theta = i * (2 * np.pi / phi)  # Calculate the angle
        r = sqrt(i / n) * max_radius  # Calculate the radius
        x = int(r * np.cos(theta)) + center[0]
        y = int(r * np.sin(theta)) + center[1]
        
        # Add only points that are within the image bounds
        if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
            fibonacci_points.append((x, y))
        
        i += 1

    return fibonacci_points

def embed_watermark(watermark_to_embed, original_image, fibonacci_spiral, block_size, alpha):

    watermarked_image = original_image.copy()
    divisions = original_image.shape[0] / block_size

    shape_LL_tmp = np.floor(original_image.shape[0]/ (2*divisions))
    shape_LL_tmp = np.uint8(shape_LL_tmp)
    Uwm, Swm, Vwm = np.linalg.svd(watermark_to_embed)  # SVD of the watermark

    # Iterate over the spiral points to embed the watermark
    for i, (x, y) in enumerate(fibonacci_spiral):
        # Check if the block size is within the image boundaries
        if x + block_size > original_image.shape[1] or y + block_size > original_image.shape[0]:
            print(f"Case {i} - out of bounds")
            continue
        
        block = original_image[x:x + block_size, y:y + block_size]
        # Apply Discrete Wavelet Transform (DWT)
        Coefficients = pywt.wavedec2(block, wavelet='haar', level=1)
        # We will embed the watermark in the LL band
        LL_tmp = Coefficients[0]

        # Apply Singular Value Decomposition (SVD)
        Uc, Sc, Vc = np.linalg.svd(LL_tmp)  # SVD of the block
        Sw = Sc + Swm[(i*shape_LL_tmp)%32 : (shape_LL_tmp+(i*shape_LL_tmp)%32)] * alpha  # Modify singular values of the block with the watermark's

        # Rebuild the block using Inverse Discrete Wavelet Transform (IDWT)
        LL_new = np.zeros((shape_LL_tmp, shape_LL_tmp))
        LL_new = np.dot(Uc, np.dot(np.diag(Sw), Vc))
        Coefficients[0] = LL_new
        block_new = pywt.waverec2(Coefficients, wavelet='haar')
        watermarked_image[x:x + block_size, y:y + block_size] = block_new

    return watermarked_image

def embedding(original_image_path, watermark_path):
    """
    Embeds a watermark into the original image using a Fibonacci spiral
    and wavelet transform techniques. It evaluates the robustness of the 
    watermarked image by applying various attacks and selects the version 
    with the best average wPSNR.

    :param original_image_path: Path to the original image (grayscale).
    :param watermark_path: Path to the watermark file (in .npy format).
    :return: The best watermarked image selected based on its robustness.
    """
    original_image = cv2.imread(original_image_path, 0)  # Load the original grayscale image
    watermark_to_embed = np.load(watermark_path).reshape(32, 32)  # Load the watermark

    block_size = BLOCK_SIZE
    n_blocks_to_embed = BLOCKS_TO_EMBED
    alpha = ALPHA

    # Defined centers for the Fibonacci spirals
    centers = [(128, 128), (384, 128), (128, 384), (384, 384), (256, 256)]
    
    quadrants = [
        (0, 0, 128, 128),       # Top-left
        (384, 0, 512, 128),     # Bottom-left
        (0, 384, 128, 512),     # Top-right
        (384, 384, 512, 512)    # Bottom-right  
    ]   

    #calcola varianza dell'immagine in quel quadrante
    #TODO:
    variances = [np.var(original_image[q[0]:q[2],q[1]:q[3]]) for q in quadrants]

    # Select the quadrant with the lower variance
    worst_quadrant_index = np.argmin(variances)
    best_quadrant_index = np.argmax(variances)
    
    # Draw a rectangle on the original image to highlight the lowest variance quadrant
    x1, y1, x2, y2 = quadrants[worst_quadrant_index]
    original_image_annotated = cv2.cvtColor(original_image.copy(), cv2.COLOR_GRAY2RGB)  # Convert to RGB for colored rectangle
    cv2.rectangle(original_image_annotated, (y1, x1), (y2, x2), (255, 0, 0), 2)  # Draw rectangle in red

    spiral_center = None

    #spirale angolo opposto del worst_quadrant_index
    opposite_corners = {
        0: centers[3],  # Opposite of top-left is bottom-right
        1: centers[2],  # Opposite of bottom-left is top-right
        2: centers[1],  # Opposite of top-right is bottom-left
        3: centers[0]   # Opposite of bottom-right is top-left
    }
    spiral_center = opposite_corners[worst_quadrant_index]

    # Use the center point of the selected quadrant for spiral selection
    fibonacci_spiral = get_fibonacci_spiral(n_blocks_to_embed, spiral_center, original_image.shape)

    #Draw each point in the Fibonacci spiral on the annotated image
    # for i, (x, y) in enumerate(fibonacci_spiral):
    #     # Ensure points are within bounds of the image
    #     if 0 <= x < original_image_annotated.shape[1] and 0 <= y < original_image_annotated.shape[0]:
    #         cv2.circle(original_image_annotated, (y, x), radius=2, color=(0, 255, 0), thickness=-1)  # Green dots for spiral points

    # # Display the annotated original image
    # plt.figure()
    # plt.title("Original Image with Lowest Variance Quadrant Highlighted")
    # plt.imshow(original_image_annotated)
    # plt.axis('off')
    # plt.show()
    
    watermarked_image_corner = embed_watermark(watermark_to_embed, original_image, fibonacci_spiral, block_size, alpha)
    watermarked_image_center = embed_watermark(watermark_to_embed, original_image, spirals[4], block_size, alpha)
   
    print('[A] wPSNR: %.2f dB' % wpsnr(original_image, watermarked_image_corner))
    print('[C] wPSNR: %.2f dB' % wpsnr(original_image, watermarked_image_center))

    if(wpsnr(original_image, watermarked_image_corner) > wpsnr(original_image, watermarked_image_center)):
        
        print('[SPIRAL CENTER]: ', fibonacci_spiral[0])
        return watermarked_image_corner
    else:
        print("CASO MEGLIO CENTRO")
        print('[SPIRAL CENTER]: ', spiral5[0])
        return watermarked_image_center

