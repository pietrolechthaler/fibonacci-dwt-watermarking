import numpy as np
import cv2
import os
import pywt
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
from math import sqrt
from matplotlib import pyplot as plt

BLOCK_SIZE = 4
BLOCKS_TO_EMBED = 32
ALPHA = 4.11

def wpsnr(img1, img2):
    img1 = np.float32(img1) / 255.0
    img2 = np.float32(img2) / 255.0
    difference = img1 - img2
    if not np.any(difference):
        return float('inf')  # Use infinity to denote perfect similarity
    csf = np.genfromtxt('utilities/csf.csv', delimiter=',')
    ew = convolve2d(difference, np.rot90(csf, 2), mode='valid')
    decibels = 20.0 * np.log10(1.0 / sqrt(np.mean(np.mean(ew ** 2))))
    return decibels

# Functions for image attacks
def blur(img, sigma):
    return gaussian_filter(img, sigma)

def awgn(img, std, seed=None):
    np.random.seed(seed)
    attacked = img + np.random.normal(0, std, img.shape)
    return np.clip(attacked, 0, 255)

def jpeg_compression(img, QF):
    cv2.imwrite('tmp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), QF])
    attacked = cv2.imread('tmp.jpg', 0)
    os.remove('tmp.jpg')
    return attacked

def generate_fibonacci_spiral(n, center, img_shape):
    """
    The generate_fibonacci_spiral function generates a list of points forming a Fibonacci spiral pattern, 
    centered at a specified point and constrained by the dimensions of an image.

    :param n: The number of points to generate in the Fibonacci spiral (int) equal to the number of subdivisions of the watermark.
    :param center: The center of the spiral as a tuple (x, y), representing pixel coordinates.
    :param img_shape: The shape of the image as a tuple (height, width), which defines the boundaries of the spiral.
    :return: A list of tuples (x, y) representing the pixel coordinates of the 32 points on the Fibonacci spiral, 
             constrained by the image size.
    """

    max_radius = min(img_shape[0], img_shape[1]) / 2
    fibonacci_points = []
    phi = (1 + sqrt(5)) / 2  # Golden ratio

    i = 0
    while len(fibonacci_points) < n:
        theta = i * (2 * np.pi / phi)
        r = sqrt(i / n) * max_radius
        x = int(r * np.cos(theta)) + center[0]
        y = int(r * np.sin(theta)) + center[1]
        
        # Add only points within the image bounds
        if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
            fibonacci_points.append((x, y))
        
        i += 1

    return fibonacci_points

def embedding(original_image_path, watermark_path):
    """
    The embedding function embeds a watermark into the original image using a Fibonacci spiral 
    pattern and wavelet transform techniques. It evaluates the robustness of the watermarked image 
    by applying various attacks and selecting the version with the best average wPSNR.

    :param original_image_path: The file path to the original image (grayscale), expected to be in a format readable by OpenCV.
    :param watermark_path: The file path to the watermark, expected to be a .npy file
    :return: The best watermarked image (numpy array), selected based on its robustness after applying attacks.
    """

    original_image = cv2.imread(original_image_path, 0)
    watermark_to_embed = np.load(watermark_path).reshape(32, 32)


    block_size = BLOCK_SIZE
    n_blocks_to_embed = BLOCKS_TO_EMBED
    alpha = ALPHA

    # Define centers for the Fibonacci spirals
    centers = [(original_image.shape[1] // 4, original_image.shape[0] // 4),
               (3 * original_image.shape[1] // 4, original_image.shape[0] // 4),
               (original_image.shape[1] // 4, 3 * original_image.shape[0] // 4),
               (3 * original_image.shape[1] // 4, 3 * original_image.shape[0] // 4),
               (original_image.shape[1] // 2, original_image.shape[0] // 2)]

    best_average_wpsnr = -1
    best_watermarked_image = None

    for center in centers:
        fibonacci_spiral = generate_fibonacci_spiral(n_blocks_to_embed, center, original_image.shape)
        print(f"Length: {len(fibonacci_spiral)} - Center: {fibonacci_spiral[0]}")
        watermarked_image = original_image.copy()

        for i, (x, y) in enumerate(fibonacci_spiral):
            if x + block_size > original_image.shape[1] or y + block_size > original_image.shape[0]:
                print("Caso {i} - fuori")
                continue
            
            block = original_image[x:x + block_size, y:y + block_size]
            Coefficients = pywt.wavedec2(block, wavelet='haar', level=1)
            LL_tmp = Coefficients[0]
            Uc, Sc, Vc = np.linalg.svd(LL_tmp)

            Uwm, Swm, Vwm = np.linalg.svd(watermark_to_embed)
            Sw = Sc + Swm[(i * LL_tmp.shape[0]) % 32] * alpha

            LL_new = np.dot(Uc, np.dot(np.diag(Sw), Vc))
            Coefficients[0] = LL_new
            block_new = pywt.waverec2(Coefficients, wavelet='haar')
            watermarked_image[x:x + block_size, y:y + block_size] = block_new

        # Evaluate watermarked image against different attacks and compute average wPSNR
        attacked_images = [blur(watermarked_image, sigma) for sigma in [0.1, 0.5, 1, 2]]
        attacked_images += [awgn(watermarked_image, std, 0) for std in [0.1, 0.5, 2]]
        attacked_images += [jpeg_compression(watermarked_image, QF) for QF in [10, 30, 50]]

        wpsnr_values = [wpsnr(original_image, attacked_image) for attacked_image in attacked_images]
        average_wpsnr = np.mean(wpsnr_values)

        if average_wpsnr > best_average_wpsnr:
            best_average_wpsnr = average_wpsnr
            best_watermarked_image = watermarked_image
            best_center = center
            best_spiral = fibonacci_spiral

    print('[AFTER ATTACKS] Best Average wPSNR: %.2f dB' % best_average_wpsnr)
    plt.title('Best Watermarked Image')
    plt.imshow(best_watermarked_image, cmap='gray')
    plt.show()
    
    print('[EMBEDDING] wPSNR: %.2f dB' % wpsnr(original_image, best_watermarked_image))
    print('Best spiral centered in: ', best_center)
    print('Best spiral points: \n', fibonacci_spiral)

    return best_watermarked_image
