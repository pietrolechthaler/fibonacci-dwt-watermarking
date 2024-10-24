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
ALPHA = 4.11            # Scaling factor for embedding the watermark (controls intensity)

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

def jpeg_compression(img, QF):
    """
    Applies JPEG compression to an image.
    
    :param img: The original image.
    :param QF: Quality Factor for the JPEG compression.
    :return: Compressed image.
    """
    # Save the image with temporary JPEG compression
    cv2.imwrite('tmp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), QF])
    
    # Reload the compressed image
    attacked = cv2.imread('tmp.jpg', 0)
    
    # Remove the temporary file
    os.remove('tmp.jpg')
    
    return attacked

def blur(img, sigma):
    """
    Applies Gaussian blur to the image.
    
    :param img: The original image.
    :param sigma: The intensity of the blur.
    :return: Blurred image.
    """
    attacked = gaussian_filter(img, sigma)
    return attacked

def awgn(img, std, seed):
    """
    Adds white Gaussian noise (AWGN) to the image.
    
    :param img: The original image.
    :param std: Standard deviation of the noise.
    :param seed: Optional seed for noise reproducibility.
    :return: Noisy image.
    """
    mean = 0.0
    # np.random.seed(seed)  # Can be enabled to reproduce the same noise
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)  # Keep image values between 0 and 255
    return attacked

def sharpening(img, sigma, alpha):
    """
    Sharpens the image by using a Gaussian filter to calculate the difference.
    
    :param img: The original image.
    :param sigma: Blur intensity used to calculate the difference.
    :param alpha: Sharpening reinforcement factor.
    :return: Sharpened image.
    """
    filter_blurred_f = gaussian_filter(img, sigma)  # Apply Gaussian blur
    attacked = img + alpha * (img - filter_blurred_f)  # Sharpen the image by adding the difference
    return attacked

def median(img, kernel_size):
    """
    Applies a median filter to the image to reduce noise while preserving edges.
    
    :param img: The original image.
    :param kernel_size: Kernel size of the median filter.
    :return: Filtered image.
    """
    from scipy.signal import medfilt
    attacked = medfilt(img, kernel_size)
    return attacked

def resizing(img, scale):
    """
    Resizes the image and then rescales it back to its original dimensions.
    
    :param img: The original image.
    :param scale: The scaling factor for resizing.
    :return: Resized and rescaled image.
    """
    from skimage.transform import rescale
    x, y = img.shape
    attacked = rescale(img, scale)  # Resize the image
    attacked = rescale(attacked, 1/scale)  # Rescale the image back to original size
    attacked = attacked[:x, :y]  # Keep the original dimensions
    return attacked

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

    # Define centers for the Fibonacci spirals
    centers = [(original_image.shape[1] // 4, original_image.shape[0] // 4),
               (3 * original_image.shape[1] // 4, original_image.shape[0] // 4),
               (original_image.shape[1] // 4, 3 * original_image.shape[0] // 4),
               (3 * original_image.shape[1] // 4, 3 * original_image.shape[0] // 4),
               (original_image.shape[1] // 2, original_image.shape[0] // 2)]

    best_average_wpsnr = -1
    best_watermarked_image = None

    # Iterate over the defined spiral centers
    for center in centers:
        # Generate Fibonacci spiral starting from the current center
        fibonacci_spiral = generate_fibonacci_spiral(n_blocks_to_embed, center, original_image.shape)
        print(f"Length: {len(fibonacci_spiral)} - Center: {fibonacci_spiral[0]}")
      
        # Copy the original image (required for wPSNR calculation)
        watermarked_image = original_image.copy()

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
            Uwm, Swm, Vwm = np.linalg.svd(watermark_to_embed)  # SVD of the watermark
            Sw = Sc + Swm[(i * LL_tmp.shape[0]) % 32] * alpha  # Modify singular values of the block with the watermark's

            # Rebuild the block using Inverse Discrete Wavelet Transform (IDWT)
            LL_new = np.dot(Uc, np.dot(np.diag(Sw), Vc))
            Coefficients[0] = LL_new
            block_new = pywt.waverec2(Coefficients, wavelet='haar')
            watermarked_image[x:x + block_size, y:y + block_size] = block_new

        # Apply different attacks to the watermarked image and calculate the average wPSNR
        attacked_images = [blur(watermarked_image, sigma) for sigma in [0.1, 0.5, 1, 2, [1, 1], [2, 1]]]
        attacked_images += [median(watermarked_image, k) for k in [3, 5, 7, 9, 11]]
        attacked_images += [awgn(watermarked_image, std, 0) for std in [0.1, 0.5, 2, 5, 10]]
        attacked_images += [jpeg_compression(watermarked_image, QF) for QF in [10, 30, 50]]

        # Calculate the wPSNR for each attacked image relative to the original image
        wpsnr_values = [wpsnr(original_image, attacked_image) for attacked_image in attacked_images]

        # Calculate the average wPSNR
        average_wpsnr = np.mean(wpsnr_values)

        # Choose the best spiral based on the average wPSNR
        if average_wpsnr > best_average_wpsnr:
            best_average_wpsnr = average_wpsnr
            best_watermarked_image = watermarked_image
            best_center = center
            best_spiral = fibonacci_spiral

    print('[AFTER ATTACKS] Best Average wPSNR: %.2f dB' % best_average_wpsnr)
    # plt.title('Best Watermarked Image')
    # plt.imshow(best_watermarked_image, cmap='gray')
    # plt.show()
    
    print('[EMBEDDING] wPSNR: %.2f dB' % wpsnr(original_image, best_watermarked_image))
    print('Best spiral centered in: ', best_center)
    print('Best spiral points: \n', fibonacci_spiral)

    return best_watermarked_image
