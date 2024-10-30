import detection_polymer


import time
import cv2
import os
import numpy as np
import pywt
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from math import sqrt
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt
from scipy.fft import dct, idct
import math
from cv2 import resize
from sklearn.metrics import roc_curve, auc

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

def jpeg_compression(img, QF):
    import cv2
    cv2.imwrite('tmp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), QF])
    attacked = cv2.imread('tmp.jpg', 0)
    os.remove('tmp.jpg')
    return attacked


def blur(img, sigma):
    attacked = gaussian_filter(img, sigma)
    return attacked

print(detection_polymer.similarity(np.load('polymer.npy'), np.load('utilities/watermark.npy')))
decision, wpsr_mes = detection_polymer.detection('../sample_images/0001.bmp', '../watermarked_images/polymer_0001.bmp', '../watermarked_images/test_0001.bmp')
print('decision not watermarked: ', decision, ', wpsnr: ', wpsr_mes)
