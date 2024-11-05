import time
import random
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import pywt
from scipy.linalg import svd
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
from math import sqrt

import embedding_polymer, detection_polymer

# Predefined spirals used for embedding and extracting the watermark
spiral1 =  [(64, 64), (48, 49), (66, 95), (87, 33), (20, 71), (106, 91), (50, 11), (37, 117), (124, 43), (2, 39), (94, 128), (143, 81), (53, 150), (133, 6), (132, 131), (0, 141), (166, 51), (87, 169), (168, 112), (24, 176), (171, 8), (132, 169), (195, 74), (64, 199), (176, 149), (205, 26), (116, 204), (212, 109), (31, 218), (169, 188), (231, 56), (87, 233)]
spiral2 =  [(192, 64), (176, 49), (194, 95), (215, 33), (148, 71), (234, 91), (178, 11), (165, 117), (130, 39), (222, 128), (125, 103), (181, 150), (99, 61), (128, 141), (105, 4), (215, 169), (85, 98), (152, 176), (73, 33), (90, 143), (192, 199), (54, 76), (115, 186), (244, 204), (54, 129), (159, 218), (32, 45), (76, 182), (215, 233), (23, 102), (119, 226), (22, 4)]
spiral3 =  [(64, 192), (48, 177), (66, 223), (87, 161), (20, 199), (106, 219), (50, 139), (37, 245), (124, 171), (2, 167), (86, 121), (143, 209), (16, 123), (133, 134), (60, 94), (166, 179), (119, 96), (168, 240), (19, 84), (171, 136), (85, 66), (195, 202), (156, 91), (39, 52), (205, 154), (121, 52), (212, 237), (194, 100), (70, 28), (231, 184), (161, 51), (10, 24)]
spiral4 =  [(192, 192), (176, 177), (194, 223), (215, 161), (148, 199), (234, 219), (178, 139), (165, 245), (130, 167), (214, 121), (125, 231), (144, 123), (99, 189), (188, 94), (105, 132), (247, 96), (85, 226), (147, 84), (73, 161), (213, 66), (102, 94), (54, 204), (167, 52), (60, 125), (113, 57), (32, 173), (198, 28), (62, 85), (23, 230), (138, 24), (22, 132), (237, 15)]
spiral5 =  [(128, 128), (112, 113), (130, 159), (151, 97), (84, 135), (170, 155), (114, 75), (101, 181), (188, 107), (66, 103), (158, 192), (150, 57), (61, 167), (207, 145), (80, 59), (117, 214), (197, 70), (35, 125), (196, 195), (124, 30), (64, 205), (230, 115), (41, 68), (151, 233), (183, 32), (21, 162), (232, 176), (83, 20), (88, 240), (235, 72), (9, 97), (196, 233)]

# List of spirals to be checked later for detection
spirals = [spiral1, spiral2, spiral3, spiral4, spiral5]

#this seed was set just to make you obtain the same result
random.seed(3)
def awgn(img, std, seed):
  mean = 0.0   # some constant
  #np.random.seed(seed)
  attacked = img + np.random.normal(mean, std, img.shape)
  attacked = np.clip(attacked, 0, 255)
  return attacked

def blur(img, sigma):
  from scipy.ndimage.filters import gaussian_filter
  attacked = gaussian_filter(img, sigma)
  return attacked

def sharpening(img, sigma, alpha):
  import scipy
  from scipy.ndimage import gaussian_filter
  import matplotlib.pyplot as plt

  #print(img/255)
  filter_blurred_f = gaussian_filter(img, sigma)

  attacked = img + alpha * (img - filter_blurred_f)
  return attacked

def median(img, kernel_size):
  from scipy.signal import medfilt
  attacked = medfilt(img, kernel_size)
  return attacked

def resizing(img, scale):
  from skimage.transform import rescale
  x, y = img.shape
  attacked = rescale(img, scale)
  attacked = rescale(attacked, 1/scale)
  attacked = attacked[:x, :y]
  return attacked

def jpeg_compression(img, QF):
  from PIL import Image
  img = Image.fromarray(img)
  img.save('tmp.jpg',"JPEG", quality=QF)
  attacked = Image.open('tmp.jpg')
  attacked = np.asarray(attacked,dtype=np.uint8)
  os.remove('tmp.jpg')

  return attacked

def random_attack(img):
  i = random.randint(1,7)
  if i==1:
    attacked = awgn(img, 3., 123)
  elif i==2:
    attacked = blur(img, [3, 3])
  elif i==3:
    attacked = sharpening(img, 1, 1)
  elif i==4:
    attacked = median(img, [3, 3])
  elif i==5:
    attacked = resizing(img, 0.8)
  elif i==6:
    attacked = jpeg_compression(img, 75)
  elif i ==7:
     attacked = img
  return attacked


def compute_roc():
    # start time
    start = time.time()
    from sklearn.metrics import roc_curve, auc

    # generate your watermark (if it is necessary)
    watermark_size = 1024
    watermark_path = "polymer.npy"
    watermark = np.load(watermark_path)

    # scores and labels are two lists we will use to append the values of similarity and their labels
    # In scores we will append the similarity between our watermarked image and the attacked one,
    # or  between the attacked watermark and a random watermark
    # In labels we will append the 1 if the scores was computed between the watermarked image and the attacked one,
    # and 0 otherwise
    scores = []
    labels = []

    # Folder containing the images
    image_folder = '../sample_images/'

    file_list = sorted([f for f in os.listdir(image_folder) if f.endswith('.bmp')])
    # Loop through all images in the folder
    for filename in file_list:

        original_image = os.path.join(image_folder, filename)

        watermarked_image = embedding_polymer.embedding(original_image, watermark_path)

        original_image = cv2.imread(original_image, 0)
  
        sample = 0
        while sample <= 100:
            # fakemark is the watermark for H0
            fakemark = np.random.uniform(0.0, 1.0, watermark_size)
            fakemark = np.uint8(np.rint(fakemark))

            # random attack to watermarked image (you can modify it)
            attacked_image = random_attack(watermarked_image)

            Coefficients_ori = pywt.dwt2(original_image, wavelet='haar')
            LL_ori, (LH_ori, HL_ori, HH_ori) = Coefficients_ori
            Coefficients_wm = pywt.dwt2(watermarked_image, wavelet='haar')
            LL_wm, (LH_wm, HL_wm, HH_wm) = Coefficients_wm

            # Find the coordinates where the original and watermarked images differ
            differences = detection_polymer.find_differences(LL_ori, LL_wm)
    
            # Identify which spiral has the maximum matching points based on the differences
            max_matching_points = 0
            spiral_index = None

            for idx, spiral in enumerate(spirals):
                matching_points = detection_polymer.check_spiral_for_differences(differences, spiral)
                
                if matching_points > max_matching_points:
                    max_matching_points = matching_points
                    spiral_index = idx

            # Raise an error if no matching spiral is found
            if spiral_index is None:
                raise ValueError("No matching spiral found.")
            
            used_spiral = spirals[spiral_index]
            wat_extracted_attacked = detection_polymer.extract_watermark(original_image, attacked_image, used_spiral)

            # compute similarity H1
            scores.append(detection_polymer.similarity(watermark, wat_extracted_attacked.flatten()))
            labels.append(1)
            
            # compute similarity H0
            scores.append(detection_polymer.similarity(fakemark, wat_extracted_attacked.flatten()))
            labels.append(0)
            sample += 1

    # print the scores and labels
    print('Scores:', scores)
    print('Labels:', labels)


    #compute ROC
    fpr, tpr, tau = roc_curve(np.asarray(labels), np.asarray(scores), drop_intermediate=False)
    #compute AUC
    roc_auc = auc(fpr, tpr)
    lw = 2

    

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (0 to 1)')
    plt.legend(loc="lower right")
    plt.savefig('roc_full_polymer.png') 
    plt.show()


    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 0.1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (0 to 0.1)')
    plt.legend(loc="lower right")
    plt.savefig('roc_zoomed_polymer.png')  
    plt.show()

    idx_tpr = np.where((fpr-0.05)==min(i for i in (fpr-0.05) if i > 0))
    print('For FPR ≈ 0.05, TPR = %0.2f' % tpr[idx_tpr[0][0]])
    print('For FPR ≈ 0.05, threshold = %0.2f' % tau[idx_tpr[0][0]])

    idx_tpr = np.where((fpr - 0.1) == min(i for i in (fpr - 0.1) if i > 0))
    print('For FPR ≈ 0.1, TPR = %0.2f' % tpr[idx_tpr[0][0]])
    print('For FPR ≈ 0.1, threshold = %0.2f' % tau[idx_tpr[0][0]])

    # end time
    end = time.time()
    print('[COMPUTE ROC] Time: %0.2f seconds' % (end - start))

compute_roc()

