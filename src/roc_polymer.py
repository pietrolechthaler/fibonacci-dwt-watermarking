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

import embedding_polymer, detection_test_polymer


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
        
        #plot original and watermarked image
        plt.subplot(1, 2, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original image')
        plt.subplot(1, 2, 2)
        plt.imshow(watermarked_image, cmap='gray')
        plt.title('Watermarked image')
        plt.show()

        sample = 0
        while sample <= 9:
            # fakemark is the watermark for H0
            fakemark = np.random.uniform(0.0, 1.0, watermark_size)
            fakemark = np.uint8(np.rint(fakemark))

            # random attack to watermarked image (you can modify it)
            attacked_image = random_attack(watermarked_image)

            # extract attacked watermark
            wat_extracted_attacked = detection_test_polymer.detection(original_image, watermarked_image, attacked_image)

            # compute similarity H1
            scores.append(detection_test_polymer.similarity(watermark, wat_extracted_attacked))
            labels.append(1)
            # compute similarity H0
            scores.append(detection_test_polymer.similarity(fakemark, wat_extracted_attacked))
            labels.append(0)
            sample += 1

    # print the scores and labels
    print('Scores:', scores)
    print('Labels:', labels)


    #compute ROC
    fpr, tpr, tau = roc_curve(np.asarray(labels), np.asarray(scores), drop_intermediate=False)
    #compute AUC
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2

    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    idx_tpr = np.where((fpr-0.05)==min(i for i in (fpr-0.05) if i > 0))
    print('For a FPR approximately equals to 0.05 corresponds a TPR equals to %0.2f' % tpr[idx_tpr[0][0]])
    print('For a FPR approximately equals to 0.05 corresponds a threshold equals to %0.2f' % tau[idx_tpr[0][0]])
    print('Check FPR %0.2f' % fpr[idx_tpr[0][0]])

    # end time
    end = time.time()
    print('[COMPUTE ROC] Time: %0.2f seconds' % (end - start))

compute_roc()

