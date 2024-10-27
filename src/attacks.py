# import detection_gruppoG
# import detection_gruppoH
# import detection_gruppoK.detection
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


'''ATTACKS PARAMETERS'''
# brute force attack
attacks = ["jpeg_compression","awgn", "blur", "sharpening", "median", "resizing"]
#attacks = ["blur", "median", "jpeg_compression"]
#attacks = ["resizing", "median"]
#attacks = ["jpeg_compression"]
#attacks = ["sharpening"]


# setting parameter ranges
# awgn
awgn_std_values = [2.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0]
# awgn_seed_values = []
awgn_mean_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

# jpeg_compression
jpeg_compression_QF_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                              26, 27, 28, 29, 30, 40, 50, 55, 60, 65, 70]

# blur
blur_sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                     1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                     2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                     3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                     [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
                     [2, 1], [2, 2], [2, 3], [2, 4], [2, 5],
                     [3, 1], [3, 2], [3, 3], [3, 4], [3, 5],
                     [4, 1], [4, 2], [4, 3], [4, 4], [4, 5],
                     [5, 1], [5, 2], [5, 3], [5, 4], [5, 5]
                     ]

# sharpening
sharpening_sigma_values = [0.01, 0.05, 0.1, 0.5, 1, 2, 3, 10, 15, 20, 25, 30]
sharpening_alpha_values = [0.01, 0.05, 0.1, 0.5, 1, 2, 3, 10, 15, 20, 25, 30]

# median
median_kernel_size_values = [[1, 3], [1, 5],
                             [3, 1], [3, 3], [3, 5],
                             [5, 1], [5, 3], [5, 5],
                             [7, 1], [7, 3], [7, 5],
                             [9, 1], [9, 3], [9, 5]]

# resizing
resizing_scale_values = [0.01, 0.05, 0.1, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

'''ATTACKS'''
def jpeg_compression(img, QF):
    import cv2
    cv2.imwrite('tmp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), QF])
    attacked = cv2.imread('tmp.jpg', 0)
    os.remove('tmp.jpg')
    return attacked


def blur(img, sigma):
    attacked = gaussian_filter(img, sigma)
    return attacked


def awgn(img, std, seed):
    mean = 0.0
    # np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked


def sharpening(img, sigma, alpha):
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)
    return attacked


def median(img, kernel_size):
    attacked = medfilt(img, kernel_size)
    return attacked



def resizing(img, scale):
  x, y = img.shape
  _x = int(x*scale)
  _y = int(y*scale)

  attacked = resize(img, (_x, _y))
  attacked = resize(attacked, (x, y))

  return attacked



def plot_attack(original_image, watermarked_image, attacked_image):
    plt.figure(figsize=(15, 6))
    plt.subplot(131)
    plt.title('Original')
    plt.imshow(original_image, cmap='gray')
    plt.subplot(132)
    plt.title('Watermarked')
    plt.imshow(watermarked_image, cmap='gray')
    plt.subplot(133)
    plt.title('Attacked')
    plt.imshow(attacked_image, cmap='gray')
    plt.show()

RESULTS_FOLDER = './results'
def write_attack_log(group_name, image_name, current_attack_params, result, wpsnr):
    with open(f'{RESULTS_FOLDER}/{group_name}/{group_name}_{image_name}.csv', 'a+') as f:
        f.write(f'{result},{wpsnr},{str(current_attack_params)}\n')


def bf_attack(original_image_path, watermarked_image_path, group_name, image_name):

    watermarked_image = cv2.imread(watermarked_image_path, 0)

    group_results_folder = f'{RESULTS_FOLDER}/{group_name}'
    if not os.path.exists(group_results_folder):
        os.makedirs(group_results_folder)

    current_best_wpsnr = 0

    for attack in attacks:
        ########## JPEG ##########
        if attack == 'jpeg_compression':
            for QF_value in reversed(jpeg_compression_QF_values):
                watermarked_to_attack = watermarked_image.copy()
                
                attacked_image = jpeg_compression(watermarked_to_attack, QF_value)
                cv2.imwrite('attack_tmp.bmp', attacked_image)

                ##################### detection #####################

                if group_name == 'gruppoA':
                    watermark_status, tmp_wpsnr = detection_gruppoA.detection(original_image_path, watermarked_image_path, 'attack_tmp.bmp')
                elif group_name == 'polymer':
                    watermark_status, tmp_wpsnr = detection_polymer.detection(original_image_path, watermarked_image_path, 'attack_tmp.bmp')
                else:
                    print('Wrong group name passed to bf_attack!')
                    return
              
                current_attack = {}
                current_attack["Attack_name"] = 'JPEG_Compression'
                current_attack["QF"] = QF_value
                current_attack["WPSNR"] = tmp_wpsnr
                current_attack["WS"] = watermark_status
                result = None
                if watermark_status == 0:
                    # watermark destroyed
                    if tmp_wpsnr >= 35.0:
                        # image quality not compromised, succesful attack
                        if tmp_wpsnr > current_best_wpsnr:
                            current_best_wpsnr = tmp_wpsnr
                        print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack),' - SUCCESS')
                        print('\n')
                        result = 'SUCCESS'
                        status = cv2.imwrite(RESULTS_FOLDER + '/' + group_name + '/' + str(tmp_wpsnr) + '_'+ attack + '_polymer_'+ group_name + '_' + image_name +'.bmp', attacked_image)
                        if status == False:
                            print("Wrong group name")
                    else:
                        # image quality compromised, failed attack
                        print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack), ' - FAILED')
                        print('\n')
                        result = 'FAILED'
                else:
                    # watermark not destroyed, failed attack
                    print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack), ' - FAILED')
                    print('\n')
                    result = 'FAILED'
                write_attack_log(group_name, image_name, current_attack, result, tmp_wpsnr)

        ########## BLUR ##########
        if attack == 'blur':
            for sigma_value in blur_sigma_values:
                watermarked_to_attack = watermarked_image.copy()
                attacked_image = blur(watermarked_to_attack, sigma_value)

                cv2.imwrite('attack_tmp.bmp', attacked_image)

                ##################### detection #####################

                if group_name == 'gruppoA':
                    watermark_status, tmp_wpsnr = detection_gruppoA.detection(original_image_path, watermarked_image_path, 'attack_tmp.bmp')
                elif group_name == 'polymer':
                    watermark_status, tmp_wpsnr = detection_polymer.detection(original_image_path, watermarked_image_path, 'attack_tmp.bmp')
                else:
                    print('Wrong group name passed to bf_attack!')
                    return
              
                current_attack = {}
                current_attack["Attack_name"] = 'blur'
                current_attack["sigma"] = sigma_value
                current_attack["WPSNR"] = tmp_wpsnr
                current_attack["WS"] = watermark_status
                result = None
                if watermark_status == 0:
                    # watermark destroyed
                    if tmp_wpsnr >= 35.0:
                        # image quality not compromised, succesful attack
                        if tmp_wpsnr > current_best_wpsnr:
                            current_best_wpsnr = tmp_wpsnr
                        print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack),' - SUCCESS')
                        print('\n')
                        result = 'SUCCESS'

                        status = cv2.imwrite(RESULTS_FOLDER + '/' + group_name + '/' + str(tmp_wpsnr) + '_'+ attack + '_polymer_'+ group_name + '_' + image_name +'.bmp', attacked_image)
                        if status == False:
                            print("Wrong group name")
                    else:
                        # image quality compromised, failed attack
                        print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack), ' - FAILED')
                        print('\n')
                        result = 'FAILED'
                else:
                    # watermark not destroyed, failed attack
                    print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack), ' - FAILED')
                    print('\n')
                    result = 'FAILED'
                write_attack_log(group_name, image_name, current_attack, result, tmp_wpsnr)

        ########## AWGN ##########
        if attack == 'awgn':
            for std_value in awgn_std_values:
                for mean_value in awgn_mean_values:
                    watermarked_to_attack = watermarked_image.copy()
                    attacked_image = awgn(watermarked_to_attack, std_value, mean_value)

                    cv2.imwrite('attack_tmp.bmp', attacked_image)

                    ##################### detection #####################

                    if group_name == 'gruppoA':
                        watermark_status, tmp_wpsnr = detection_gruppoA.detection(original_image_path, watermarked_image_path, 'attack_tmp.bmp')
                    elif group_name == 'polymer':
                        watermark_status, tmp_wpsnr = detection_polymer.detection(original_image_path, watermarked_image_path, 'attack_tmp.bmp')
                    else:
                        print('Wrong group name passed to bf_attack!')
                        return
                
                    current_attack = {}
                    current_attack["Attack_name"] = 'awgn'
                    current_attack["std"] = std_value
                    current_attack["mean"] = mean_value
                    current_attack["WPSNR"] = tmp_wpsnr
                    current_attack["WS"] = watermark_status
                    result = None
                    if watermark_status == 0:
                        # watermark destroyed
                        if tmp_wpsnr >= 35.0:
                            # image quality not compromised, succesful attack
                            if tmp_wpsnr > current_best_wpsnr:
                                current_best_wpsnr = tmp_wpsnr
                            print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack),' - SUCCESS')
                            print('\n')
                            result = 'SUCCESS'
                            status = cv2.imwrite(RESULTS_FOLDER + '/' + group_name + '/' + str(tmp_wpsnr) + '_'+ attack + '_polymer_'+ group_name + '_' + image_name +'.bmp', attacked_image)
                            if status == False:
                                print("Wrong group name")

                        else:
                            # image quality compromised, failed attack
                            print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack), ' - FAILED')
                            print('\n')
                            result = 'FAILED'
                    else:
                        # watermark not destroyed, failed attack
                        print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack), ' - FAILED')
                        print('\n')
                        result = 'FAILED'
                    write_attack_log(group_name, image_name, current_attack, result, tmp_wpsnr)

        ########## SHARPENING ##########
        if attack == 'sharpening':
            for sigma_value in sharpening_sigma_values:
                for alpha_value in sharpening_alpha_values:
                    watermarked_to_attack = watermarked_image.copy()
                    attacked_image = sharpening(watermarked_to_attack, sigma_value, alpha_value)

                    cv2.imwrite('attack_tmp.bmp', attacked_image)

                    ##################### detection #####################

                    if group_name == 'gruppoA':
                        watermark_status, tmp_wpsnr = detection_gruppoA.detection(original_image_path, watermarked_image_path, 'attack_tmp.bmp')
                    elif group_name == 'polymer':
                        watermark_status, tmp_wpsnr = detection_polymer.detection(original_image_path, watermarked_image_path, 'attack_tmp.bmp')
                    else:
                        print('Wrong group name passed to bf_attack!')
                        return
                
                    current_attack = {}
                    current_attack["Attack_name"] = 'Sharpening'
                    current_attack["sigma"] = sigma_value
                    current_attack["alpha"] = alpha_value
                    current_attack["WPSNR"] = tmp_wpsnr
                    current_attack["WS"] = watermark_status
                    result = None
                    if watermark_status == 0:
                        # watermark destroyed
                        if tmp_wpsnr >= 35.0:
                            # image quality not compromised, succesful attack
                            if tmp_wpsnr > current_best_wpsnr:
                                current_best_wpsnr = tmp_wpsnr
                            print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack),' - SUCCESS')
                            print('\n')
                            result = 'SUCCESS'
                            status = cv2.imwrite(RESULTS_FOLDER + '/' + group_name + '/' + str(tmp_wpsnr) + '_'+ attack + '_polymer_'+ group_name + '_' + image_name +'.bmp', attacked_image)
                            if status == False:
                                print("Wrong group name")

                        else:
                            # image quality compromised, failed attack
                            print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack), ' - FAILED')
                            print('\n')
                            result = 'FAILED'
                    else:
                        # watermark not destroyed, failed attack
                        print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack), ' - FAILED')
                        print('\n')
                        result = 'FAILED'
                    write_attack_log(group_name, image_name, current_attack, result, tmp_wpsnr)

        ########## MEDIAN ##########
        if attack == 'median':
            for kernel_size_value in median_kernel_size_values:
                watermarked_to_attack = watermarked_image.copy()
                attacked_image = median(watermarked_to_attack, kernel_size_value)

                cv2.imwrite('attack_tmp.bmp', attacked_image)

                ##################### detection #####################

                if group_name == 'gruppoA':
                    watermark_status, tmp_wpsnr = detection_gruppoA.detection(original_image_path, watermarked_image_path, 'attack_tmp.bmp')
                elif group_name == 'polymer':
                    watermark_status, tmp_wpsnr = detection_polymer.detection(original_image_path, watermarked_image_path, 'attack_tmp.bmp')
                else:
                    print('Wrong group name passed to bf_attack!')
                    return
              
                current_attack = {}
                current_attack["Attack_name"] = 'median'
                current_attack["kernel_size_value"] = kernel_size_value
                current_attack["WPSNR"] = tmp_wpsnr
                current_attack["WS"] = watermark_status
                result = None
                if watermark_status == 0:           
                    # watermark destroyed
                    if tmp_wpsnr >= 35.0:
                        # image quality not compromised, succesful attack
                        if tmp_wpsnr > current_best_wpsnr:
                            current_best_wpsnr = tmp_wpsnr
                        print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack),' - SUCCESS')
                        print('\n')
                        result = 'SUCCESS'
                        status = cv2.imwrite(RESULTS_FOLDER + '/' + group_name + '/' + str(tmp_wpsnr) + '_'+ attack + '_polymer_'+ group_name + '_' + image_name +'.bmp', attacked_image)
                        if status == False:
                            print("Wrong group name")

                    else:
                        # image quality compromised, failed attack
                        print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack), ' - FAILED')
                        print('\n')
                        result = 'FAILED'
                else:
                    # watermark not destroyed, failed attack
                    print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack), ' - FAILED')
                    print('\n')
                    result = 'FAILED'
                write_attack_log(group_name, image_name, current_attack, result, tmp_wpsnr)

        ########## RESIZING ##########
        if attack == 'resizing':
            for scale_value in resizing_scale_values:

                watermarked_to_attack = watermarked_image.copy()
                attacked_image = resizing(watermarked_to_attack, scale_value)
                
                cv2.imwrite('attack_tmp.bmp', attacked_image)

                ##################### detection #####################

                if group_name == 'gruppoA':
                    watermark_status, tmp_wpsnr = detection_gruppoA.detection(original_image_path, watermarked_image_path, 'attack_tmp.bmp')
                elif group_name == 'polymer':
                    watermark_status, tmp_wpsnr = detection_polymer.detection(original_image_path, watermarked_image_path, 'attack_tmp.bmp')
                else:
                    print('Wrong group name passed to bf_attack!')
                    return
              
                current_attack = {}
                current_attack["Attack_name"] = 'resizing'
                current_attack["scale"] = scale_value
                current_attack["WPSNR"] = tmp_wpsnr
                current_attack["WS"] = watermark_status
                result = None
                if watermark_status == 0:
                    # watermark destroyed
                    if tmp_wpsnr >= 35.0:
                        # image quality not compromised, succesful attack
                        if tmp_wpsnr > current_best_wpsnr:
                            current_best_wpsnr = tmp_wpsnr
                        print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack),' - SUCCESS')
                        print('\n')
                        result = 'SUCCESS'
                        status = cv2.imwrite(RESULTS_FOLDER + '/' + group_name + '/' + str(tmp_wpsnr) + '_'+ attack + '_polymer_'+ group_name + '_' + image_name +'.bmp', attacked_image)
                        if status == False:
                            print("Wrong group name")

                    else:
                        # image quality compromised, failed attack
                        print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack), ' - FAILED')
                        print('\n')
                        result = 'FAILED'
                else:
                    # watermark not destroyed, failed attack
                    print(image_name, ';', group_name, ';', tmp_wpsnr, ';', str(current_attack), ' - FAILED')
                    print('\n')
                    result = 'FAILED'
                write_attack_log(group_name, image_name, current_attack, result, tmp_wpsnr)

import sys
######## 
# The script can be used also specifying path to the original image (or a folder containing original images),
# the path to the watermarked image (or a folder containing watermarked images), watermarked_folder and group_name as command arguments
########
if len(sys.argv) == 4:
    script_name, original_location, watermarked_location, group_name = sys.argv


    if os.path.isfile(original_location):
        print('[ORIGINAL IMAGE]: ', original_location)
        print('[WATERMARKED IMAGE]: ', watermarked_location)
        print('[GROUP NAME]: ', group_name)
        #Perform attacks on the specified image
        filename = os.path.split(original_location)[1]
        bf_attack(original_location, watermarked_location, group_name, os.path.splitext(filename)[0])
    else:
        original_location = os.path.normpath(original_location)
        watermarked_location = os.path.normpath(watermarked_location)

        print('[ORIGINAL IMAGES FOLDER]: ', original_location)
        print('[WATERMARKED IMAGES FOLDER]: ', watermarked_location)
        print('[GROUP NAME]: ', group_name)
        #List and sort image files in the folder with the .bmp extension
        file_list = sorted([f for f in os.listdir(original_location) if f.endswith('.bmp')])
        #Iterate attacks on all images of the specified folder
        for filename in file_list:
            bf_attack(f'{original_location}/{filename}', f'{watermarked_location}/{group_name}_{filename}', group_name, os.path.splitext(filename)[0])

else:
    image_name_1 = '0000'
    image_name_2 = '0001'
    image_name_3 = '0002'

    original_image_path_1 = "../sample_images/0000.bmp"
    original_image_path_2 = "../sample_images/0001.bmp"
    original_image_path_3 = "../sample_images/0002.bmp"

    watermarked_image_path_1 = "../watermarked_images/polymer_0000.bmp"
    watermarked_image_path_2 = "../watermarked_images/polymer_0001.bmp"
    watermarked_image_path_3 = "../watermarked_images/polymer_0002.bmp"

    watermarked_img = "watermarked.bmp"

    bf_attack(original_image_path_1, watermarked_image_path_1, 'polymer', image_name_1)
    bf_attack(original_image_path_2, watermarked_image_path_2, 'polymer', image_name_2)
    bf_attack(original_image_path_3, watermarked_image_path_3, 'polymer', image_name_3)

'''
##### single attacks 
watermarked = cv2.imread(watermarked_image_path_2)

#attacked = jpeg_compression(watermarked, 99)
#attacked = median(watermarked, 3)
#attacked = blur(watermarked, 5)
#attacked = resizing(watermarked, 0.5)
attacked = sharpening(watermarked, 0.1, 1.5)
#attacked = awgn(watermarked, 0.1)
cv2.imwrite('attacked_single.bmp', attacked)
dec, wpsnr = detection_gruppoD.detection(original_image_path_2,watermarked_image_path_2 , 'attacked_single.bmp')
print(dec, wpsnr)
'''


