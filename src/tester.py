import embedding_polymer, detection_polymer
import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt

def jpeg_compression(img, QF):
  from PIL import Image
  img = Image.fromarray(img)
  img.save('tmp.jpg',"JPEG", quality=QF)
  attacked = Image.open('tmp.jpg')
  attacked = np.asarray(attacked,dtype=np.uint8)
  os.remove('tmp.jpg')
  return attacked

def sharpening(img, sigma, alpha):
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)
    return attacked

# Folder containing the images
image_folder = '../sample_images/'
GROUP_NAME = 'polymer'
WATERMARKED_FOLDER = '../watermarked_images'

file_list = sorted([f for f in os.listdir(image_folder) if f.endswith('.bmp')])
#limita a 5 immagini
# Loop through all images in the folder
for filename in file_list:
    if filename.endswith('.bmp'):
        image_path = os.path.join(image_folder, filename)
        print(f' --------------  Processing {filename}... --------------')

        watermarked_img = embedding_polymer.embedding(image_path, 'polymer.npy')
        output_image = f'{WATERMARKED_FOLDER}/{GROUP_NAME}_{filename}'
        cv2.imwrite(output_image, watermarked_img)

        attacked = jpeg_compression(watermarked_img, 30)
        cv2.imwrite('attacked.bmp', attacked)

        start = time.time()
        dec, wpsnr = detection_polymer.detection(image_path, watermarked_img, 'attacked.bmp')
        print(f'[TIME CONSUMED]: {(time.time() - start)} s')
        print(f"[DETECTION fallito = 1 , successo = 0]: {dec}")
        print('[WPSNR DETECTION]: %.2f dB' % wpsnr)


