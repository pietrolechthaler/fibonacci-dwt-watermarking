import embedding_polymer, detection_test_polymer
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

# Folder containing the images
image_folder = '../sample_images/'

# Loop through all images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith('.bmp'):
        image_path = os.path.join(image_folder, filename)
        print(f' --------------  Processing {filename}... --------------')

        watermarked = embedding_polymer.embedding(image_path, 'polymer.npy')
        file_name_without_ext = os.path.splitext(filename)[0]
        output_watermarked_name = f'{file_name_without_ext}_w.bmp'
        cv2.imwrite(output_watermarked_name, watermarked)

        #attack
        attacked = jpeg_compression(watermarked, 30)
        cv2.imwrite('attacked.bmp', attacked)

        #detection
        start = time.time()
        dec, wpsnr = detection_test_polymer.detection(image_path, output_watermarked_name, 'attacked.bmp')
        print(f'[TIME CONSUMED]: {(time.time() - start)} s')
        print(f"[DETECTION fallito = 1 , successo = 0]: {dec}")
        print('[WPSNR DETECTION]: %.2f dB' % wpsnr)


