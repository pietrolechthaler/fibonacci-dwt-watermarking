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

#watermarked = embedding_polymer.embedding('../sample_images/0001.bmp', 'polymer.npy')
# cv2.imwrite('watermarked.bmp', watermarked)
# #attack
# attacked = jpeg_compression(watermarked, 99)
# cv2.imwrite('attacked.bmp', attacked)
# plt.imshow(attacked)
# plt.show()
# #
# start = time.time()
# dec, wpsnr = detection_polymer.detection('../sample_images/0001.bmp', 'watermarked.bmp', 'attacked.bmp')
# print('time consumed: ', time.time() - start)

# print(dec)
# print(wpsnr)

