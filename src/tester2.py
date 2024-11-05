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

def blur(img, sigma):
  from scipy.ndimage.filters import gaussian_filter
  attacked = gaussian_filter(img, sigma)
  return attacked

wat_img = embedding_polymer.embedding('../sample_images/0000.bmp', 'polymer.npy')
print(np.shape(wat_img))
cv2.imwrite('wat_img.bmp' ,wat_img)


decision, wpsr_mes = detection_polymer.detection('../sample_images/0000.bmp', './wat_img.bmp', './wat_img.bmp')
print('decision not attacked: ', decision, ', wpsnr: ', wpsr_mes)

im = cv2.imread('wat_img.bmp', 0)
att_wat_img = jpeg_compression(im, 99)
att_wat_img = blur(att_wat_img, 10)
cv2.imwrite('att_wat_img.bmp' ,att_wat_img)
start = time.time()
decision, wpsr_mes = detection_polymer.detection('../sample_images/0000.bmp', './wat_img.bmp', './att_wat_img.bmp')
print("time: ",time.time() - start)

print('decision  attacked: ', decision, ', wpsnr: ', wpsr_mes)

decision, wpsr_mes = detection_polymer.detection('../sample_images/0000.bmp', './wat_img.bmp', '../sample_images/0000.bmp')
print('decision not watermarked: ', decision, ', wpsnr: ', wpsr_mes)