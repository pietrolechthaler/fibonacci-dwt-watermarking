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


watermarked = embedding_polymer.embedding('../sample_images/0000.bmp', 'polymer.npy')
cv2.imwrite('watermarkedA.bmp', watermarked)

watermarked = embedding_polymer.embedding('../sample_images/0000.bmp', 'utilities/watermark.npy')
cv2.imwrite('watermarkedB.bmp', watermarked)

#attack
# attacked = jpeg_compression(watermarked, 99)
# cv2.imwrite('attacked.bmp', attacked)
# plt.imshow(attacked)
# plt.show()
# #
# start = time.time()
# dec, wpsnr = detection_howimetyourmark.detection('../sample_images/0000.bmp', 'watermarked.bmp', 'attacked.bmp')
# print('time consumed: ', time.time() - start)

# print(dec)
# print(wpsnr)
