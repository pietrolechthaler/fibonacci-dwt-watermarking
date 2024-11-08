import embedding_polymer, detection_polymer
import numpy as np
import os
import cv2
import time
import sys
from PIL import Image
from scipy.ndimage import gaussian_filter

def jpeg_compression(img, QF):
    """
    Applies JPEG compression to an image at a specified quality factor (QF).

    Parameters:
        img (numpy.ndarray): The input image to be compressed as a NumPy array.
        QF (int): The JPEG quality factor (between 1 and 100). Lower values mean higher compression.

    Returns:
        numpy.ndarray: The compressed image as a NumPy array.
    """

    img = Image.fromarray(img)
    img.save('tmp.jpg', "JPEG", quality=QF)  
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked, dtype=np.uint8) 
    os.remove('tmp.jpg')  
    return attacked

#Folder containing the images to process
image_folder = '../sample_images/'          

#If specified first argument is the folder containing the images to process
if len(sys.argv) > 1:                        
    image_folder = sys.argv[1] 
    image_folder = os.path.normpath(image_folder)

GROUP_NAME = 'polymer'
WATERMARKED_FOLDER = '../watermarked_images' #Folder containing the processed images

#Ensure the watermarked images folder exists
os.makedirs(WATERMARKED_FOLDER, exist_ok=True)

#List and sort image files in the folder with the .bmp extension
file_list = sorted([f for f in os.listdir(image_folder) if f.endswith('.bmp')])
# file_list = file_list[0:2]
for filename in file_list:

    image_path = os.path.join(image_folder, filename)
    print(f' --------------  Processing {filename}... --------------')

    #Embedding watermark in the image
    watermarked_img = embedding_polymer.embedding(image_path, 'polymer.npy')
    output_image = f'{WATERMARKED_FOLDER}/{GROUP_NAME}_{filename}'
    
    #Save the watermarked image
    cv2.imwrite(output_image, watermarked_img)  
    
    #Apply JPEG compression
    attacked = jpeg_compression(watermarked_img, 60)
    cv2.imwrite('attacked.bmp', attacked) # Save the attacked image for detection

    #Start time for detection
    start = time.time()  
    dec, wpsnr = detection_polymer.detection(image_path, output_image, 'attacked.bmp') 
    print(f'[TIME CONSUMED]: {(time.time() - start):.2f} s') 
    print(f"[DETECTION failed = 1 , success = 0]: {dec}")
