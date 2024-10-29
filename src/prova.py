from numpy import array, diag, dot
import numpy as np
from scipy.linalg import svd
import pywt

# Define the image and watermark matrices
img = array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
             ])

# img = img.astype(float)

wm   = array([[0, 1, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 0, 1, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 0, 0]
            ])

positions = [ [0,0], [3,3], [5,2], [6,5], [10,8], [13,11], [8,2], [13,0]]

alpha = 5.70

# Perform Singular Value Decomposition (SVD) on both matrices
U_img, s_img, VT_img = np.linalg.svd(img)
s_orig = s_img.copy()  # Store the original s_img values for later comparison
img_copy = img.copy()
print(img)
print('WATERMARK 1:\n', wm)
U_wm, s_wm, VT_wm = np.linalg.svd(wm)

# Adjust s_img by adding s_wm values to every second position
# for i in range(len(s_wm)):
#     s_img[i] += s_wm[i]

# Block per block embedding
i = 0
for (x,y) in positions:
    block = img[x:x+2,y:y+2]

    # apply wavelet to the block
    Coefficients = pywt.wavedec2(block, wavelet='haar', level=1)
    LL_block = Coefficients[0]

    # perform svd on the LL band
    U_bl, s_bl, VT_bl = np.linalg.svd(LL_block)
    # embed the watermark in LL band
    s_bl[0] += s_wm[i]*alpha
    # reconstruct the LL band
    r_block = np.dot(U_bl, np.dot(np.diag(s_bl), VT_bl))

    Coefficients[0] = r_block
    block_new = pywt.waverec2(Coefficients, wavelet='haar')

    # replace the embedded block in the image
    img[x:x+2,y:y+2] = block_new
    i+=1

S_wm_reconstructed = []
for (x,y) in positions:
    block = img[x:x+2,y:y+2]
    block_ori = img_copy[x:x+2,y:y+2]

    # apply wavelet to the block
    Coefficients = pywt.wavedec2(block, wavelet='haar', level=1)
    LL_block = Coefficients[0]
    Coefficients = pywt.wavedec2(block_ori, wavelet='haar', level=1)
    LL_block_ori = Coefficients[0]

    # perform svd on the LL band
    U_bl, s_bl, VT_bl = np.linalg.svd(LL_block)
    # perform svd on the original LL band
    U_bl_ori, s_bl_ori, VT_bl_ori = np.linalg.svd(LL_block_ori)
    # extract the watermark in the block
    S_wm_reconstructed.append((s_bl[0] - s_bl_ori[0])/alpha)

# Convert S_wm_copy to a diagonal matrix and reconstruct the watermark
Watermark = np.dot(U_wm, np.dot(np.diag(S_wm_reconstructed), VT_wm))

Watermark = np.round(Watermark, decimals=0).astype(int)  # Round and cast to int
print("Reconstructed Watermark from image embedded with watermark 1:\n", Watermark)

# Define the image and watermark matrices
img = array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
             ])

wm_2 = array([[1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 1, 0, 1, 1],
            [0, 1, 0, 1, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 1, 1, 1, 0, 0]
            ])

print('WATERMARK 2:\n', wm_2)

# Perform Singular Value Decomposition (SVD) on both matrices
U_img, s_img, VT_img = np.linalg.svd(img)
s_orig = s_img.copy()  # Store the original s_img values for later comparison
img_copy = img.copy()
U_wm2, s_wm2, VT_wm2 = np.linalg.svd(wm_2)

i = 0
for (x,y) in positions:
    block = img[x:x+2,y:y+2]

    # apply wavelet to the block
    Coefficients = pywt.wavedec2(block, wavelet='haar', level=1)
    LL_block = Coefficients[0]

    # perform svd on the LL band
    U_bl, s_bl, VT_bl = np.linalg.svd(LL_block)
    # embed the watermark in LL band
    s_bl[0] += s_wm2[i]*alpha
    # reconstruct the LL band
    r_block = np.dot(U_bl, np.dot(np.diag(s_bl), VT_bl))

    Coefficients[0] = r_block
    block_new = pywt.waverec2(Coefficients, wavelet='haar')

    # replace the embedded block in the image
    img[x:x+2,y:y+2] = block_new
    i+=1

S_wm_reconstructed = []
for (x,y) in positions:
    block = img[x:x+2,y:y+2]
    block_ori = img_copy[x:x+2,y:y+2]

    # apply wavelet to the block
    Coefficients = pywt.wavedec2(block, wavelet='haar', level=1)
    LL_block = Coefficients[0]
    Coefficients = pywt.wavedec2(block_ori, wavelet='haar', level=1)
    LL_block_ori = Coefficients[0]

    # perform svd on the LL band
    U_bl, s_bl, VT_bl = np.linalg.svd(LL_block)
    # perform svd on the original LL band
    U_bl_ori, s_bl_ori, VT_bl_ori = np.linalg.svd(LL_block_ori)
    # extract the watermark in the block
    S_wm_reconstructed.append((s_bl[0] - s_bl_ori[0])/alpha)

# Convert S_wm_copy to a diagonal matrix and reconstruct the watermark
Watermark = np.dot(U_wm, np.dot(np.diag(S_wm_reconstructed), VT_wm))

Watermark = np.round(Watermark, decimals=0).astype(int)  # Round and cast to int
print("Reconstructed Watermark from image embedded with watermark 2:\n", Watermark)