import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Generate your watermark
N = 1024  # watermark size
watermark = (np.random.normal(0, 1, N) > 0).astype(int)
print('Watermark: ', watermark)

# Save the watermark to a .npy file
np.save('watermark.npy', watermark)
