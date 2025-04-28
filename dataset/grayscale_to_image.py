from PIL import Image
import numpy as np
import cv2

# input_txt = "./grayscale/8x8.txt_transform_1.txt"  # Your .txt file
# input_txt = "./reconstructed/easy_2048_level_1_t_0.00.txt"
input_txt = "./transformed/easy_2048_level_1_t_0.00.txt"
output_img = "./compress/easy_level_1_t_0.00.jpg"     # Desired image output file

# Load grayscale matrix from .txt file
arr = np.loadtxt(input_txt)

# Convert to 8-bit unsigned integer
img_uint8 = arr.astype(np.uint8)

# Save using OpenCV
cv2.imwrite(output_img, img_uint8, [cv2.IMWRITE_JPEG_QUALITY, 95])

# Optionally display with PIL
img = Image.fromarray(img_uint8)
