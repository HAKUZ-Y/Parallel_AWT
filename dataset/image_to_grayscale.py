from PIL import Image
import numpy as np
import cv2

input_file = "./origin/NISEL"
size = 16384

img = Image.open(input_file + ".jpg").convert("L")
img = img.resize((size, size))
arr = np.array(img)

np.savetxt(input_file + "_" + str(size) + "_" + ".txt", arr, fmt="%.1f")
cv2.imwrite("restored.jpg", arr.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])

