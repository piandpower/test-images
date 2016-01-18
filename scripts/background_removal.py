import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

data_path = '/Users/ryoungblood/opencv_tests/data/'

show_intermediate = False

# 1) Load the source image
boot = cv2.imread(os.path.join(data_path,'boot.jpg'))
plt.imshow(cv2.cvtColor(boot,cv2.COLOR_BGR2RGB))
plt.show()

# 2) Convert to Grayscale
boot_gray = cv2.cvtColor(boot, cv2.COLOR_BGR2GRAY)
if show_intermediate: plt.imshow(boot_gray, cmap='gray')

# 2.5) Invert the grayscale image
boot_negate = 255 - boot_gray
if show_intermediate: plt.imshow(boot_negate, cmap='gray')

# 3) Get the Sobel-filtered negative
# Get the horizontal Sobel
#!!!!!!! PARAMETER !!!!!!!!! (ksize)
boot_sobel_h_64f = cv2.Sobel(boot_negate,cv2.CV_64F,1,0,ksize=3)
boot_sobel_h_64f_abs = np.absolute(boot_sobel_h_64f)
boot_sobel_h_8u = np.uint8(boot_sobel_h_64f_abs)

# Get the vertical Sobel
boot_sobel_v_64f = cv2.Sobel(boot_negate,cv2.CV_64F,0,1,ksize=3)
boot_sobel_v_64f_abs = np.absolute(boot_sobel_v_64f)
boot_sobel_v_8u = np.uint8(boot_sobel_v_64f_abs)

# Bitwise OR those two
boot_sobel = cv2.bitwise_or(boot_sobel_h_8u,boot_sobel_v_8u)
#print(boot_sobel[:10,:10])

if show_intermediate: plt.imshow(boot_sobel, cmap='gray')

# 4) Gaussian blur the Sobel
#!!!!!!! PARAMETER !!!!!!!!! (gauss kernel size)
boot_gauss = cv2.GaussianBlur(boot_sobel, (3,3), 0)
if show_intermediate: plt.imshow(boot_gauss, cmap='gray')

# 5) Threshold-filter to drop away (to pure black) anything that's not white-ish
threshold = 20 #!!!!!!! PARAMETER !!!!!!!!! (threshold)
ret,boot_tozero = cv2.threshold(boot_gauss,threshold,255,cv2.THRESH_TOZERO)
if show_intermediate: plt.imshow(boot_tozero, cmap='gray')

# 5.5) Convert from single-channel (grayscale) back to BGR
boot_tozero_bgr = cv2.cvtColor(boot_tozero, cv2.COLOR_GRAY2BGR)
if show_intermediate: plt.imshow(boot_tozero_bgr)

# 6) Local adaptive threshold - looks for lone pixels in 5x5 areas and removes those which are not in a cluster
# TODO: implement

# 7) Flood fill - fills
# TODO: update to flood fill from all of the corners
mask = np.zeros((boot_tozero.shape[0]+2,boot_tozero.shape[1]+2),np.uint8)
diff = (6,6,6)
cv2.floodFill(image=boot_tozero_bgr,mask=mask,seedPoint=(0,0),
             newVal=(255,0,255),loDiff=diff,upDiff=diff, flags=4)
if show_intermediate: plt.imshow(boot_tozero_bgr)

# 8) Resulting Mask
if show_intermediate: plt.imshow(mask, cmap='gray')
mask = mask * 255
mask_bgr = cv2.cvtColor(mask[1:-1,1:-1],cv2.COLOR_GRAY2BGR)

# 9) Background Removed
boot_bgremoved = cv2.bitwise_or(mask_bgr,boot)
plt.imshow(cv2.cvtColor(boot_bgremoved,cv2.COLOR_BGR2RGB))
plt.show()

titles = ['1) Original Image', '2) Negate', '3) Sobel Filter',
          '4) Gaussian Blur', '5) Threshold', '6) Local Adaptive Threshold',
          '7) Flood Fill', '8) OR Mask', '9) Background Removed']
images = [boot, boot_negate, boot_sobel, boot_gauss, boot_tozero, boot_tozero,
          boot_tozero_bgr, mask_bgr, boot_bgremoved]


for i in [0,6,7,8]:
    plt.subplot(3,3,i+1),plt.imshow(cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

for i in [1,2,3,4,5]:
    plt.subplot(3,3,i+1),plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()