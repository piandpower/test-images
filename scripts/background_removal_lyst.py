import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

data_path = '/Users/ryoungblood/opencv_tests/data/'

show_intermediate = False

def remove_bg(file,output_file=None,show_intermediate=False):

    # 1) Load the source image
    img = cv2.imread(file)
    # if show_intermediate:
    #     plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    #     plt.show()

    # 2) Convert to Grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if show_intermediate:
    #     plt.imshow(img_gray, cmap='gray')
    #     plt.show()

    # 2.5) Invert the grayscale image
    img_negate = 255 - img_gray
    # if show_intermediate:
    #     plt.imshow(img_negate, cmap='gray')
    #     plt.show()

    # 3) Get the Sobel-filtered negative
    # Get the horizontal Sobel
    #!!!!!!! PARAMETER !!!!!!!!! (ksize)
    img_sobel_h_64f = cv2.Sobel(img_negate,cv2.CV_64F,1,0,ksize=3)
    img_sobel_h_64f_abs = np.absolute(img_sobel_h_64f)
    img_sobel_h_8u = np.uint8(img_sobel_h_64f_abs)

    # Get the vertical Sobel
    img_sobel_v_64f = cv2.Sobel(img_negate,cv2.CV_64F,0,1,ksize=3)
    img_sobel_v_64f_abs = np.absolute(img_sobel_v_64f)
    img_sobel_v_8u = np.uint8(img_sobel_v_64f_abs)

    # Bitwise OR those two
    img_sobel = cv2.bitwise_or(img_sobel_h_8u,img_sobel_v_8u)
    #print(img_sobel[:10,:10])

    # if show_intermediate:
    #     plt.imshow(img_sobel, cmap='gray')
    #     plt.show()

    # 4) Gaussian blur the Sobel
    #!!!!!!! PARAMETER !!!!!!!!! (gauss kernel size)
    img_gauss = cv2.GaussianBlur(img_sobel, (3,3), 0)
    # if show_intermediate:
    #     plt.imshow(img_gauss, cmap='gray')
    #     plt.show()

    # 5) Threshold-filter to drop away (to pure black) anything that's not white-ish
    threshold = 20 #!!!!!!! PARAMETER !!!!!!!!! (threshold)
    ret,img_tozero = cv2.threshold(img_gauss,threshold,255,cv2.THRESH_TOZERO)
    # if show_intermediate:
    #     plt.imshow(img_tozero, cmap='gray')
    #     plt.show()

    # 5.5) Convert from single-channel (grayscale) back to BGR
    img_tozero_bgr = cv2.cvtColor(img_tozero, cv2.COLOR_GRAY2BGR)
    # if show_intermediate:
    #     plt.imshow(img_tozero_bgr)
    #     plt.show()

    # 6) Local adaptive threshold - looks for lone pixels in 5x5 areas and removes those which are not in a cluster
    # TODO: implement

    # 7) Flood fill - fills
    # TODO: update to flood fill from all of the corners
    mask = np.zeros((img_tozero.shape[0]+2,img_tozero.shape[1]+2),np.uint8)
    diff = (6,6,6)
    cv2.floodFill(image=img_tozero_bgr,mask=mask,seedPoint=(0,0),
                 newVal=(255,0,255),loDiff=diff,upDiff=diff, flags=4)
    # if show_intermediate:
    #     plt.imshow(img_tozero_bgr)
    #     plt.show()

    # 8) Resulting Mask
    # if show_intermediate:
    #     plt.imshow(mask, cmap='gray')
    #     plt.show()
    mask = mask * 255
    mask_bgr = cv2.cvtColor(mask[1:-1,1:-1],cv2.COLOR_GRAY2BGR)

    # 9) Background Removed
    img_bgremoved = cv2.bitwise_or(mask_bgr,img)
    # if show_intermediate:
    #     plt.imshow(cv2.cvtColor(img_bgremoved,cv2.COLOR_BGR2RGB))
    #     plt.show()

    if show_intermediate:
        titles = ['1) Original Image', '2) Negate', '3) Sobel Filter',
                  '4) Gaussian Blur', '5) Threshold', '6) Local Adaptive Threshold',
                  '7) Flood Fill', '8) OR Mask', '9) Background Removed']
        images = [img, img_negate, img_sobel, img_gauss, img_tozero, img_tozero,
                  img_tozero_bgr, mask_bgr, img_bgremoved]


        for i in [0,6,7,8]:
            plt.subplot(3,3,i+1),plt.imshow(cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])

        for i in [1,2,3,4,5]:
            plt.subplot(3,3,i+1),plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])

        plt.show()

    if output_file:
        print(output_file)
        cv2.imwrite(output_file,img_bgremoved)