test_data_path = '/Users/ryoungblood/image_similarity/'
src_image_path = 'test_images'
dst_image_path = 'transformed_images'

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def img_transform(src_path, dst_path):
    img_bgr = cv2.imread(src_path)
    file, ext = os.path.splitext(dst_path)

    print(img_bgr.shape)

    # 0) Base
    transformation = 'base'
    cv2.imwrite(file+'_'+transformation+ext,img_bgr)

    # 1) Grayscale
    transformation = 'gray'
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(file+'_'+transformation+ext,img_gray)

    # 2) Resize to 50%
    transformation = 'resize50pct'
    img_rs50pct = cv2.resize(img_bgr, dsize=(0,0), fx=0.5, fy=0.5)
    cv2.imwrite(file+'_'+transformation+ext,img_rs50pct)

    # 3) Resize to arbitrary 500x500
    transformation = 'resize50pct'
    img_rs500x500 = cv2.resize(img_bgr, dsize=(500,500))
    cv2.imwrite(file+'_'+transformation+ext,img_rs50pct)

    # 4) Rotate 90 degrees CCW
    transformation = 'rot90CCW'
    rows, cols = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,cols/2),90,1)
    img_rot90CCW = cv2.warpAffine(img_bgr,M,(rows,cols))
    cv2.imwrite(file+'_'+transformation+ext,img_rot90CCW)

    # 5)

for img in os.listdir(os.path.join(test_data_path,src_image_path)):
    file, ext = os.path.splitext(os.path.join(test_data_path,dst_image_path,img))

    img_transform(os.path.join(test_data_path,src_image_path,img),
                  os.path.join(test_data_path,dst_image_path,img))

