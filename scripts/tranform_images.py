import os
import numpy as np
import matplotlib.pyplot as plt
import mimetypes

import cv2

from scripts import background_removal_lyst

dirname = os.path.dirname
base_path = dirname(dirname(__file__))

src_image_path = os.path.join(base_path,'original_images')
transformed_image_path = os.path.join(base_path,'transformed_images')
bg_removed_image_path = os.path.join(base_path,'bg-removed_images')

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

    # 2) Resize to X%
    transformation = 'resizeXpct'
    scale = .25
    img_resizeXpct = cv2.resize(img_bgr, dsize=(0,0), fx=scale, fy=scale)
    cv2.imwrite(file+'_'+transformation+ext,img_resizeXpct)

    # 2.5) Resize back to 100%
    transformation = 'resize100pct'
    scale = 1/scale
    img_resize100pct = cv2.resize(img_resizeXpct, dsize=(0,0), fx=scale, fy=scale)
    cv2.imwrite(file+'_'+transformation+ext,img_resize100pct)

    # 3) Resize to arbitrary 500x500
    transformation = 'resize500x500'
    img_rs500x500 = cv2.resize(img_bgr, dsize=(500,500))
    cv2.imwrite(file+'_'+transformation+ext,img_rs500x500)

    # 4) Rotate 90 degrees CCW
    transformation = 'rot90CCW'
    rows, cols = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((round(cols/2),round(cols/2)),90,1)
    img_rot90CCW = cv2.warpAffine(img_bgr,M,(rows,cols))
    cv2.imwrite(file+'_'+transformation+ext,img_rot90CCW)

    # 5) Image subset (middle X% of each axis)
    transformation = 'centerXpct'
    x = .8
    img_sub = img_bgr[
        int(round((1-x)/2*img_bgr.shape[0],0)):int(round((x+(1-x)/2)*img_bgr.shape[0],0)),
        int(round((1-x)/2*img_bgr.shape[1],0)):int(round((x+(1-x)/2)*img_bgr.shape[1],0))
    ]
    cv2.imwrite(file+'_'+transformation+ext,img_sub)

    # 6) Gaussian Blur
    transformation = 'gauss'
    img_gauss = cv2.GaussianBlur(img_bgr, (13,13), 0)
    cv2.imwrite(file+'_'+transformation+ext,img_gauss)

for img in os.listdir(src_image_path):
    file, ext = os.path.splitext(img)
    print(img)
    if mimetypes.guess_type(img)[0] is not None and mimetypes.guess_type(img)[0].startswith('image'):
        img_transform(os.path.join(src_image_path,img),
                      os.path.join(transformed_image_path,img))
        background_removal_lyst.remove_bg(file=os.path.join(src_image_path,img),
                                     output_file=os.path.join(bg_removed_image_path,file+'_bg-removed'+ext),
                                     show_intermediate=False)

