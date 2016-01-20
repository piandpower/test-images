import os
import numpy as np
import matplotlib.pyplot as plt
import mimetypes
import random
import math

import cv2

from scripts import background_removal_lyst

random.seed = 42

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

    # 4) Rotate random degrees
    transformation = 'randrot'

    # rotate function from http://john.freml.in/opencv-rotation
    def rotate_about_center(src, angle, scale=1.):
        w = src.shape[1]
        h = src.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

    angle = random.randrange(0,360,1)
    print(angle)
    img_rot = rotate_about_center(img_bgr,angle)
    cv2.imwrite(file+'_'+transformation+ext,img_rot)

    # 5) Random Image Subset
    transformation = 'randsub'

    rows, cols = img_bgr.shape[:2]
    # will take x_size_pct percent of the x-axis, starting at column x_loc
    x_pct_range = (.1,.9)
    x_size_pct = random.uniform(x_pct_range[0],x_pct_range[1])
    x_size = int(round(x_size_pct * cols))

    y_pct_range = (.1,.9)
    y_size_pct = random.uniform(y_pct_range[0],y_pct_range[1])
    y_size = int(round(y_size_pct * rows))

    #if x_size is 0.7, then x_loc (the left-bound of the window) can be 0-0.3
    x_loc_pct = random.uniform(0,1-x_size_pct)
    x_loc = int(round(x_loc_pct * cols))
    y_loc_pct = random.uniform(0,1-y_size_pct)
    y_loc = int(round(y_loc_pct * rows))
    #print(x_loc,x_size)
    #print(y_loc,y_size)
    img_rsub = img_bgr[y_loc:(y_loc+y_size),x_loc:(x_loc+x_size)]

    cv2.imwrite(file+'_'+transformation+ext,img_rsub)

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
