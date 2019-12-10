import numpy as np
import cv2 as cv

# SURF is good at handling images with blurring and rotation, 
# but not good at handling viewpoint change and illumination change.
base_path = 'opencv/data/'
img =  cv.imread(base_path + 'butterfly.jpg', 0)
surf = cv.xfeatures2d.SURF_create(400) # Because the same problem, I put it away.