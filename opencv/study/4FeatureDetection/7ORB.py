import numpy as np
import cv2 as cv

'''ORB is a fusion of FAST keypoint detector and BRIEF descriptor.
'''
base_path = 'opencv/data/'
img = cv.imread(base_path + 'blox.jpg')
orb = cv.ORB_create()
kp = orb.detect(img, None)
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
cv.imshow('img', img2)
cv.waitKey(0)
cv.destroyAllWindows()

# to be continued: match