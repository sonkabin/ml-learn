import numpy as np
import cv2 as cv

# FAST (Features from Accelerated Segment Test) algorithm
base_path = 'opencv/data/'
img = cv.imread(base_path + 'blox.jpg')

# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img, None)
img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

cv.imshow('img2', img2)
cv.waitKey(0)
cv.destroyAllWindows()

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
img3 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))

# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

cv.imshow('img3', img3)
cv.waitKey(0)
cv.destroyAllWindows()

# Okay, let's go.
fast.setNonmaxSuppression(1)
fast.setThreshold(40)
# For the neighborhood, three flags are defined:
# cv.FAST_FEATURE_DETECTOR_TYPE_5_8, cv.FAST_FEATURE_DETECTOR_TYPE_7_12 and cv.FAST_FEATURE_DETECTOR_TYPE_9_16
fast.setType(cv.FAST_FEATURE_DETECTOR_TYPE_9_16)
kp = fast.detect(img, None)
img4 = cv.drawKeypoints(img, kp, None, color=(0, 0, 255))

cv.imshow('img4', img4)
cv.waitKey(0)
cv.destroyAllWindows()