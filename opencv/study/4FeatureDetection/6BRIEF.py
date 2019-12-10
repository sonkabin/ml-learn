import numpy as np
import cv2 as cv

'''Binary Robust Independent Elementary Features
BRIEF is a feature descriptor, it doesn't provide any method to find the features. 
一般过程：首先是找到detector，然后再用descriptor，最后match
因为在实际的match中，SIFT和SURF虽然维数多但可能用不到
'''

base_path = 'opencv/data/'
img = cv.imread(base_path + 'blox.jpg',0)

# Initiate STAR detector(CenSurE detector)
star = cv.xfeatures2d.StarDetector_create()

# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

# find the keypoints with STAR
kp = star.detect(img, None)

# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)

print( brief.descriptorSize() )
print( des.shape )
img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
cv.imshow('img2', img2)
cv.waitKey(0)
cv.destroyAllWindows()

# to be continued: match