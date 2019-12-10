import numpy as np
import cv2 as cv

# Scale Invariant Feature Transform 
# 对于Harris算法来说，当某个角点放大之后，它可能就不是角点了
base_path = 'opencv/data/'
img = cv.imread(base_path + 'home.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create() # 版权所有，没有安装，告辞
kp = sift.detect(gray, None)
img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()