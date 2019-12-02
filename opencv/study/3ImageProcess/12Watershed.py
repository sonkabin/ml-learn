import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

base_path = 'opencv/data/'
img = cv.imread(base_path + 'coins.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2) # 移除白点噪声
# sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
# my test
erosion = cv.erode(opening, kernel, iterations=1)

# Finding unknown region?
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers += 1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]
# because markers have -1 values, we can't show a image
# print(markers.shape, img.shape, thresh.shape)

# cv.imshow('thresh', thresh)
# cv.imshow('sure_bg', sure_bg)
# cv.imshow('sure_fg', sure_fg)
# cv.imshow('erosion', erosion)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()