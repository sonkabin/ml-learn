import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

base_path = 'opencv/data/'

# Find Histogram: 1. opencv function ;2. numpy function
img = cv.imread(base_path + '2.png', 0)
hist = cv.calcHist([img], [0], None, [256], [0, 256]) # opencv function for histogram, 40x faster than numpy funtion
# print(hist)
hist, bins = np.histogram(img.ravel(), 256, [0, 256]) # numpy function for histogram
# print(hist)

# Plotting Histograms: 1. matplotlib function; 2.opencv function
plt.hist(img.ravel(), 256, [0, 256])
plt.show()
img = cv.imread(base_path + '2.png')
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()

# Application of Mask
img = cv.imread(base_path + '2.png', 0)
# create mask, white color on the region you want to find histogram and black otherwise
mask = np.zeros(img.shape, np.uint8)
mask[100:600, 100:400] = 255
masked_img = cv.bitwise_and(img, img, mask=mask)
hist_full = cv.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv.calcHist([img], [0], mask, [256], [0, 256])
plt.subplot(221),   plt.imshow(img, 'gray')
plt.subplot(222),   plt.imshow(mask, 'gray')
plt.subplot(223),   plt.imshow(masked_img, 'gray')
plt.subplot(224),   plt.plot(hist_full),    plt.plot(hist_mask)
plt.xlim([0, 256])
plt.show()

# Histograms Equalization in OpenCV
equ = cv.equalizeHist(img)
cv.imshow('img', img)
cv.imshow('equ', equ)
cv.waitKey(0)
cv.destroyAllWindows()

# 2D Histograms
img = cv.imread(base_path + '2.png')
# in opencv
hsv = cv.cvtColor(img, cv.BGR2HSV)
hist_cv = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# ??? 有什么用呢
# Histogram Backprojection不是很懂