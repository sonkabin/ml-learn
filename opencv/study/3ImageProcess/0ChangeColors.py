# -*- coding: utf-8 -*-  
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

base_path = './opencv/data/'

img = cv.imread(base_path + '1.jpg')
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
mask = cv.inRange(hsv, lower_blue, upper_blue)
res = cv.bitwise_and(img, img, mask=mask)

cv.imshow('img', img)
cv.imshow('res', res)
cv.waitKey(0)
cv.destroyAllWindows()

# How to find HSV values to track?
# Just pass the BGR values you want. For example, find the HSV value of Green.
green = np.uint8([[[0, 255, 0]]])
hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
print(hsv_green)