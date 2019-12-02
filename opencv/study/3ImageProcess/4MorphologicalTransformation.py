import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

base_path = './opencv/data/'
img = cv.imread(base_path + 'j.png', 0)
kernel = np.ones((5, 5), np.uint8) # np.unit8,8位无符号整型
'''腐蚀
作用：去除白点噪声
原理：一个像素点，以它为中心的kernel大小的区块，如果该区块全部为1，则该像素保留为1；否则为0
结果：缩小一圈
''' 
erosion = cv.erode(img, kernel, iterations=1) 
'''扩张
作用：去除黑点噪声
原理：和腐蚀相反
结果：扩大一圈
'''
dilation = cv.dilate(img, kernel, iterations=1)
img2 = cv.dilate(erosion, kernel, iterations=1)

cv.imshow('img', img)
cv.imshow('erosion', erosion)
cv.imshow('dilation', dilation)
cv.imshow('img2', img2)
cv.waitKey(0)
cv.destroyAllWindows()

opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel) # 先erosion再dilation
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel) # 先dilation再erosion
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel) # 像把轮廓捞出来
cv.imshow('opening', opening)
cv.imshow('closing', closing)
cv.imshow('gradient', gradient)
cv.waitKey(0)
cv.destroyAllWindows()

kernel = np.ones((9, 9), np.uint8)
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

cv.imshow('tophat', tophat)
cv.imshow('blackhat', blackhat)
cv.waitKey(0)
cv.destroyAllWindows()

# 创建矩形/圆形/十字kernel
print(cv.getStructuringElement(cv.MORPH_RECT, (5, 5)))
print(cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2)))
print(cv.getStructuringElement(cv.MORPH_CROSS, (5, 5)))