import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 霍夫线变换
# 作用：检测形状（如果该形状能用数学表示）
# 原理：暂不明白
# First parameter, Input image should be a binary image, so apply threshold or use canny edge detection before applying hough transform.
base_path = 'opencv/data/'
img = cv.imread(base_path + '1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150, apertureSize=3)

# 第2、3个参数表示精度。此时rho的精度为1，就是说100*100的图像中，rho会占100行
# 第4个参数表示阈值。至少需要几票才将其确定为直线
lines = cv.HoughLines(edges, 1, np.pi/180, 100) 
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0+1000*(-b))
    y1 = int(y0+1000*(a))
    x2 = int(x0-1000*(-b))
    y2 = int(y0-1000*(a))

    cv.line(img, (x1, y1), (x2, y2), 255, 2)
cv.imshow('canny', edges)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

# 基于概率的霍夫线变换
# 提出的原因：霍夫线变换的计算量大
img = cv.imread(base_path + '1.jpg')
# 第4个参数仍旧是几票。第5个参数指线的最短长度，第6个参数指两线之间的距离小于值时被视为一条直线
lines = cv.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=80, maxLineGap=10)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()