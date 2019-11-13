import numpy as np
import cv2 as cv
import time

base_path = './opencv/data/'

x = np.uint8([250])
y = np.uint8([10])
print(x+y) # 模操作
print(cv.add(x, y)) # 饱和操作

width = 500; height = 300
img1 = cv.imread(base_path + '1.jpg')
img1 = cv.resize(img1, (width, height))
img2 = cv.imread(base_path + '2.png')
img2 = cv.resize(img2, (width, height))

dst = cv.addWeighted(img1, 0.5, img2, 0.5, 0)
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()

# Load two images
img1 = cv.imread(base_path + '2.png')
img2 = cv.imread(base_path + '1.jpg')
img2 = cv.resize(img2, (160, 220))
# I want to put image on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ] # img1的top-left部分
# Now create a mask of image and create its inverse mask also
img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)
# Now black-out the area of image in ROI
img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
# Take only region of image from image.
img2_fg = cv.bitwise_and(img2,img2,mask = mask)
# Put image in ROI and modify the main image
dst = cv.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst
cv.imshow('res',img1)
cv.waitKey(0)
cv.destroyAllWindows()

# 实现幻灯片效果
width = 500; height = 300
img1 = cv.imread(base_path + '1.jpg')
img1 = cv.resize(img1, (width, height))
img2 = cv.imread(base_path + '2.png')
img2 = cv.resize(img2, (width, height))

global alpha
alpha = 0
def slide(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONUP:
        global alpha
        if alpha < 10:
            alpha = alpha + 1
        alpha2 = alpha / 10
        dst = cv.addWeighted(img1, alpha2, img2, 1-alpha2, 0)
        cv.imshow('dst', dst)

cv.namedWindow('dst')
cv.setMouseCallback('dst', slide)
dst = cv.addWeighted(img1, alpha, img2, 1-alpha, 0)
cv.imshow('dst', dst)
    
cv.waitKey(0)
cv.destroyAllWindows()