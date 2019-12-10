import cv2 as cv
import numpy as np

base_path = 'opencv/data/'
img = cv.imread(base_path + 'chessboard.jpg')
img2 = img.copy()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

# 1)k=0.04，dst即为R。为角点时，R很大；为边缘时，R<0；为普通区域时，|R|很小
dst = cv.cornerHarris(gray, 2, 3, 0.04)

# print(dst[3,:]>0.01*dst.max())
dst = cv.dilate(dst, None)
img[dst>0.01*dst.max()] = [0, 0, 255]

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

# 根据1可以做测试
img2[dst<-100] = [0, 0, 255]
cv.imshow('img2', img2)
cv.waitKey(0)
cv.destroyAllWindows()