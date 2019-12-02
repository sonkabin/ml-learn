import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

base_path = './opencv/data/'

img = cv.imread(base_path + '1.jpg')
res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
# or
# height, width = img.shape[:2]
# res = cv.resize(img, (2*width, 2*height), interpolation=cv.INTER_CUBIC)

cv.imshow('img', img)
cv.imshow('res', res)
cv.waitKey(0)
cv.destroyAllWindows()

# 偏移
img = cv.imread(base_path + '1.jpg', 0)
rows, cols = img.shape
print(rows, cols)
M = np.float32([[1, 0, 100], [0, 1, 50]])
dst = cv.warpAffine(img, M, (rows, cols))
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()

# 旋转
M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 90, 0.5)
dst = cv.warpAffine(img,M,(cols,rows))
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()

# 透视变换/投影变换
img = cv.imread(base_path + '1.jpg')
res = img[:,:,::-1]
pts1 = np.float32([[100,0],[400,0],[0,400],[400,400]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(res,M,(300,300))
plt.subplot(121),plt.imshow(res),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()