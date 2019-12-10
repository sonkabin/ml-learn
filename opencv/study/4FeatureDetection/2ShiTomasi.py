import cv2 as cv
import numpy as np

base_path = 'opencv/data/'
img = cv.imread(base_path + 'blox.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 25个角点，低于0.01的角点将被拒绝，两个角点之前最小的距离为10
corners = cv.goodFeaturesToTrack(gray, 25, 0.1, 10)
corners = np.int0(corners) # intp的别名

for i in corners:
    x, y = i.ravel()
    cv.circle(img, (x, y), 3, 255, -1)

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()