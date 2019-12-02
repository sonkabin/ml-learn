import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

base_path = './opencv/data/'
img = cv.imread(base_path + '1.jpg', 0)
edges = cv.Canny(img, 100, 200)

plt.subplot(121), plt.imshow(img, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('Original image')
plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title('Edge image')

plt.show()

# tracker
def nothing(x):
    pass

cv.namedWindow('Edge image')
cv.createTrackbar('minVal', 'Edge image',50, 150, nothing)
cv.createTrackbar('maxVal', 'Edge image',200, 300, nothing)

edges = img
while True:
    cv.imshow('Edge image', edges)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

    minVal = cv.getTrackbarPos('minVal', 'Edge image')
    maxVal = cv.getTrackbarPos('maxVal', 'Edge image')
    edges = cv.Canny(img, minVal, maxVal)

cv.destroyAllWindows()