import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

base_path = './opencv/data/'
img = cv.imread(base_path + '1.jpg')
img = img[:,:,::-1]

plt.subplot(321), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
for i in range(5):
    kernel = np.ones((5, 5), np.float32)/25
    dst = cv.filter2D(img, -1-i, kernel)
    plt.subplot(3, 2, i+2), plt.imshow(dst), plt.title('Averaging' + str(i))
    plt.xticks([]), plt.yticks([])
plt.show()

blur = cv.blur(img, (5, 5))
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()