import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)
responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)

red = trainData[responses.ravel() == 0]
plt.scatter(red[:, 0], red[:, 1], s=80, c='r', marker='^')

blue = trainData[responses.ravel() == 1]
plt.scatter(blue[:, 0], blue[:, 1], s=80, c='b', marker='s')

# plt.show()

# test data is markered with green
newcomer = np.random.randint(0, 100, (3, 2)).astype(np.float32)
print(newcomer)
plt.scatter(newcomer[:, 0], newcomer[:, 1], s=80, c='g', marker='o')

knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, results, neighbours, dist = knn.findNearest(newcomer, 3)
print('results:', results)
print('neighbours:', neighbours)
print('distance:', dist)
plt.show()
