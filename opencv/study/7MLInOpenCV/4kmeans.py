import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

'''Parameters
Input parameters:
    1.samples : It should be of np.float32 data type, and each feature should be put in a single column.
    2.nclusters(K) 
    3.criteria : a tuple (type, max_iter, epsilon)
                type: cv.TERM_CRITERIA_EPS or cv.TERM_CRITERIA_MAX_ITER or cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER
                max_iter: An integer specifying maximum number of iterations
                epsilon: Required accuracy
    4.attempts : Flag to specify the number of times the algorithm is executed using different initial labellings. 
    5.flags : 初始化中心选用的方法: cv.KMEANS_PP_CENTERS and cv.KMEANS_RANDOM_CENTERS
Output parameters:
    1.compactness: 每个点到中心的平方距离之和
    2.labels : `0`, `1` and so on
    3.centers: This is array of centers of clusters.
'''

def one_feature():
    x = np.random.randint(25, 100, 25)
    y = np.random.randint(175, 255, 25)
    z = np.hstack((x, y))
    print(x.shape, z.shape)
    z = z.reshape((50, 1))
    z = np.float32(z)
    plt.subplot(121)
    plt.hist(z, 256, [0, 256])

    # Define criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    flags = cv.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv.kmeans(z, 2, None, criteria, 10, flags)
    A = z[labels==0]
    B = z[labels==1]
    # Now plot 'A' in red, 'B' in blue, 'centers' in yellow
    plt.subplot(122)
    plt.hist(A,256,[0,256],color = 'r')
    plt.hist(B,256,[0,256],color = 'b')
    plt.hist(centers,32,[0,256],color = 'y')
    plt.show()

def multiple_features():
    X = np.random.randint(25, 50, (25, 2))
    Y = np.random.randint(60, 85, (25, 2))
    Z = np.vstack((X, Y))
    print(Z.shape)
    Z = np.float32(Z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, labels, centers = cv.kmeans(Z, 2, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    A = Z[labels.ravel() == 0]
    B = Z[labels.ravel() == 1]

    # Plot the data
    plt.scatter(A[:,0],A[:,1])
    plt.scatter(B[:,0],B[:,1],c = 'r')
    plt.scatter(centers[:,0],centers[:,1],s = 80,c = 'y', marker = 's')
    plt.xlabel('Height'),plt.ylabel('Weight')
    plt.show()

def color_quantization():
    base_path = 'opencv/data/'
    img = cv.imread(base_path + 'home.jpg')
    Z = img.reshape((-1, 3))
    # print(Z.shape)
    Z = np.float32(Z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    compactness, labels, centers = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    # print(centers.shape, labels.shape, res.shape)
    res2 = res.reshape((img.shape))
    cv.imshow('img', img)
    cv.imshow('res2', res2)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    # one_feature()
    # multiple_features()
    color_quantization()
    