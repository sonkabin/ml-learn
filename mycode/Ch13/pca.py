import numpy as np

def loadDataSet(filename, delim='\t'):
    with open(filename) as f:
        stringArr = [line.strip().split(delim) for line in f.readlines()]
    dataArr = [list(map(float, line)) for line in stringArr]
    return dataArr

'''
import mycode.Ch13.pca as pca
import numpy as np
dataArr = pca.loadDataSet('./mycode/Ch13/testSet.txt')
lowDMat, reconMat = pca.pca(dataArr)
pca.plot(np.mat(dataArr), reconMat)

from importlib import reload
'''
def pca(dataSet, topNfeat=9999999):
    dataMat = np.mat(dataSet)
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=0) # 协方差矩阵
    eigVals, eigVects = np.linalg.eig(np.mat(covMat)) # 求解特征值与特征向量，其中特征向量是列向量
    eigValIndex = np.argsort(eigVals)
    eigValIndex = eigValIndex[:-(topNfeat+1):-1] # 取topNfeat个较大特征对应的索引
    redEigVects = eigVects[:,eigValIndex]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    print(eigVects)
    print(redEigVects)
    print(eigVects * eigVects.T)
    print(redEigVects * redEigVects.T)
    return lowDDataMat, reconMat

def plot(dataMat, reconMat):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()