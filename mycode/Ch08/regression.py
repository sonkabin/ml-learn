import numpy as np
from numpy import linalg as la

def loadDataSet(filename):
    numFeat = 0
    dataArr = []; labelArr = []
    with open(filename) as f:
        numFeat = len(f.readline().split('\t')) - 1
        for line in f.readlines():
            lineArr = []
            currLine = line.strip().split('\t')
            for i in range(numFeat):
                lineArr.append(float(currLine[i]))
            dataArr.append(lineArr)
            labelArr.append(float(currLine[-1]))
    return dataArr, labelArr
# 正规方程
def standRegres(xArr, yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print('This matrix is singular')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws
# 梯度下降
def granDescent(dataArr, labelArr, iterNum=200):
    dataMat = np.mat(dataArr); labelMat = np.mat(labelArr).T
    m,n = np.shape(dataMat)
    weights = np.ones((n,1))
    alpha = 0.001 # 步长为0.01时，就会导致无法找到最小值
    for j in range(iterNum):
        h = dataMat * weights
        error = h - labelMat
        weights = weights - alpha*dataMat.T*error
    return weights
'''
from importlib import reload
import mycode.Ch08.regression as regression
xArr, yArr = regression.loadDataSet('./mycode/Ch08/ex0.txt')

'''

def plot(dataArr, labelArr, weights):
    import matplotlib.pyplot as plt
    xArr = np.array(dataArr).T[1] # 取每个样本的第二个特征
    yArr = np.array(labelArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xArr, yArr)
    # 将点按序排列，否则绘图会出现问题
    xCopy = dataArr.copy()
    xCopy.sort()
    # 由于x0=0，故h=w[0]+w[1]*x1，故x1作为x轴
    h = np.array(xCopy) * weights # 计算预测值
    ax.plot(np.array(xCopy).T[1], h)
    plt.show()


# 局部加权线性回归
# testPoint:给定向量
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr);    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))
    for i in range(m):
        diffMat = testPoint - xMat[i,:]
        weights[i,i] = np.exp(diffMat * diffMat.T / (-2*k**2))
    xTx = xMat.T * (weights * xMat)
    if la.det(xTx) == 0.0:
        print('the matrix is singular, cannot do inverse')
        return 
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def plot2(xArr, yArr, yHatArr):
    xMat = np.mat(xArr)
    sortIndex = xMat[:,1].argsort(0)
    xSort = xMat[sortIndex,1]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort, yHatArr[sortIndex])
    ax.scatter(xMat[:,1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()

# 预测鲍鱼的年龄
'''
abX, abY = regression.loadDataSet('./mycode/Ch08/abalone.txt')
yHat01 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
yHat1 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
yHat10 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)

regression.rssError(abY[0:99], yHat01.T)
regression.rssError(abY[0:99], yHat1.T)
regression.rssError(abY[0:99], yHat10.T)

yHat01 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
regression.rssError(abY[100:199], yHat01.T)
yHat1 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
regression.rssError(abY[100:199], yHat1.T)
yHat10 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
regression.rssError(abY[100:199], yHat10.T)

'''
def rssError(yArr, yHatArr):
    return np.sum((yArr - yHatArr)**2)
# end

# 岭回归
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + lam * np.eye(np.shape(xMat)[1])
    if la.det(denom) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr);    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMean = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMean) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i,:] = ws.T
    return wMat

'''
abX, abY = regression.loadDataSet('./mycode/Ch08/abalone.txt')
ridgeWeights = regression.ridgeTest(abX, abY)
regression.plot3(ridgeWeights)
'''
def plot3(weights):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(weights)
    plt.show()
# end

# 前向逐步线性回归
def stageWise(xArr, yArr, eps=0.01, numIter=100):
    xMat = np.mat(xArr);    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMean = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMean) / xVar
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIter, n))
    ws = np.zeros((n,1));   wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIter):
        print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if lowestError > rssE:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat
