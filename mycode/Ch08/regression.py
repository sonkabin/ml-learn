import numpy as np

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