from numpy import *
import os
# 将文件中的数据存到数组中
def loadDataSet():
    dataMat = []; labelMat = []
    with open('./Ch05/testSet.txt') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            # 两个特征X1,X2，即z=w0*x0+w1*x1+w2*x2，而x0=1
            # 由假设函数可知，两个分类的分界线是z=0
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

# sigmoid函数
def sigmoid(inX):
    return 1.0 / (1+exp(-inX))

# 梯度上升算法：求最大值。对应的梯度下降算法求最小值
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn) # ndarray转matrix
    labelMat = mat(classLabels).transpose() # 等价于mat(classLabels).T
    m, n = shape(dataMatrix) #矩阵的维数
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1)) # numpy.ndarray
    for i in range(maxCycles):
        h = sigmoid(dataMatrix*weights) # 矩阵可直接和数组相乘
        error = labelMat - h 
        weights = weights + alpha*dataMatrix.transpose()*error
        '''
        梯度下降算法
        error = h - labelMat
        weights = weights - alpah*dataMatrix.T*error
        '''
    return weights

# 随机梯度上升算法
def stocGranAscent0(dataArr, classLabels, iterNum=1):
    m,n = shape(dataArr) # dataArr需要为numpy.ndarray
    alpha = 0.01
    weights = ones(n)
    for j in range(iterNum):
        for i in range(m):
            h = sigmoid(sum(dataArr[i]*weights))
            error = classLabels[i] - h
            weights = weights + alpha*error*dataArr[i]
    return weights

# 改进的随机梯度上升算法：1.每次迭代更新alpha 2.随机选取样本更新回归系数：防止系数出现周期性波动（部分样本不能正确分类）
def stocGranAscent1(dataArr, classLabels, iterNum=150):
    m, n =shape(dataArr)
    weights = ones(n)
    for j in range(iterNum):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.001 # 这里留有疑问
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataArr[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha*error*dataArr[randIndex]
            del(dataIndex[randIndex])
    return weights

# 画图
def plotBestFit(weights) :
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat) # 转成ndarray
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111) # 画在1行1列，第一块
    # scatter(x,y,s=size,c=color,marker=形状)，画散点图
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x) / weights[2] # z=0是两个分类的分界线，由此解出x1与x2的关系
    ax.plot(x,y) # 根据x和y画线
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()
    
def classifyVector(inX, weights):
    p = sigmoid(sum(inX*weights))
    if p > 0.5: return 1.0
    else: return 0.0
    
def colicTest():
    trainingSet = []; trainingLabels = []
    with open('./Ch05/horseColicTraining.txt') as ftTrain:
        for line in ftTrain.readlines():
            currLine = line.strip().split('\t')
            lineArr = []
            for i in range(21):
                lineArr.append(float(currLine[i]))
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[21])) # 标签
    trainWeights = stocGranAscent1(array(trainingSet), trainingLabels,500)
    errorCount = 0; numTestVec = 0.0
    with open('./Ch05/horseColicTest.txt') as frTest:
        for line in frTest.readlines():
            numTestVec += 1.0
            currLine = line.strip().split('\t')
            lineArr = []
            for i in range(21):
                lineArr.append(float(currLine[i]))
            if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
                errorCount += 1
    errorRate = float(errorCount) / numTestVec
    print('error rate is', errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after',numTests,'iterations the average over error rate is',errorSum/float(numTests))


if __name__ == "__main__":
    # print(os.getcwd())
    dataArr, labelMat = loadDataSet()
    result = gradAscent(dataArr, labelMat)
    print(result)
    plotBestFit(result.getA()) # matrix.getA()：矩阵转ndarray
