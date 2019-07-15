import numpy as np

'''
dataMat, labels = knn.file2matrix('mycode/Ch02/datingTestSet2.txt')
'''
def file2matrix(filename):
    with open(filename) as f:
        lines = f.readlines()
    returnMat = np.zeros((len(lines), 3))
    index = 0
    labels = []
    for line in lines:
        currLine = line.strip().split('\t')
        returnMat[index,:] = currLine[0:3]
        labels.append(int(currLine[-1]))
        index += 1  
    return returnMat, labels
            

# inX：待分类的向量。dataSet：训练集。 lables：训练集标签
def classify0(inX, dataSet, labels, k):
    m= np.shape(dataSet)[0]
    diffMat = np.tile(inX, (m, 1)) - dataSet
    squareDiffMat = diffMat**2
    distances = np.sum(squareDiffMat, axis=1)**0.5 # 欧式距离
    sortedDistanceIndicies = np.argsort(distances)
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistanceIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda e:e[1], reverse=True)
    return sortedClassCount[0][0]

def plot(dataMat, labels):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,1], dataMat[:,2], 15.0*np.array(labels), 15.0*np.array(labels))
    plt.show()

# 归一化处理
def autoNorm(dataMat):
    # new = (old - min) / (max - min)
    minVal = dataMat.min(0)
    maxVal = dataMat.max(0)
    ranges = maxVal - minVal
    m = np.shape(dataMat)[0]
    normDataMat = np.zeros(np.shape(dataMat))
    normDataMat = (dataMat - np.tile(minVal, (m,1))) / np.tile(ranges, (m,1))
    return normDataMat, ranges, minVal

def datingClassTest(k=3):
    hoRation = 0.1
    datingDataMat, datingDataLabels = file2matrix('mycode/Ch02/datingTestSet2.txt')
    normMat, ranges, minVal = autoNorm(datingDataMat)
    m = np.shape(normMat)[0]
    numTestVects = int(m * hoRation)
    error = 0.0
    for i in range(numTestVects):
        classifierResult = classify0(normMat[i,:], normMat[numTestVects:], datingDataLabels[numTestVects:], k)
        print('the classifier came back with', classifierResult, ', the real answer is ', datingDataLabels[i])
        if classifierResult != datingDataLabels[i]: error += 1.0
    print('the total error rate is', error/numTestVects)