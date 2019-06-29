import numpy as np

def loadDataSet(filename):
    dataMat = []
    with open(filename) as f:
        for line in f.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine))
            dataMat.append(fltLine)
    return dataMat

# 两个向量之间的欧式距离
def dist(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector1 - vector2, 2)))

# 选择k个簇质心
def randSelect(dataSet, k):
    m, n = np.shape(dataSet)
    centroids = np.mat(np.zeros([k, n]))
    for i in range(k):
        # 这里可能会有问题：两次随机选择，选择了相同的簇中心时，我觉得会出问题
        index = int(np.random.uniform(0, m))
        centroids[i,:] = dataSet[index]
    return centroids

'''
import mycode.Ch10.kMeans as kmeans
import numpy as np
dataMat = np.mat(kmeans.loadDataSet('./mycode/Ch10/testSet.txt'))

from importlib import reload
'''
def kMeans(dataSet, k, distMeas = dist, createCent = randSelect):
    m = np.shape(dataSet)[0]
    centroids = createCent(dataSet, k)
    clusterAssment = np.mat(np.zeros((m, 2)))
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(dataSet[i,:], centroids[j,:])
                if distJI < minDist:
                    minDist = distJI;   minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        print(centroids)
        # 重新计算簇中心
        for cent in range(k):
            ptsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = np.mean(ptsInCluster, axis=0) # axis=0表示沿矩阵列方向进行均值计算
    return centroids, clusterAssment

'''
dataMat = np.mat(kmeans.loadDataSet('./mycode/Ch10/testSet2.txt'))
centList, myNewAssement = kmeans.biKmeans(dataMat, 3)
'''
def biKmeans(dataSet, k, distMeas=dist):
    m = np.shape(dataSet)[0]
    centroid0 = np.mean(dataSet, axis=0).tolist()[0] # 所有点作为一个簇，计算簇中心
    clusterAssment = np.mat(np.zeros((m, 2)))
    centList = [centroid0]
    for i in range(m):
        clusterAssment[i,1] = distMeas(dataSet[i],np.mat(centroid0))**2 # 初始化样本到簇中心的距离
    while len(centList) < k:
        lowestSSE = np.inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A == i)[0],:] # 属于当前簇的样本
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) # 当前簇的样本划分为两个簇
            sseSplit = np.sum(splitClustAss[:,1]) # 划分后两个簇的平方误差和
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A != i)[0], 1]) # 其他簇的平方误差和
            print('sseSplit:', sseSplit,', and notSplit:', sseNotSplit)
            if sseSplit + sseNotSplit < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy() # 平方误差和在此处赋值
                lowestSSE = sseNotSplit + sseSplit
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0], 0] = len(centList) # 新增簇
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0], 0] = bestCentToSplit # 保留原来的簇
        print('the bestCentToSplit is', bestCentToSplit)
        print('the len of bestClustAss is', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] # 更新原来的簇中心
        centList.append(bestNewCents[1,:].tolist()[0]) # 添加新增簇的簇中心
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss # 将原来的簇中心替换为新的两个簇中心
    return np.mat(centList), clusterAssment

# 余弦定理计算经纬度之间的距离
def distSLC(vector1, vector2): 
    a = np.sin(vector1[0,1]*np.pi/180) * np.sin(vector2[0,1]*np.pi/180)
    b = np.cos(vector1[0,1]*np.pi/180) * np.cos(vector2[0,1]*np.pi/180) * \
        np.cos((vector1[0,0]-vector2[0,0])*np.pi/180)
    return np.arccos(a+b)*6371.0

def clusterClubs(numClust=5):
    import matplotlib
    import matplotlib.pyplot as plt
    datList = []
    with open('./mycode/Ch10/places.txt') as f:
        for line in f.readlines():
            lineArr = line.split('\t')
            datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    
    fig = plt.figure()
    rect = [0.1,0.1,0.8,0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('./mycode/Ch10/Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:,0].A == i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], \
                    ptsInCurrCluster[:,1].flatten().A[0],\
                    marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()