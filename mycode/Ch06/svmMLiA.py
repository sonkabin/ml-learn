import numpy as np

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    with open(fileName) as f:
        for line in f.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(np.random.uniform(0,m))
    return j
# 调整大于H和小于L的alpha
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter=10):
    dataMatrix = np.mat(dataMatIn); labelMatrix = np.mat(classLabels).T
    b = 0; m = np.shape(dataMatrix)[0]
    alphas = np.mat(np.zeros((m, 1)))
    iteration = 0
    while(iteration < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            gXi = float(np.multiply(alphas, labelMatrix).T * (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = gXi - float(labelMatrix[i])
            if ((labelMatrix[i]*Ei < -toler ) and (alphas[i] < C)) or ((labelMatrix[i]*Ei > toler) and alphas[i] > 0):
                j = selectJrand(i, m)
                gXj = float(np.multiply(alphas, labelMatrix).T * (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = gXj - float(labelMatrix[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMatrix[i] != labelMatrix[j]):
                    L = max(0, alphaJold - alphaIold)
                    H = min(C, C + alphaJold - alphaIold)
                else:
                    L = max(0, alphaJold + alphaIold - C)
                    H = min(C, alphaJold + alphaIold)
                if L == H: print('L == H'); continue
                # eta = Kii + Kjj - Kij*2
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - \
                    dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print('eta>=0'); continue
                alphas[j] -= labelMatrix[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print('j not moving enough'); continue
                alphas[i] += labelMatrix[j] * labelMatrix[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i,:]*dataMatrix[i,:].T - \
                    labelMatrix[j]*(alphas[j] - alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i,:]*dataMatrix[j,:].T - \
                    labelMatrix[j]*(alphas[j] - alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print('iter:', iteration, 'i:', i, ', pairs changed', alphaPairsChanged)
        if (alphaPairsChanged == 0): iteration += 1
        else: iteration = 0
        print('iteration number:', iteration)
    return b, alphas

# 完整版SMO实现

class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.toler = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        # eCache第一列给出eCache是否有效的标志位，第二列给出实际的E
        self.eCache = np.mat(np.zeros((self.m,2)))

def calEk(oS, k):
    gXK = float(np.multiply(oS.alphas,oS.labelMat).T * (oS.X*oS.X[k,:].T)) + oS.b
    Ek = gXK - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei] # 将Ei设为有效（即表示已经计算好了）
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0] # 返回非0元素的索引
    if len(validEcacheList) > 1:
        # 返回改变最大的
        for k in validEcacheList:
            if k == i: continue
            Ek = calEk(oS,k)
            deltaE = abs(Ei-Ek)
            if deltaE > maxDeltaE:
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    # 如果是第一次循环，则随机选择一个。可用其他方法代替
    else:
        j = selectJrand(i, oS.m)
        Ej = calEk(oS, j)
    return j, Ej

def updateEk(oS, k):
    Ek = calEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS): # 内层循环
    Ei = calEk(oS, i)
    if (oS.labelMat[i]*Ei < -oS.toler and oS.alphas[i] < oS.C) or (oS.labelMat[i]*Ei > oS.toler and oS.alphas[i] > 0):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, alphaJold - alphaIold)
            H = min(oS.C, oS.C + alphaJold - alphaIold)
        else:
            L = max(0, alphaJold + alphaIold - oS.C)
            H = min(oS.C, alphaJold + alphaIold)
        if L == H: print('L == H'); return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0: print('eta>=0'); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):   print('j not moving enough');   return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - \
             oS.labelMat[j]*(oS.alphas[j] - alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - \
             oS.labelMat[i]*(oS.alphas[j] - alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if 0 < oS.alphas[i] and oS.C > oS.alphas[i]:  oS.b = b1
        elif 0 < oS.alphas[j] and oS.C > oS.alphas[j]: oS.b = b2
        else:   oS.b = (b1 + b2) / 2.0
        return 1
    else:   return 0

'''
import mycode.Ch06.svmMLiA as svm
dataArr, labelArr = svm.loadDataSet('./mycode/Ch06/testSet.txt')
b,alphas = svm.smoP(dataArr,labelArr,0.6,0.001,40)

from importlib import reload
'''
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).T, C, toler)
    iteration = 0
    entireSet = True; alphaPairsChanged = 0
    while (iteration < maxIter) and ((alphaPairsChanged > 0) or entireSet): # 达到迭代次数，或者遍历整个数据集后没有alpha发生变化，则结束
        alphaPairsChanged = 0
        if entireSet: # 遍历整个数据集
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            print('fullSet, iteration:', iteration, 'i',i,',pairs changed',alphaPairsChanged)
            iteration += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0] # ?
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
            print('non-bound, iteration:',iteration,'i:',i,',pairs changed',alphaPairsChanged)
            iteration += 1
        if entireSet:   entireSet = False
        elif (alphaPairsChanged == 0):  entireSet = True
        print('iteration number:', iteration) # 注意：此处的iteration和simpleSmo中的不同，这里是一次循环过程记为一次迭代
    return oS.b, oS.alphas
                 

def calWs(alphas,dataArr,classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w
