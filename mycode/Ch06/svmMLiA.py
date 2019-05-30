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
    b = 0; m, n = np.shape(dataMatrix)
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
                # eta = Kii + Kjj - pow(Kij,2)
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