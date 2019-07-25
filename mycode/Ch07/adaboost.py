import numpy as np

def loadDataSet():
    dataMat = np.mat(([1, 2.1],
                      [2, 1.1],
                      [1.3, 1],
                      [1, 1],
                      [2, 1]))
    classLabels = [1, 1, -1, -1, 1]
    return dataMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, thresIneq): # stump: 树桩
    returnArr = np.ones((np.shape(dataMatrix)[0], 1))
    if thresIneq == 'lt':
        returnArr[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        returnArr[dataMatrix[:, dimen] > threshVal] = -1.0
    return returnArr

def buildStump(dataArr, classLabels, D):
    dataMat = np.mat(dataArr);  labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMat)
    numSepts = 10.0;    bestStump = {}; bestClassEstimate = np.zeros((m, 1))
    minError = np.inf
    for i in range(n): 
        rangeMin = dataMat[:, i].min(); rangeMax = dataMat[:, i].max()
        septSize = (rangeMax - rangeMin)/numSepts
        for j in range(-1, int(numSepts) + 1): # 阈值可以在整个取值范围之外
            for inequal in ['lt', 'gt']: # 将小于（或大于）阈值的设置为-1
                threshVal = (rangeMin + float(j) * septSize)
                predictedVals = stumpClassify(dataMat, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                print('split: dim', i, ', thresh', threshVal, 'thresh inequal:', inequal, ', the weighted error is', weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClassEstimate = predictedVals.copy()
                    bestStump['dim'] = i # 以第几个特征值进行决策树构建
                    bestStump['thresh'] = threshVal # 树桩判定值
                    bestStump['ineq'] = inequal # 树桩判断条件
    return bestStump, minError, bestClassEstimate

'''
import mycode.Ch07.adaboost as ada
import numpy as np
from importlib import reload
dataMat, classLabels = ada.loadDataSet()
D = np.mat(np.ones((5, 1))/5)
ada.buildStump(dataMat, classLabels, D)
'''

'''
classifierArray = ada.adaBoostTrainDS(dataMat, classLabels, 9)
'''
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1))/m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print('D:', D.T)
        alpha = float(0.5 * np.log((1 - error)/max(error, 1e-16))) # max(error, 1e-16)确保不会发生除零溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst:', classEst.T)
        # w_m+1,i = w_m,i/Z_m * exp(-alpha_m * y_i * yHat)
        # Z_m = \sum_i=1^{N} w_mi * exp(-alpha_m * y_i * yHat), 其中令yHat = G_m(x_i)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst # f_m(x) = f_(m-1)(x) + alpha_m * yHat
        print('aggClassEst', aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m ,1)))
        errorRate = aggErrors.sum() / m
        print('total error:', errorRate)
        if errorRate == 0:  break
    return weakClassArr

# ada.adaClassify([[0,0]], classifierArray)
# ada.adaClassify([[5, 5], [0,0]], classifierArray)
def adaClassify(data, classifierArray):
    dataMat = np.mat(data)
    m = np.shape(data)[0]
    aggClassEst = np.mat(np.zeros((m ,1)))
    for i in range(len(classifierArray)):
        classEst = stumpClassify(dataMat, classifierArray[i]['dim'], classifierArray[i]['thresh'], classifierArray[i]['ineq'])
        aggClassEst += classifierArray[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)

