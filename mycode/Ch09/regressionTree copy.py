import numpy as np

'''
import mycode.Ch09.regressionTree as regTree
import numpy as np
myData = regTree.loadDataSet('mycode/Ch09/ex00.txt')
myMat = np.mat(myData)
regTree.createTree(myMat)
myData1 = regTree.loadDataSet('mycode/Ch09/ex0.txt')
myMat1 = np.mat(myData1)
regTree.createTree(myMat1)

myData2 = regTree.loadDataSet('mycode/Ch09/ex2.txt')
myMat2 = np.mat(myData2)
myTree = regTree.createTree(myMat2, ops=(0, 1))
myDataTest = regTree.loadDataSet('mycode/Ch09/ex2test.txt') 
myMatTest = np.mat(myDataTest)
regTree.prune(myTree, myMatTest)

myMat2 = np.mat(regTree.loadDataSet('mycode/Ch09/exp2.txt'))
regTree.createTree(myMat2, regTree.modelLeaf, regTree.modelErr, (1, 10))
'''

def loadDataSet(filename):
    dataArr = []
    with open(filename) as f:
        for line in f.readlines():
            curLine = line.strip().split('\t')
            floatLine = list(map(float, curLine))
            dataArr.append(floatLine)
    return dataArr

# 将数据集合按照给定特征和特征值切分成两个子集
def binSplitDataSet(dataMat, feature, value):
    mat1 = dataMat[np.nonzero(dataMat[:, feature] > value)[0]]
    mat2 = dataMat[np.nonzero(dataMat[:, feature] <= value)[0]]
    return mat1, mat2
 
def regLef(dataMat):
    return np.mean(dataMat[:, -1])

def regErr(dataMat):
    return np.var(dataMat[:, -1]) * np.shape(dataMat)[0] # 均方差*样本总数=总方差

def chooseBestFeature(dataMat, leafType, errType, ops):
    totalS = ops[0];    totalN = ops[1] # totalS是允许的误差下降值，totalN是允许的最少样本数
    if len(set(dataMat[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataMat)
    m, n = np.shape(dataMat)
    S = errType(dataMat)
    bestS = np.inf; bestIndex = 0;  bestValue = 0
    for featureIndex in range(n - 1):
        for splitValue in set(dataMat[:,featureIndex].T.tolist()[0]):
            mat1, mat2 = binSplitDataSet(dataMat, featureIndex, splitValue)
            if np.shape(mat1)[0] < totalN or np.shape(mat2)[0] < totalN:    continue
            newS = errType(mat1) + errType(mat2)
            if newS < bestS:
                bestS = newS;   bestIndex = featureIndex;   bestValue = splitValue
    if (S - bestS) < totalS: # 误差减小不大
        return None, leafType(dataMat)
    mat1, mat2 = binSplitDataSet(dataMat, bestIndex, bestValue)
    if np.shape(mat1)[0] < totalN or np.shape(mat2)[0] < totalN:   
        return None, leafType(dataMat)
    return bestIndex, bestValue

def createTree(dataMat, leafType=regLef, errType=regErr, ops=(1,4)):
    feature, value = chooseBestFeature(dataMat, leafType, errType, ops)
    if feature == None: return value
    returnTree = {}
    returnTree['spInd'] = feature
    returnTree['spVal'] = value
    lSet, rSet = binSplitDataSet(dataMat, feature, value)
    returnTree['left'] = createTree(lSet, leafType, errType, ops)
    returnTree['right'] = createTree(rSet, leafType, errType, ops)
    return returnTree

# 剪枝
def isTree(obj):
    return type(obj).__name__ == 'dict'

def getMean(tree):
    if isTree(tree['left']):    tree['left'] = getMean(tree['left'])
    if isTree(tree['right']):   tree['right'] = getMean(tree['right'])
    return (tree['left'] + tree['right'])/2.0

def prune(tree, testData):
    if np.shape(testData)[0] == 0:  return getMean(tree) # 没有测试数据进行塌陷处理
    if isTree(tree['left']) or isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):    tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):   tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']): # 左右都是叶节点
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
                       np.sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print('merge')
            return treeMean
        else:   return tree
    else:   return tree

# 模型树（多变量决策树）
def linerSolve(dataMat):
    m, n = np.shape(dataMat)
    X = np.mat(np.ones((m, n)));  Y = np.mat(np.ones((m,1)))
    X[:, 1:n] = dataMat[:, 0:n-1];  Y = dataMat[:,-1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def modelLeaf(dataMat):
    ws, X, Y = linerSolve(dataMat)
    return ws

def modelErr(dataMat):
    ws, X, Y = linerSolve(dataMat)
    yHat = X * ws
    return np.sum(np.power(yHat - Y, 2))

'''
模型树与回归树比较

'''
def regTreeEval(model, inData):
    return float(model)

def modelTreeEval(model, inData): # inData是n维，model的第一个权值为1
    n = np.shape(inData)[1]
    X = np.mat(np.ones(1, n+1))
    X[:, 1:n+1] = inData
    return float(X * model)

def treeForecast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):    return modelEval(tree)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForecast(tree['left'], inData, modelEval)
        else:   return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForecast(tree['right'], inData, modelEval)
        else:   return modelEval(tree['right'], inData)

def createForecast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForecast(tree, testData[i], modelEval)
    return yHat
