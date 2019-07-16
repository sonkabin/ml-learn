# coding=utf-8 #
import numpy as np
from math import log2

# x^i = (x^i_1, x^i_2, ..., x^i_n), dataSet^i = (x^i, label)
def calcShannonEnt(dataSet):
    num = len(dataSet)
    labels = {}
    for featureVec in dataSet:
        label = featureVec[-1]
        if label not in labels.keys():
            labels[label] = 0
        labels[label] += 1
    shannonEnt = 0.0
    for key in labels.keys():
        prob = labels[key] * 1.0 / num
        shannonEnt -= prob * log2(prob)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# dataSet: 待划分的数据集, axis: 划分特征, value: 特征取值
def splitDataSet(dataSet, axis, value):
    returnDataSet = []
    for featureVec in dataSet:
        if featureVec[axis] == value:
            reducedFeatureVec = featureVec[:axis]
            reducedFeatureVec.extend(featureVec[axis+1:])
            returnDataSet.append(reducedFeatureVec)
    return returnDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0]) - 1
    dataSetLen = len(dataSet)
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeature):
        featureList = [example[i] for example in dataSet] # 某个特征的所有取值
        uniqueVals = set(featureList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            newEntropy += len(subDataSet) / dataSetLen * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList): # 投票表决
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount, key=lambda e:e[1], reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): # 类别完全相同，停止划分
        return classList[0]
    if len(dataSet[0]) == 1: # 所有的特征都已经遍历过了
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel: {}}
    del(labels[bestFeature])
    featureVals = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featureVals)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return myTree

'''
画图
'''
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
def plot(myTree):
    import matplotlib.pyplot as plt 
    # 定义文本框格式
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks= [], yticks=[])
    plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(myTree))
    plotTree.totalD = float(getTreeDepth(myTree))
    plotTree.xOff = -0.5/plotTree.totalW;   plotTree.yOff = 1.0
    plotTree(myTree, (0.5, 1.0), '')
    plt.show()
    

def plotNode(nodeText, centerPt, parentPt, nodeType):
    # 定义箭头格式
    arrow_args = dict(arrowstyle="<-")
    plot.ax1.annotate(nodeText, xy=parentPt, xycoords='axes fraction', xytext=centerPt, 
                            textcoords='axes fraction', va="center", ha="center", bbox=nodeType,arrowprops=arrow_args)

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotMidText(centerPt, parentPt, textString):
    xMid = (parentPt[0] - centerPt[0]) / 2.0 + centerPt[0]
    yMid = (parentPt[1] - centerPt[1]) / 2.0 + centerPt[1]
    plot.ax1.text(xMid, yMid, textString)

def plotTree(myTree, parentPt, nodeText):
    numLeafs = getNumLeafs(myTree)
    firstStr = list(myTree.keys())[0]
    centerPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(centerPt, parentPt, nodeText)
    plotNode(firstStr, centerPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff -= 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],centerPt,str(key))
        else:
            plotTree.xOff += 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), centerPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), centerPt, str(key))
    plotTree.yOff += 1.0/plotTree.totalD

'''
分类
'''
def classify(inputTree, featureLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featureIndex = featureLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featureIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featureLabels, testVec)
            else:   classLabel = secondDict[key]
    return classLabel

# 存储  filename='mycode/Ch03/classifierStorage.txt'
def storeTree(inputTree, filename):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(inputTree, f)
def loadTree(filename):
    import pickle
    with open(filename, 'rb') as f:
        tree = pickle.load(f)
    return tree

# 由于没有剪枝，存在过拟合问题
def megane():
    with open('mycode/Ch03/lenses.txt') as f:
        lines = f.readlines()
    lenses = [inst.strip().split('\t') for inst in lines]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenseTree = createTree(lenses, lensesLabels)
    return lenseTree