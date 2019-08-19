import numpy as np

class treeNode:
    def __init__(self, name, numOccur, parentNode):
        self.name = name
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}
    def inc(self, numOccur):
        self.count += numOccur
    def disp(self, ind=1):
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)
    
def createTree(dataSet, minSupport=1): # dataSet为字典
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable.keys()):
        if headerTable[k] < minSupport:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:   return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0] # headerTable在上面改成了[count, None]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p:p[1], reverse=True)] # 按出现次数进行排序
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children: # 第一个元素项是否作为子节点存在
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 更新头指针表
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1: # 迭代调用自身，每次去掉列表的第一个元素，直到剩余最后一个元素
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)
    
def updateHeader(nodeToTest, targetNode):
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def loadSimpleDat():
    simpleDat = [['r', 'z', 'h', 'j', 'p'],
                 ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                 ['z'],
                 ['r', 'x', 'n', 'o', 's'],
                 ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                 ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpleDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

'''
import mycode.Ch12.fpGrowth as fp
from importlib import reload
simpleDat = fp.loadSimpleDat()
initSet = fp.createInitSet(simpleDat)
myFPtree, myHeaderTab = fp.createTree(initSet, 3)
myFPtree.disp()
'''

def ascendTree(leafNode, prefixPath): # 从叶节点回溯树
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode): # 以给定元素项结尾的所有路径
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

'''
fp.findPrefixPath('x', myHeaderTab['x'][1])
fp.findPrefixPath('r', myHeaderTab['r'][1])
'''

# 暂时先放一下了，该函数没有理解，整体的思路也没有很好的理解
def mineTree(inTree, headerTable, minSupport, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[0])]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPatBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPatBases, minSupport)
        if myHead !=  None:
            mineTree(myCondTree, myHead, minSupport, newFreqSet, freqItemList)
    
'''
freqItems = []
fp.mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
'''