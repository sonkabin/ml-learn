import numpy as np

def loadDataSet():
    return [[1,3,4], [2,3,5], [1,2,3,5], [2,5]]

def createC1(dataSet): # 构建大小为1的所有候选项集合，如{{1}, {2}, {3}, {4}, {5}}
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))

def scanD(dataSet, C, minSupport): # C为候选项集列表。作用为：搜寻C中的频繁项集并返回
    D = list(map(set, dataSet)) # 以set表示的数据集D
    asCnt = {}
    for transaction in D:
        for candidate in C:
            if candidate.issubset(transaction):
                if candidate not in asCnt:    asCnt[candidate] = 1
                else:   asCnt[candidate] += 1
    num = len(D)
    retList = []
    supportData = {}
    for key in asCnt:
        support = asCnt[key]/num
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData
    
'''
import mycode.Ch11.apriori as apriori
from importlib import reload
dataSet = apriori.loadDataSet()
C1 = apriori.createC1(dataSet)
L1, supportData = apriori.scanD(dataSet, C1, 0.5)
'''

def aprioriGen(Lk, k): # Lk为频繁项集列表，k为项集元素个数。作用：构建k个项组成的候选项集列表，如k=3，则{{1,2,3}, {1,2,4}, {2,3,4}}
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort();  L2.sort()
            if L1 == L2: # 若前k-2个元素都相等，则将两个集合合并成一个大小为k集合
                retList.append(Lk[i] | Lk[j]) # 取并集
    return retList

def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    L1, supportData = scanD(dataSet, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(dataSet, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

# 接下来是从频繁项集中挖掘关联规则
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)): 
        for freqSet in L[i]: # 忽略单元素项集，因为只有两个及以上的项集才能构建规则
            H1 = [frozenset([item]) for item in freqSet] # 取频繁集的单元素组成列表
            if (i > 1): # 对于项集元素超过2个的，需要进一步合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, br, minConf):
    prunedH = [] # 存放规则右件
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] # A-->B：可信度=support(A,B)/support(A)
        if conf >= minConf:
            print(freqSet-conseq,'--->', conseq, 'conf:', conf)
            br.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, br, minConf):
    m = len(H[0])
    print(freqSet, H, m)
    if len(freqSet) > (m+1): # 检测频繁集能否大到能移除大小为m的子集
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br, minConf)
        if len(Hmp1) > 1: # 若不止一条规则满足要求，则进一步组合这些规则
            rulesFromConseq(freqSet, Hmp1, supportData, br, minConf)

'''
L, supportData = apriori.apriori(dataSet)
rules = apriori.generateRules(L, supportData, minConf=0.7)
rules = apriori.generateRules(L, supportData, minConf=0.5)
'''