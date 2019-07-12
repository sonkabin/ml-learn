import numpy as np
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
'''
from importlib import reload

myMat=np.mat(svdRec.loadExData())
myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
myMat[3,3]=2
myMat
'''
def testSVD():
    data = loadExData()
    U, Sigma, VT = la.svd(data)
    print(Sigma)
    Sig3 = np.mat([[Sigma[0],0,0], [0,Sigma[1],0], [0,0,Sigma[2]]])
    X = U[:,:3] * Sig3 * VT[:3,:]
    print(X)

# 欧式距离计算相似度
def euclidSim(vectorA, vectorB):
    return 1.0 / (1.0 + la.norm(vectorA - vectorB))

# 皮尔逊相关系数计算相似度
def pearsSim(vectorA, vectorB):
    if len(vectorA) < 3:    return 1.0
    return 0.5 + 0.5 * np.corrcoef(vectorA, vectorB, rowvar=False)[0][1]

# 余弦相似度计算相似度
def cosSim(vectorA, vectorB):
    num = float(vectorA.T * vectorB) # 这里假设了向量是列向量
    denom = la.norm(vectorA) * la.norm(vectorB) 
    return 0.5 + 0.5 * (num / denom)

# 基于物品的相似度
def standEstimate(dataMat, user, item, simMeans):
    n = np.shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 :    continue
        '''
        1. 不会碰到自身，因为没有评价过，在上一步就跳过了
        2. 第i个物品评价过的所有用户，第item个物品评价过的所有用户，两者取交集
        '''
        overlap = np.nonzero(np.logical_and(dataMat[:,j].A >0, dataMat[:,item].A > 0))[0]
        if len(overlap) == 0:   similarity = 0
        else:   similarity = simMeans(dataMat[overlap,item], dataMat[overlap,j])
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:   return 0.0
    else:   return ratSimTotal / simTotal

def recommend(dataMat, user, N=3, simMeans=cosSim, estMethod=standEstimate):
    # np.nonzero(dataMat[user,:].A == 0) ：返回第几行第几列
    unratedItems = np.nonzero(dataMat[user,:].A == 0)[1]
    if len(unratedItems) == 0:  return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, item, simMeans)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj:jj[1], reverse=True)[:N]

# 下面将展示SVD
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
'''
dataMat = np.mat(svdRec.loadExData2())
'''
def svdEstimate(dataMat, user, item, simMeans):
    n = np.shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    Sigma4 = np.mat(np.eye(4) * Sigma[:4])
    # 1. xformedItems到底是什么??? 2. 吴的视频中，降维用的是U_reduce^T * x，测试了一下感觉也没多大变化
    xformedItems = dataMat.T * U[:,:4] * Sigma4.I 
    for j in range(n):
        userRating = dataMat[user, j] # ???
        if userRating == 0: continue
        similarity = simMeans(xformedItems[item,:].T, xformedItems[j,:].T)
        # print('the', item, 'and', j, 'similarity is', similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:    return 0
    else:   return ratSimTotal/simTotal