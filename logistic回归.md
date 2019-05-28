# logistic回归

## 仍有疑惑之处

1. 改进的随机梯度上升算法中，alpha的值更新的准则

## 假设函数

$$
h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}
$$

## 代价函数

$$
J(\theta)=\frac{1}{m}\sum_{i=1}^{m}cost(h_\theta(x^{(i)})-y^{(i)})
$$


$$
cost(h_\theta(x^{(i)}),y^{(i)})=-(y^{(i)}\log{h_\theta(x^{(i)})}+(1-y^{(i)})\log{(1-h_\theta(x^{(i)}))})
$$

## 梯度上升/下降算法


### 思路

```
重复直到收敛：
	计算整个数据集的梯度
	使用alpha*gradient更新回归系数向量
返回回归系数
```

Repeat{

​	$\theta_j:=\theta_j+\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j ​$

}

### 部分说明

- 首先，对于上述公式来说，alpha这个步长是要试出来的，对于某个具体情况，m是常数，若取alpha=0.01，若m很大则等于白给，故将alpha/m作为一个整体，看作步长（这也是Ng视频中求出梯度后将1/m省略不写的原因）
- 梯度等于第j个特征*每个样本的误差 之和
- 如有必要，再证明一次公式，其中$(h_\theta(x))'=h_\theta(x)*(1-h_\theta(x))​$

### 实现

```python
from numpy import *
import os
# 将文件中的数据存到数组中
def loadDataSet():
    dataMat = []; labelMat = []
    with open('./Ch05/testSet.txt') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            # 两个特征X1,X2，即z=w0*x0+w1*x1+w2*x2，而x0=1
            # 由假设函数可知，两个分类的分界线是z=0
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

# sigmoid函数
def sigmoid(inX):
    return 1.0 / (1+exp(-inX))

# 梯度上升算法：求最大值。对应的梯度下降算法求最小值
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn) # ndarray转matrix
    labelMat = mat(classLabels).transpose() # 等价于mat(classLabels).T
    m, n = shape(dataMatrix) #矩阵的维数
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1)) # numpy.ndarray
    for i in range(maxCycles):
        h = sigmoid(dataMatrix*weights) # 矩阵可直接和数组相乘
        error = labelMat - h 
        weights = weights + alpha*dataMatrix.transpose()*error # dataMatrix.transpose()*error：将第j个特征和误差相乘并求和。若不转置，则变成了第i个样本的每个特征*误差，和思路中的公式不同了
        '''
        梯度下降算法
        error = h - labelMat
        weights = weights - alpah*dataMatrix.T*error
        '''
    return weights

# 画图
def plotBestFit(weights) :
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat) # 转成ndarray
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111) # 画在1行1列，第一块
    # scatter(x,y,s=size,c=color,marker=形状)，画散点图
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x) / weights[2] # z=0是两个分类的分界线，由此解出x1与x2的关系
    ax.plot(x,y) # 根据x和y画线
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

if __name__ == "__main__":
    # print(os.getcwd())
    dataArr, labelMat = loadDataSet()
    result = gradAscent(dataArr, labelMat)
    print(result)
    plotBestFit(result.getA()) # matrix.getA()：矩阵转ndarray

```

此方法一次迭代需要进行300次乘法。你能说出为什么吗？

## 随机梯度上升

**梯度上升算法在数据集小时计算量仍旧过大，需改进**

### 思路

```
对数据集中的每个样本：
	计算该样本的梯度
	使用alpha*gradient更新回归系数值
返回回归系数
```

### 公式

$$
\theta_j:=\theta_j-\alpha(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j
$$

```python
# 随机梯度上升算法
def stocGranAscent0(dataArr, classLabels, iterNum=1):
    m,n = shape(dataArr)  # dataArr需要为numpy.ndarray
    alpha = 0.01
    weights = ones(n)
    for j in range(iterNum):
        for i in range(m):
            h = sigmoid(sum(dataArr[i]*weights))
            error = classLabels[i] - h
            weights = weights + alpha*error*dataArr[i]
    return weights
```

## 改进的随机梯度上升

```python
# 改进的随机梯度上升算法：1.每次迭代更新alpha 2.随机选取样本更新回归系数：防止系数出现周期性波动（因为部分样本不能正确分类）
def stocGranAscent1(dataArr, classLabels, iterNum=150):
    m, n =shape(dataArr)
    weights = ones(n)
    for j in range(iterNum):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.001 # 这里留有疑问
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataArr[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha*error*dataArr[randIndex]
            del(dataIndex[randIndex])
    return weights
```

## 正则化

$$
J(\theta)=-\frac{1}{m}[\sum_{i=1}^{m}y^{(i)}logh_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]+\frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^{2}
$$

则

$\theta_0:=\theta_0-\frac{\alpha}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_0$

$\theta_j:=\theta_j(1-\frac{\alpha\lambda}{m})-\frac{\alpha}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$

