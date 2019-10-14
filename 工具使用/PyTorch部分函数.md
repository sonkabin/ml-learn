# PyTorch部分知识

## 函数

### torch.norm()

默认为L2范式

公式为$\|x\|_p = (x_1^p +x_2^p + ... + x_n^p)^{\frac{1}{p}}$

### Tensor.backward()

计算梯度

```python
'''根据导数知识，可知 y=4x
'''
x = torch.randn(3, requires_grad=True)
y = x * x * 2
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad) # 结果为 4x_1 * 0.1, 4x_2, 4x_3 * 0.0001
```

### torchvision.transforms.Normalize(object)

归一化操作
$$
x_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

```python
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 每个通道 均值为0.5，标准差为0.5
```

### Pillow（PIL）

图像处理主要采用的库：skimage, opencv-python, Pillow (PIL)。



### size()和view()

```python
>>> a = tensor([6,7,6,8])
>>> a.size() # tensor([6,7,6,8])
>>> a.size(0) # 4
```



view()：作用等于numpy的reshape()



## torch.max()

取最大值。添加了dim参数后，将返回`(values, indices)` 

Args:

- input
- dim：取值范围[-2, 1]，其中1，-1相同，按行取；0，-2相同，按列取。
- ...



## 类

### CrossEntropyLoss

整合`nn.LogSoftmax` 和:`nn.NLLLoss` 进一个类

用于**分类**问题

### NLLLoss

The negative log likelihood loss. 

用于**分类**问题。输入为每个类的log-probability，要得到该输入可在最后一层之后加入`LogSoftmax`层。若用`CrossEntropyLoss`，则不需要加入额外的层。

```python
m = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.tensor([1, 0, 4])
output = loss(m(input), target) # 加入LogSoftmax层
output.backward()
```

### LogSoftmax

对softmax取自然对数：ln(softmax)



### 

