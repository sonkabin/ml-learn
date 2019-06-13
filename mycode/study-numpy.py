import numpy as np
# numpy分为两种类型：array和matrix
a = np.arange(15).reshape(3, 5)
print(a)
print(type(a))
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([1,2,2])
print(np.sum(a*b))

print(a.T[0])
print(b)
print(b.T)
print('--------------------')

c = np.nonzero([2,5,0,4,9])
print(c)