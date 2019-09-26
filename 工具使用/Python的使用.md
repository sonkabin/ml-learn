# Python的使用

## 导入模块

1. 同一目录

   `import a`

2. 导入子目录中的模块

   ```python
   '''
   a.py
   b
     | c.py
     | d.py
   '''
   # a.py文件中
   import b.c as bc
   from b import d
   ```


## reload()

```python
from imp import reload #__main__:1: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
# 故使用如下模块
from importlib import reload
```



## 字典排序

sorted(iterable,key,reverse)

iterable可以是dict.items()、dict.keys()、dict.values()

```python
sorted(dict.values()) # 等价于 sorted(dict.items(), key=lambad e:e[1])
```



## list的extend和append

```python
l = [1,2,3]
a = l[:1]
a.append(l[2:]) # [1, [3]]
l = [1,2,3]
a = l[:1]
a.extend(l[2:]) # [1, 3]
```



## 字典的keys()

返回dict_keys对象，py3中不能直接用索引

1. list(d.keys())[0]
2. next(iter(d.keys()))



## Python对象

python只有对象（函数也是），函数有\_\_doc\_\_属性，函数的属性处理有两种方式

```python
def foo():
    pass
# 方式1
setattr(foo, 'a', 12)
print(getattr(foo, 'a'))
# 方式2
foo.a = 1
print(foo.a)
```



## [0] * 3的意义

[0] * 3 等于 [0, 0, 0]



## 获取文件编码

```python
import chardet
with open('1.txt', 'rb') as f:
    print(chardet.detect(f.read())['encoding']) 
```



## 正则表达式

re模块。使用原生字符串可以解决反斜杠的困扰，如`r'\d'`匹配一个数字，`r'\\'`匹配反斜杠



## 集合

```python
a = set([1,2,3,4])
b = set([2,5])
print(a - b) # {1,3,4}，即取差集
print(a | b) # {1,2,3,4,5},即取并集
```



## range

```python
for i in range(0, 10, 2):	print(i) # 0, 2, 4, 6, 8
for i in range(0, 10, 3):	print(i) # 0, 3, 6, 9

a = [1,2,3,4,5,6,7,8,9,10]
c = [a[i:i+3]for i in range(0, 10, 3)] # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
```



## zip

```python
'''
也适用于numpy
zip(iter1 [,iter2 [...]]) --> zip object

Return a zip object whose .__next__() method returns a tuple where the i-th element comes from the i-th iterable argument. The .__next__() method continues until the shortest iterable in the argument sequence is exhausted and then it raises StopIteration.
翻译：它返回zip对象，它的__next__()方法：返回iter1, iter2, ...中取相同索引的元素组成tuple，当某个iter达到最后一个元素时停止
'''
for i in zip([1,2,7], [4,5,8,9], [6,6,6]):	print(i) # (1,4,6),(2,5,6),(7,8,6)
for i in zip([1,2], [4,5,8,9], [6,6,6]):	print(i) # (1,4,6),(2,5,6)
```



## 广播

```python
(m, n), (1, n), (m, 1) are matrix
(m, n) +-*/ (1, n) ==> (m, n) # (1, n)扩展为(m, n)
(m, n) +-*/ (m, 1) ==> (m, n) # (m, 1)扩展为(m ,n)
```



## 类

定义类

```python
class a(object):
    def walk(self):
        print('a walk')
```

继承类

```python
class b(a):
    def walk(self): # 重写父类方法
        print('b walk')
    def talk(self): # 自定义方法
        print('b talk')
```



## enumerate()函数

参数：

- sequence -- 一个序列、迭代器或其他支持迭代对象
- start -- 下标起始位置

将一个可遍历的数据对象，列出数据和数据下标

```python
seq = ['one', 'two', 'three']
for i, element in enumerate(seq, 0):
    print(i, element) # 0 one		1 two 		2 three
```

