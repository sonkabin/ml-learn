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

