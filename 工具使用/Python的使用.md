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

