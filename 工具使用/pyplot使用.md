# matplotlib.pyplot


## pyplot.plot()

```python
plot(y) # 以0...N-1为横坐标，y为纵坐标绘制图（y是一维的）
plot(y) # 若y是二维的，按照列来绘图
plot(x, y) # x取一个，y取一个，故根据坐标点(x1, y1), (x2, y2)...画线
```




## 基本使用

### 一、（二、三）通用代码


```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2

plt.figure() # 定义一个图像窗口
```

### 二、坐标轴设置

```python
plt.plot(x, y1, color='red', linewidth=1, linestyle='-') # 根据x和y画曲线

plt.xlim((-1, 2)) # 设置坐标轴左右顶点
plt.ylim((-2, 3))
plt.xlabel('x')
plt.ylabel('y')

plt.xticks(np.linspace(-1, 2, 5)) # 设置刻度，-1到2分成5个刻度，中间有4个间隔。故每两个刻度间隔为3/4=0.75
plt.yticks([-2, 0.5, 3], [r'$bad$', r'$normal', r'$good$']) # 设置y具体刻度的名称

plt.show()

# 调整坐标轴位置
ax = plt.gca() # 获取当前坐标轴信息
ax.spines['right'].set_color('none') # 设置右边框为无
ax.spines['top'].set_color('none') 
ax.xaxis.set_ticks_position('bottom') # 设置x坐标刻度数字或名称的位置
ax.spines['bottom'].set_position(('data', 0)) # 设置边框位置，对于x来说，有top、bottom、both、default、none
ax.spines['left'].set_position(('data', 0)) # 设置y坐标轴的位置，在这里取data就是使其位于y=0处
```

### 三、图例

```python
l1, = plt.plot(x, y1, label='linear line') # l1, 以逗号结尾，因为plt.plot() 返回的是一个元素为1的列表
plt.legend(loc='upper right') # 在右上角显示label信息
plt.legend(handles=[l1, l2], labels=['up', 'down'],  loc='best') # 调整label的位置和名称

'''关于逗号的解释'''
a = [1]
b, = a # b = 1
c = a # c = [1]
```

### 四、标注

```python
plt.plot(x, y1)
# r'$$'：latex写法
# xycoords='data'：根据数据的值选位置；xytext=(+30, -30)：标注位置的描述；textcoords='offset points'：xy 偏差值
plt.annotate(r'$2x+1=%s$' % 3, xy=(1, 3), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2")) 
plt.text(-3.7, 3, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
         fontdict={'size': 16, 'color': 'r'}) # 在图中添加注释text，其中-3.7、3选取text位置
```

### 五、tick能见度

```python
# 在 plt 2.0.2 或更高的版本中, 设置 zorder 给 plot 在 z 轴方向排序
plt.plot(x, y, linewidth=10, zorder=1)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    # 在 plt 2.0.2 或更高的版本中, 设置 zorder 给 plot 在 z 轴方向排序
    # bbox：设置透明度相关参数；alpha设置透明度
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.7, zorder=2))
```

## 各种图



