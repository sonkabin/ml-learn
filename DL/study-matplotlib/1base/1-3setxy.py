import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2

fig = plt.figure() # 定义一个图像窗口
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=1, linestyle='-') # 根据x和y画曲线
plt.xlim((-1, 2)) # 设置坐标轴左右顶点
plt.xticks(np.linspace(-1, 2, 5)) # 设置
plt.yticks([-2, 0.5, 3], [r'$bad$', r'$normal', r'$good$']) # 设置y具体刻度的名称
plt.xlabel('X') # x轴的label

ax = plt.gca()
print(ax)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('default')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

plt.show() # 展示