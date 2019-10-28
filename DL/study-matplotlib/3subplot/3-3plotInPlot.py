import numpy as np
import matplotlib.pyplot as plt

# 初始化figure
fig = plt.figure()

# 创建数据
x = [1, 2, 3, 4, 5, 6, 7]
y = [1, 3, 4, 2, 5, 8, 6]

left, bottom, width, height = 0.1, 0.1, 0.8, 0.8 # 代表整个figure坐标系的百分比
ax1 = fig.add_axes([left, bottom, width, height])
ax1.plot(x, y, 'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('title')

left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(y, x)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('inside')

left, bottom, width, height = 0.6, 0.2, 0.25, 0.25
ax3 = fig.add_axes([left, bottom, width, height])
ax3.plot(y, x)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('inside2')

plt.show()
