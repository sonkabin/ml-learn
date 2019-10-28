import numpy as np
import matplotlib.pyplot as plt

def height_function(x, y):
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)

n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y) # 二维平面中将每个x和y对应起来组成栅格

plt.contourf(X, Y, height_function(X, Y), 8, alpha=.75, cmap=plt.cm.hot) # alpha:透明度 color map通过height_function(X, Y)的值中寻找暖色调
C = plt.contour(X, Y, height_function(X, Y), 8, colors='black', linewidth=.5)
plt.clabel(C, inline=True, fontsize=10) # 将Label画在线里面
plt.xticks(())
plt.yticks(())

plt.show()