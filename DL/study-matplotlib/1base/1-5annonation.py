import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 50)
y = 2*x + 1

plt.figure(num=1, figsize=(8, 5))
plt.plot(x, y)

x0 = 1
y0 = 2*x0 + 1
plt.plot([x0, x0], [0, y0], 'k--', linewidth=2.5) # plot是x取一个，y取一个，故坐标点为(x0, 0), (x0, y0)
plt.scatter([x0,], [y0,], s=50, color='b')

# 添加标注, r'$$' latex写法
plt.annotate(r'$2x+1=%s$'% y0, xy=(x0, y0), xycoords='data', xytext=(+3, -30), textcoords='offset points',
             fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=2'))
plt.text(-3.7, 3, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
         fontdict={'size': 16, 'color': 'r'})

plt.show()