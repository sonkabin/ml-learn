import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2

plt.figure()
l1, = plt.plot(x, y1)
l2, = plt.plot(x, y2, color='red', linewidth=1, linestyle='--', label='red')
# plt.legend(loc='upper right')
plt.legend(handles=[l1, l2], labels=['up', 'down'],  loc='best')
plt.show()