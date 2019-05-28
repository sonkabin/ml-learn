import matplotlib.pyplot as plt
import numpy as np
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0,6,0,20])
plt.xlabel('cnb')
plt.ylabel('cnm')
plt.show()

data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', c='c', s=30, data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()

plt.plot([1,2,3])
plt.subplot(221)
plt.subplot(222)
plt.show()