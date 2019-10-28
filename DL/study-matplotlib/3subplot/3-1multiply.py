import matplotlib.pyplot as plt

plt.figure()
plt.subplot(221) # 创建2行2列，当前位置1
plt.plot([0,1],[0,1])
plt.subplot(222)
plt.plot([0,1], [0, 2])
plt.subplot(223)
plt.plot([0, 1], [0, -1])
plt.subplot(224)
plt.plot([0, 1], [0, 1])

plt.show()

plt.figure()
plt.subplot(211)
plt.plot([0, 1], [0, 1])
plt.subplot(234) # 由于一开始是2行1列，第一幅子图占据了第一行，这时划分为2行3列，第一行3个位置被第一幅子图占据了，只能从第4个位置开始
plt.plot([0, 1], [0, 1])
plt.subplot(235)
plt.plot([0, 1], [0, 1])
plt.subplot(236)
plt.plot([0, 1], [0, 1])

plt.show()