# matplotlib.pyplot

```python
fig = plt.figure() # 创建一个figure
ax = fig.add_subplot(111) # 画在1行1列，第一块， 共九块
ax.scatter(x, y, s=30, c='red', marker='s') # 横纵坐标，画散点图，且x要按顺序排列
ax.plot(x,y) # 根据x和y画线
plt.xlabel('X1') # x轴的名称
plt.show() # 展示
```

## pyplot.plot()

```python
plot(y) # 以0...N-1为横坐标，y为纵坐标绘制图（y是一维的）
plot(y) # 若y是二维的，按照列来绘图
```

