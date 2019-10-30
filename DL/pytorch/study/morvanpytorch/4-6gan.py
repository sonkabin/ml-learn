import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Hyper parameters
BATCH_SIZE = 64
LR_G = 0.0001 # 生成器学习速度
LR_D = 0.0001 # 判别器学习速度
N_IDEAS = 5 # 想象成生成艺术作品的主意
ART_COMPONENTS = 15 # 想象成画一副图用15个关键点
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])
# 验证np.vstack()和np.array()在此情况下是否是一样的效果
# PAINT_POINTS = np.array([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

def artist_works(): # 生成著名画家的画
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a*np.power(PAINT_POINTS, 2) + (a-1)
    paintings = torch.from_numpy(paintings).float()
    return paintings

# 展示著名作家的画的范围
# plt.plot(PAINT_POINTS[0], 2*np.power(PAINT_POINTS[0], 2) + 1, c='b', lw=3, label='upper bound')
# plt.plot(PAINT_POINTS[0], 1*np.power(PAINT_POINTS[0], 2) + 0, c='r', lw=3, label='lower bound')
# plt.legend(loc='best')
# plt.show()

G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS)
)

D = nn.Sequential(
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid() # 计算是著名画家的画的概率
)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()   # something about continuous plotting

for step in range(10000):
    artist_painting = artist_works()
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS) # 生成BATCH_SIZE * N_IDEAS的 tensor
    G_paintings = G(G_ideas) # 批量画画

    prob_artist0 = D(artist_painting) # 由最后的激活函数可知，该概率期望越大越好
    prob_artist1 = D(G_paintings) # 概率期望越小越好
    
    # 最大化log(prob_artist) + log(1-prob_artist)，但在torch中反向传播用的是最小化误差
    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1 - prob_artist1)) # 为什么用mean？因为是批量操作
    # G的性能要提升，则需要让D分辨不了，也就是说增加prob_artist1的概率期望
    G_loss = torch.mean(torch.log(1 - prob_artist1))

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)      # retain_graph 这个参数是为了再次使用计算图纸
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward() # 要用到D网络的参数，具体的理解需要画图
    opt_G.step()

    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3));plt.legend(loc='upper right', fontsize=10);plt.draw();plt.pause(0.01)

plt.ioff()
plt.show()