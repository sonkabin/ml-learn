import numpy as np
import torch

# Hyper parameters
BATCH_SIZE = 64
LR_G = 0.0001 # 生成器学习速度
LR_D = 0.0001 # 判别器学习速度
N_IDEAS = 5 # 想象成生成艺术作品的主意
ART_COMPONENTS = 15 # 想象成画一副图用15个关键点
PAINT_POINTS = np.vstack(np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE))