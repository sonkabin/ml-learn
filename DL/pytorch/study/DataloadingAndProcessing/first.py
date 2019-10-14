import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings('ignore') # 抑制警告

# plt.ion() # interactive mode

def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:,0], landmarks[:, 1], s=10, marker='.', c='b')

# a csv file contains: image_name, part_0_x, part_0_y, ... , part_67_x, part_67_y
landmarks_frame = pd.read_csv('DL/pytorch/data/faces/face_landmarks.csv')
n = 60
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)
print('Image name:{}'.format(img_name))
print('Landmarks shape:{}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))
    
image = io.imread(os.path.join('DL/pytorch/data/faces/', img_name))
plt.figure()
show_landmarks(image, landmarks)
plt.show()
