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

landmarks_frame = pd.read_csv('DL/pytorch/study/faces/face_landmarks.csv')
n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)
print('Image name:{}'.format(img_name))
print('Landmarks shape:{}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks))

def show_landmarks(img_name, landmarks):
    plt.figure()
    plt.imshow(io.imread(os.path.join('DL/pytorch/study/faces/', img_name)))
    plt.scatter(landmarks[:,0], landmarks[:, 1], s=10, marker='.', c='b')
    

show_landmarks(img_name, landmarks)
plt.show()
