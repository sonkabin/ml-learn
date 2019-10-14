import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module): # 继承

    # https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernal
        # 输入为32*32*1，则cov1后为30*30*6，max-pool1后为15*15*6，cov2后为13*13*16，max-pool2后为6*6*16
        self.conv1 = nn.Conv2d(1, 6, 3) 
        self.conv2 = nn.Conv2d(6, 16, 3) 
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120) # 全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x): # override
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        print('wdnmd', x.size()[1:])
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print('wdnmd,wsdd', x.size()[1:])
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_feature = 1
        for s in size:
            num_feature *= s
        return num_feature

net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
for i in range(len(params)):
    print('i:', i, ',size:', params[i].size())

inpt = torch.randn(1, 1, 32, 32)
output = net(inpt) # 输入经过神经网络之后的10个输出值
# print(output)

# 定义损失函数
target = torch.randn(10) # for example, 假设是真实值
target = target.view(1, -1) # make it the same shape as output
criterion = nn.MSELoss() # mean-squared error
loss = criterion(output, target) # 真实值与预计值之间的损失
print(loss)
print('うん,c')
# compute graph
# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#       -> view -> linear -> relu -> linear -> relu -> linear
#       -> MSELoss
#       -> loss
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad()     # zeroes the gradient buffers of all parameters
print('conv1.bias.grad before backward:', net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward:', net.conv1.bias.grad)

# w更新
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
# in your training loop:
optimizer.zero_grad() # zero the gradient buffers
output = net(inpt)
loss = criterion(output, target)
loss.backward()
optimizer.step() # Does the update