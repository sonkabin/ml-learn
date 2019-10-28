import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
'''保存和恢复神经网络
'''

base_path = 'DL/pytorch/data/tmp/'

x = torch.unsqueeze(torch.linspace(-1, 1, 50), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

net1 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)
optimizer = optim.SGD(net1.parameters(), lr = 0.5)
loss_function = torch.nn.MSELoss()
for i in range(100):
    predict = net1(x)
    loss = loss_function(predict, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def save():
    torch.save(net1, base_path+'net1.pkl') # 保存全部的网络信息
    torch.save(net1.state_dict(), base_path+'net1_params.pkl') # 只保存网络参数

    predict = net1(x)
    # plt
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x, y)
    plt.plot(x, predict.data.numpy(), 'r-', lw=5)

def load():
    net2 = torch.load(base_path+'net1.pkl')
    predict = net2(x)

    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x, y)
    plt.plot(x, predict.data.numpy(), 'r-', lw=5)

def load_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    path = base_path+'net1_params.pkl'
    net3.load_state_dict(torch.load(path)) # 对于只保存网络参数来说，需要提供相同的网络结构才能恢复
    predict = net3(x)

    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x, y)
    plt.plot(x, predict.data.numpy(), 'r-', lw=5)
    plt.show()

save()
load()
load_params()