import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 50), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

plt.ion()
plt.show()

# 定义神经网络的第二种方法
net2 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)
# print(net2)

net = Net(1, 10 ,1)
optimizer = optim.SGD(net.parameters(), lr = 0.5)
loss_function = torch.nn.MSELoss()
for i in range(100):
    predict = net(x)
    loss = loss_function(predict, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 5 == 0:
        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, predict.data.numpy(), 'r-',lw=5)
        
        # the following annotation code is wrong
        # plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size':20, 'color':'blue'})
        # num = loss.item()
        # plt.text(0.5, 0, 'Loss=%.4f' % num, fontdict={'size':20, 'color':'blue'})
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size':20, 'color':'blue'})
        plt.pause(0.2)

plt.ioff()
plt.show()