import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as Data

LR = 0.01
EPOCH = 12
BATCH_SIZE = 32

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))
# plt.figure()
# plt.scatter(x, y)
# plt.show()
dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

netSGD = Net()
netMometumn = Net()
netRMSprop = Net()
netAdam = Net()
nets = [netSGD, netMometumn, netRMSprop, netAdam]

optSGD = optim.SGD(netSGD.parameters(), lr=LR)
optMometumn = optim.SGD(netMometumn.parameters(), lr=LR, momentum=0.8)
optRMSprop = optim.RMSprop(netRMSprop.parameters(), lr=LR, alpha=0.9)
optAdam = optim.Adam(netAdam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [optSGD, optMometumn, optRMSprop, optAdam]

lossFuction = torch.nn.MSELoss()
lossesHistory = [[], [], [], []]

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(loader):
        for net, opt, lossHis in zip(nets, optimizers, lossesHistory):
            predict = net(b_x)
            loss = lossFuction(predict, b_y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            lossHis.append(loss.data.numpy())
    
labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, lossHis in enumerate(lossesHistory):
    plt.plot(lossHis, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()