import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Load the data  

train = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])), batch_size=10, shuffle=True)

test = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])), batch_size=10, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
    	#conv1
        x = self.conv1(x)
        x = F.relu(x)
        #conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #dropout1
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        #fc1
        x = self.fc1(x)
        x = F.relu(x)
        #droput2
        x = self.dropout2(x)
        #fc2
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

net = Net() 

# Training model 
epochs = 3 
optimizer = optim.Adam(net.parameters(), lr=1e-3) 
criterion = F.nll_loss 

for epoch in range(epochs):

	for i, data in enumerate(train, 0):
		# data is a list of [X, y]
		X, y = data 
		# Analagous to net.zero_grad() as optimizer is given net.parameters()
		optimizer.zero_grad() 
		outputs = net(X)
		loss = criterion(outputs, y)
		loss.backward() 
		optimizer.step() 

# Save trained model 
PATH = './MNIST_net.pth'
torch.save(net.state_dict(), PATH) 