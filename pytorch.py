import torch 
import torchvision 
from torchvision import transforms, datasets 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
import matplotlib.pyplot as plt
import numpy as np 
import cv2 

# Load the data  

train = torch.utils.data.DataLoader(
	datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])),
	batch_size=10, shuffle=True) 

test = torch.utils.data.DataLoader(
	datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])),
	batch_size=10, shuffle=True) 



# Model Architecture 

class Net(nn.Module):

	def __init__(self):
		super().__init__()
		# 4 Linear hidden layers 
		self.fc1 = nn.Linear(784, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 64)
		self.fc4 = nn.Linear(64, 10)

	def forward(self, x):
		# Activation functions 
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.log_softmax(self.fc4(x), dim=1) 

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
		outputs = net(X.view(-1, 784))
		loss = criterion(outputs, y)
		loss.backward() 
		optimizer.step() 




# Save trained model 
PATH = './MNIST_net.pth'
torch.save(net.state_dict(), PATH) 

# Checking the accuracy of the model on our training set
correct = 0 
total = 0 

for data in train:
	X, y = data 
	output = net(X.view(-1, 784))
	for idx, i in enumerate(output):
		if torch.argmax(i) == y[idx]:
			correct += 1 
		total += 1 

print(f"Accuracy: {correct/total}")


for data in test:
	X, y = data

	plt.imshow(X[0].view(28,28))
	plt.show() 
	output = net(X[0].view(-1, 784))
	print(torch.argmax(output))

	plt.imshow(X[1].view(28,28))
	plt.show() 
	output = net(X[1].view(-1, 784))
	print(torch.argmax(output))

	plt.imshow(X[2].view(28,28))
	plt.show() 
	output = net(X[2].view(-1, 784))
	print(torch.argmax(output))

	break 

transform = transforms.Compose([transforms.ToTensor()])
img = cv2.imread('testnumber.png', cv2.IMREAD_GRAYSCALE)
#img = img/255
#img = Variable(torch.Tensor(img))











	




