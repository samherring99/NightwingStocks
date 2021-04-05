import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import random
import copy

# Sample code for a neural network training on random ticker values

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.inputl = nn.Linear(4, 16)
        
        self.hidden  = nn.Linear(16, 16)
        self.output = nn.Linear(16, 4)
         
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x):
        
        x = self.inputl(x)
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x
        
def choice(bar):
	
	gt = []
	
	if bar[0] > 100.0:
		# Buy
		gt = [1.0, 0.0, 0.0, 0.0]
	elif bar[0] > 200.0:
		# Hold
		gt = [0.0, 1.0, 0.0, 0.0]
	elif bar[0] > 300.0:
		# Sell
		gt = [0.0, 0.0, 1.0, 0.0]
	else:
		# Do not buy
		gt = [0.0, 0.0, 0.0, 1.0]
		
	return gt
	
n = Network()
#n.load_state_dict(torch.load('best_custom_model_weight.pth'))
n.train()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(n.parameters(), lr=0.002, momentum=0.9)

scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

best_model_weights = copy.deepcopy(n.state_dict())
best_acc = 0.0 
running_loss = 0.0
running_correct = 0

for i in range(100):

	print(i)
	
	optimizer.zero_grad()
	
	with torch.set_grad_enabled(True):

		rand = torch.tensor([random.random() * 1000.0, random.random() * 1000.0, random.random() * 1000.0, random.random()  * 1000.0], requires_grad=True)
		o = n.forward(rand)

		output = torch.zeros(1, 4)
		
		
	
		for j in range(4):
			output[0][j] = o[j]
			
		_, preds = torch.max(output, 1)
	
		truth = choice(rand)

		target = np.argmax(truth)
	
		target = torch.tensor([target])
	
		loss = criterion(output, target)
	
		loss.backward()
	
		optimizer.step()
		
	running_loss += loss.item() * 4
	running_correct += torch.sum(preds == target.data)
	
	scheduler.step()
	
	epoch_loss = running_loss / 100
	epoch_acc = running_correct.double() / 100
	print('Loss: {: .4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
	
	print(loss.item())
		
	if epoch_acc > best_acc: 
		best_acc = epoch_acc
		best_model_weights = copy.deepcopy(n.state_dict())
		torch.save(best_model_weights, 'best_custom_model_weight.pth')

