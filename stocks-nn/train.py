import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import random
import copy
import csv

# This program is used to train a model for the neural network on historical stock ticker data.

INFILE='sample_nvda_data_file.csv'

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.inputl = nn.Linear(4, 8)
        self.hidden  = nn.Linear(8, 64)
        self.hidden1  = nn.Linear(64, 8)
        self.output = nn.Linear(8, 4)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.best_model_weights = copy.deepcopy(self.state_dict())
        self.best_acc = 0.0 
        self.running_loss = 0.0
        self.running_correct = 0
        
    def forward(self, x):
        x = self.inputl(x)
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.hidden1(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x
        
def choice(bar):

	gt = [0.0, 0.0, 0.0, 1.0]
	if bar[1] >= bar[0] and bar[0] - bar[3] > 0.1: # Buy
		gt = [1.0, 0.0, 0.0, 0.0]
	elif bar[0] - bar[3] < 0.5 and bar[2] - bar[1] < 0.5:# Hold
		gt = [0.0, 1.0, 0.0, 0.0]
	elif bar[2] - bar[0] < 0.5 and bar[1] - bar[3] < 0.5 and bar[0] - bar[1] > 0.0: # Sell
		gt = [0.0, 0.0, 1.0, 0.0]
	else:
		gt = [0.0, 0.0, 0.0, 1.0]
	return gt
	
n = Network()
#n.load_state_dict(torch.load('best_custom_model_weight.pth'))
n.train()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(n.parameters(), lr=0.001, momentum=0.7)

scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

reader = csv.reader(open(INFILE, 'rt', encoding='utf8'))

chunk, chunksize = [], 100

counter = 1

def process_chunk(chuck):

    optimizer.zero_grad()
    for i in range(len(chuck)):
    	with torch.set_grad_enabled(True):
    		line = [float(chuck[i][0]), float(chuck[i][1]), float(chuck[i][2]), float(chuck[i][3])]
    		rand = torch.tensor(line)
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
    	n.running_loss += loss.item() * 4
    	n.running_correct += torch.sum(preds == target.data)
    	scheduler.step()
    	epoch_loss = n.running_loss / 100
    	epoch_acc = n.running_correct.double() / 100
    	print(loss.item())
    	if epoch_acc > n.best_acc: 
    		n.best_acc = epoch_acc
    		n.best_model_weights = copy.deepcopy(n.state_dict())
    		torch.save(n.best_model_weights, 'best_custom_model_weight.pth')
    		

for i, line in enumerate(reader):
    if (i % chunksize == 0 and i > 0):
        process_chunk(chunk)
        print("------------Starting Batch {}------------".format(i/chunksize))
        del chunk[:] 
    chunk.append(line)

process_chunk(chunk)
