import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import random
import copy

import alpaca_trade_api as tradeapi
from alpaca_trade_api import StreamConn
import websocket, json
import random
import time, threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

# This code is the combination of both the Alpaca streaming/trading code and the neural network created to train on ticker values.

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

ACCESS_KEY = "PKA75YWQ04BA33UNY09Z"
SECRET_KEY = "gZXqbVXq8TtGKMx2CYlPWN97ywe9qjQU9RJCZh2E"

TICKER = "AM.SNDL"

socket = "wss://alpaca.socket.polygon.io/stocks"
s2 = "wss://data.alpaca.markets"

ACCOUNT_URL = "{}/v2/account".format(ALPACA_BASE_URL)

current_order_id = ""

conn = tradeapi.stream2.StreamConn(ACCESS_KEY, SECRET_KEY, ALPACA_BASE_URL)

loop = asyncio.get_event_loop()

class TradeBot:
	def __init__(self):
		self.network = Nightwing()
		self.network.train()
		#self.network.load_state_dict(torch.load('best_custom_model_weight.pth'))
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.network.parameters(), lr=0.002, momentum=0.9)
		self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
		self.best_model_weights = copy.deepcopy(self.network.state_dict())
		self.best_acc = 0.0 
		self.running_loss = 0.0
		self.running_correct = 0
		self.alpaca = tradeapi.REST(ACCESS_KEY, SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
	def run(self):
	
		self.optimizer.zero_grad()
		self.position = []
		
		self.account = self.alpaca.get_account()
		
		if len(self.alpaca.list_positions()) > 0:
			self.position = self.alpaca.get_position(TICKER[3:])
		
		#
		
		if self.account.trading_blocked:
    			print('Account is currently restricted from trading.')

		# Check how much money we can use to open new positions.
		print('${} is available as buying power.'.format(self.account.buying_power))
		
		self.ws = websocket.WebSocketApp(socket, on_open=authenticate, on_message=update)
		self.ws.run_forever()
		#asyncio.run(my_server(self.ws))
		
	def limit_order(self, fill_price):
		limit = 1.01 * fill_price
		print("Submitting 1% profit limit order for {} at {}".format(TICKER, limit))
		self.alpaca.submit_order(TICKER[3:], 1000, 'sell', 'limit', 'gtc', limit)
		
	def sell(self):
		print("Submitting sell order for {}".format(TICKER))
		self.alpaca.submit_order(TICKER[3:], 1000, 'sell', 'market', 'gtc')

		
	def buy(self):
		print("Submitting buy order for {}".format(TICKER))
		self.alpaca.submit_order(TICKER[3:], 1000, 'buy', 'market', 'day')

class Nightwing(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.inputl = nn.Linear(4, 16)
        # Output layer, 10 units - one for each digit
        self.hidden  = nn.Linear(16, 16)
        self.output = nn.Linear(16, 4)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.inputl(x)
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x
        
    def choice(self, bar):
    	#print(bar)
    	gt = [0.0, 0.0, 0.0, 1.0]
    	if bar[1] >= bar[0] and bar[0] - bar[3] > 0.1: # Buy
    		gt = [1.0, 0.0, 0.0, 0.0]
    		print("Ideal choice is to buy {}".format(TICKER))
    	elif bar[0] - bar[3] < 0.5 and bar[2] - bar[1] < 0.5:# Hold
    		gt = [0.0, 1.0, 0.0, 0.0]
    		print("Ideal choice is to hold {}".format(TICKER))
    	elif bar[2] - bar[0] < 0.5 and bar[1] - bar[3] < 0.5 and bar[0] - bar[1] > 0.0: # Sell
    		gt = [0.0, 0.0, 1.0, 0.0]
    		print("Ideal choice is to sell {}".format(TICKER))
    	else:
    		gt = [0.0, 0.0, 0.0, 1.0] # Stay
    		print("Ideal choice is to not to buy {}".format(TICKER))
    	return gt

master = TradeBot()

@conn.on(r'^account_updates$')
async def on_account_updates(conn, channel, account):
    print('account', account)

@conn.on(r'^trade_updates$')
async def on_trade_updates(conn, channel, trade):
    #print('trade', trade)
    
    if trade.event == 'fill':
    	print("Filled order at: {}".format(trade.price))

def ws_start():
	print("Starting streaming for trade and account updates on a new thread...")
	conn.run(['account_updates', 'trade_updates'])		
			
def work(w):
	w = websocket.WebSocketApp(socket, on_open=authenticate, on_message=update)
	w.run_forever()
	pass
			
ws_thread = threading.Thread(target=ws_start, daemon=True)
ws_thread.start()

def make_decision(layer):

	index = np.argmax(layer.detach().numpy())

	a = master.alpaca.list_positions()
	if index == 0 or (index == 1 and len(a) == 0):
		# Buy
		master.buy()
		print("Network chose to buy {}".format(TICKER))
	elif index == 2 and len(a) > 0 and float(master.position.change_today) > 0.0:
		# Sell
		master.sell()
		print("Network chose to sell {}".format(TICKER))
	else:
		print("Hold/stay")

def authenticate(ws):
	print("Authenticating Nightwing TradeBot with Alpaca...")
	
	auth_data = {
		"action":"auth",
		"params" : ACCESS_KEY
	}
	
	ws.send(json.dumps(auth_data))
	
	channel_data = {
		"action" : "subscribe",
		"params" : TICKER
	}
	
	ws.send(json.dumps(channel_data))
	
def update(ws, message):
	items = str(message)[1:-1]
	print("-----------RECIEVING UPDATE--------------")
	if json.loads(items)['ev'] == 'status':
		print(json.loads(items)['message'] )
	elif json.loads(items)['ev'] == 'AM':
		data = json.loads(items)
		print("{} : Open {} Close {} High {} Low {}".format(data["sym"], data["o"], data["c"], data["h"], data["l"]))
		inp = torch.tensor([float(data["o"]), float(data["c"]), float(data["h"]), float(data["l"])], requires_grad=True)
		
		balance_change = float(master.account.equity) - float(master.account.last_equity)
		print("Today's profit: {}".format(balance_change))
		
		with torch.set_grad_enabled(True):
			out = master.network.forward(inp) # make dataframe
			output = torch.zeros(1, 4)
			make_decision(out)
			for j in range(4):
				output[0][j] = out[j]
			_, preds = torch.max(output, 1)
			truth = master.network.choice(inp)
			target = np.argmax(truth)
			target = torch.tensor([target])
			loss = master.criterion(output, target)
			print("Network loss: {}".format(loss.item()))
			loss.backward()
			master.optimizer.step()
		master.running_loss += loss.item() * 4
		master.running_correct += torch.sum(preds == target.data)
	
		master.scheduler.step()
	
		epoch_loss = master.running_loss / 100
		epoch_acc = master.running_correct.double() / 100
		print('Loss: {: .4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
		if epoch_acc > master.best_acc:
			print("Network accuracy has improved!")
			master.best_acc = epoch_acc
			master.best_model_weights = copy.deepcopy(master.network.state_dict())
			torch.save(master.best_model_weights, 'best_custom_model_weight.pth')
	else:
		print("Something else")
		
async def my_server(wbs):
	executor = ThreadPoolExecutor()
	await loop.run_in_executor(executor, work, wbs)
	
master.run()


