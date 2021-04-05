import alpaca_trade_api as tradeapi
from alpaca_trade_api import StreamConn
import websocket, json
import random
import time, threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Full streaming and trading code through Alpaca

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

#### CHANGE THESE ### 

ACCESS_KEY = ""
SECRET_KEY = ""

TICKER = "FCEL" # Ticker to trade on
PERCENT = 1.01 # Percent growth to place a limit order at.
SHARES = 1 # Quantity of shares to trade.

#####################

socket = "wss://stream.data.alpaca.markets/v2/iex"
s2 = "wss://api.alpaca.markets"

ACCOUNT_URL = "{}/v2/account".format(ALPACA_BASE_URL)

current_order_id = ""

conn = tradeapi.stream2.StreamConn(ACCESS_KEY, SECRET_KEY, ALPACA_BASE_URL)

loop = asyncio.get_event_loop()

class TradeBot:
	def __init__(self):
		time.sleep(1)
		self.alpaca = tradeapi.REST(ACCESS_KEY, SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
		self.owned = False
		self.fill_price = 0.0
		self.finished = False
	def run(self):
		
		self.account = self.alpaca.get_account()
		
		if TICKER[3:] in self.alpaca.list_positions():
			self.owned = True
		
		if self.account.trading_blocked:
    			print('Account is currently restricted from trading.')

		print('${} is available as buying power.'.format(self.account.buying_power))
		
		self.ws = websocket.WebSocketApp(socket, on_open=authenticate, on_message=update)
		self.ws.run_forever()
		
	def limit_order(self, fill_price):
		limit = PERCENT * fill_price
		print("Submitting {}% profit limit order of {} shares for {} at {}".format(int(float(PERCENT - 1.0) * 100.0), SHARES, TICKER, limit))
		self.alpaca.submit_order(TICKER, SHARES, 'sell', 'limit', 'gtc', limit)

		
	def buy(self):
		print("Submitting buy order of {} shares for {}".format(SHARES, TICKER))
		self.alpaca.submit_order(TICKER, SHARES, 'buy', 'market', 'day')
		
	def sell(self):
		print("Submitting sell order of {} shares for {}".format(SHARES, TICKER))
		self.alpaca.submit_order(TICKER, SHARES, 'sell', 'market', 'day')
			
nightwing = TradeBot()
					
@conn.on(r'^account_updates$')
async def on_account_updates(conn, channel, account):
    print('account', account)

@conn.on(r'^trade_updates$')
async def on_trade_updates(conn, channel, trade):
    
    if trade.event == 'fill':

    	if trade.order['order_type'] == 'limit' or trade.order['order_type'] == 'sell':
    		print("Filled sell order of {} shares for {} at: {} per share".format(SHARES, TICKER, trade.price))
    		nightwing.owned = False
    		nightwing.finished = True
    	else:
    		print("Filled buy order of {} shares for {} at: {} per share".format(SHARES, TICKER, trade.price))
    		nightwing.fill_price = float(trade.price)
    		nightwing.limit_order(float(trade.price))

def ws_start():
	print("Starting streaming for trade and account updates on a new thread...")
	conn.run(['account_updates', 'trade_updates'])		
			
def work(w):
	w = websocket.WebSocketApp(socket, on_open=authenticate, on_message=update)
	w.run_forever()
	pass
			
ws_thread = threading.Thread(target=ws_start, daemon=True)
ws_thread.start()

def authenticate(ws):
	print("Authenticating Nightwing TradeBot with Alpaca...")
	
	auth_data = {
		"action":"auth",
		"key" : ACCESS_KEY,
		"secret": SECRET_KEY
	}
	
	ws.send(json.dumps(auth_data))
	
	channel_data = {
		"action" : "subscribe",
		"bars" : [TICKER]
	}
	
	ws.send(json.dumps(channel_data))
	
def update(ws, message):

	items = str(message)[1:-1]
	
	if json.loads(items)['T'] == 'success':
		print(json.loads(items)['msg'])
	elif json.loads(items)['T'] == 'subscription':
		print("Subscribed to {}".format(TICKER))
	elif json.loads(items)['T'] == 'b':
		data = json.loads(items)
		print("{} : Open {} Close {} High {} Low {}".format(data["S"], data["o"], data["c"], data["h"], data["l"]))
		
		chng = 0.0
		
		for pos in nightwing.alpaca.list_positions():
			if pos.symbol == TICKER[3:]:
				print(pos.unrealized_intraday_plpc)
				chng = pos.unrealized_intraday_plpc
		
		if float(data["c"]) >= float(data["o"]) and float(data["o"]) - float(data["l"]) > 0.01 and nightwing.owned == False and nightwing.finished == False:
			print("Buying {} on Doji Candle.".format(TICKER))
			nightwing.buy()
			nightwing.owned = True
			
		profit = nightwing.fill_price * PERCENT
		
		if float(data["l"]) > profit and float(data["o"]) > float(data["c"]) and nightwing.owned == True and nightwing.finished == False:
			
			if profit == 0.0:
				if chng > 0.0:
					nightwing.sell()
			print("Sold {} shares of{} for ${} profit. Finished trading today.".format(SHARES, TICKER, profit - nightwing.fill_price))
			nightwing.finished = True
	else:
		print("Nothing")
		
async def my_server(wbs):
	executor = ThreadPoolExecutor()
	await loop.run_in_executor(executor, work, wbs)

nightwing.run()


