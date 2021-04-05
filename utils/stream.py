import websocket, json

# Sample to stream ticker data from Alpaca

TICKERS = "Q.MSFT"

socket = "wss://alpaca.socket.polygon.io/stocks"

def authenticate(ws):
	print("opened")
	
	auth_data = {
		"action":"auth",
		"params" : ""
	}
	
	ws.send(json.dumps(auth_data))
	
	channel_data = {
		"action" : "subscribe",
		"params" : TICKERS
	}
	
	ws.send(json.dumps(channel_data))
	
def update(ws, message):
	print(message)

ws = websocket.WebSocketApp(socket, on_open=authenticate, on_message=update)
ws.run_forever()
