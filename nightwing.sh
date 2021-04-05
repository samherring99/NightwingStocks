#!/bin/bash
while true; do
	echo "-------Running Nightwing TradeBot--------"
	
	# SIMPLE TRADING
	python stream_and_trade.py
	
	# NEURAL NETWORK TRADING
	#python utils/stream.py
	
	echo "Connection Closed"
	
	sleep 5
done
