# NightwingStocks

This repository contains all the code for the stock market side of the Nightwing TradeBot program. 

The code is comprised of 3 main parts: 

1. Simple stocks trading: The main Python file in the directory handles both paper and real money trading through buying on a Doji candle and selling at a specified profit percent by placing a limit order.

2. Neural Network based stock trading: This is my first attempt at using AI to learn stock patterns down to a minute level and trading on a single learned ticker to see the outcome. I realized later that a time-series dependent network like an LSTM would produce more more meaningful results, so expect that work some time in the future.

3. Utilities: This directory contains early files used to learn both the API for streaming/trading stocks and the data flow to create a trading bot.

## Requirements
- Python 3.7+
- alpaca_trade_api
- Pytorch
- Pandas
- NumPy

## Notes Before Using This Code
- Alpaca is a free stock trading API as well as a broker, more info can be found at https://alpaca.markets/, and you must create an account to get authentication keys before running this code.
- Alpaca can trade both paper and real money. Always test with paper money before putting anything in to Alpaca.  Simply switch the API endpoints from "paper-api..." to just "api..." to use your real funds.
- According to Alpaca's rules on PDT (Pattern Daytrader Protection), you may place up to 3 day trades within 5 consecutive business days, also found here https://alpaca.markets/docs/trading-on-alpaca/user-protections/. This means, if you place 3 trades (a buy and then a sell) within a single day, you must wait 5 days before placing another day trade. If your account has over $25,000, this rule can be disregarded. Swing trades do not count towards this total. Also, pay attention to the fact that this rule applies to paper trading as well, but you usually start with $100,000 so unless you set it to be less or lose it all you. can trade many more times per day.

## Running Nightwing TradeBot for Stocks

Once your Alpaca account is setup, requirements are installed, and API authentication keys are set, you are ready to start trading!

### Simple Automated Trading

Make sure your ticker, profit percent, and quantity fields in stream_and_trade.py are set how you want, then run

_python stream_and_trade.py_

to start streaming ticker bar data and waiting for a Doji candle. Once the program places a buy and limit order, it can quit itself and will not need any further interaction.

If you have over $25,000 or are paper trading with that amount and want to see as many trades as possible, run

_bash nightwing.sh_

to allow the code to continuously loop. (Quit with CTRL+C)

Check Alpaca account page to see order history and profits.

### Neural Network Based Trading: stocks-nn

The steps to train and use a model with this code on a single ticker are as follows:
- Use history.py with a given stock ticker and time range to download minute level training data into a .csv file.
- Use train.py to train a Pytorch model on the historical data.
- Use learn.py to trade according to our trained model's decisions.

Change ticker and start/end dates for desired data range in history.py, then run

_python history.py_

to pull all minute level data in that time. Then, change the input file to the previously generated .csv data file in train.py and run

_python train.py_

to start training. When complete, set the TICKER value in learn.py to AM. followed by your ticker and run

_python learn.py_

to begin neural network trading.

The websockets loop does tend to close frequently, so if you want an extended time period to have the neural network trade paper stocks you can comment out the line

_python stream_and_trade.py_ 

from nightwing.sh and uncomment the line python learn.py in that file, then save and run 

_bash nightwing.sh_

to run the neural network based trade loop forever (Quit with CTRL+C).

## DISCLAIMER

**Do not attempt to use any this program with real money if you are not comfortable losing all of it, as you would be doing so at your own risk. **

I am not claiming responsibility for any money lost or PDT flags.

I can not advocate that any part of this repository will gurantee you profits, nor do I think the neural network is anywhere close to a stage where it should be trading with real money. That being said, if you know what you are getting into, please feel free to use and adapt any part of this project for your own personal needs or interests with what I've provided here, and don't hesistate to reach out with any questions to samherrring99@gmail.com.



