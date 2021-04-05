import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime
import calendar
import csv

# This program downloads minute level ticker barset data for a given range of time from Alpaca.

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

ACCESS_KEY = ""
SECRET_KEY = ""

TICKER = 'NVDA'

NY = 'America/New_York'

api = tradeapi.REST(ACCESS_KEY, SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

start=pd.Timestamp('2020-01-01 9:30', tz=NY)
end=pd.Timestamp('2020-01-01 16:00', tz=NY)

with open('{}_data_file.csv'.format(TICKER.lower()), mode='w') as data_file:

	for x in range(3):

		for j in range(12):

			for i in range(32):

				print(start)
		#print(i)


				barset = api.get_barset(TICKER, 'minute', start=start.isoformat(), end=end.isoformat(), limit=1000)

				aapl_bars = barset[TICKER]

				counter = 0
			
			
				tickerwriter = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				for item in barset[TICKER]:
					counter += 1
					tickerwriter.writerow([item.o, item.c, item.h, item.l])
					print(counter)	
		
				start = start.replace(day=(i+2))
				end = end.replace(day=(i+2))
	
				if calendar.monthrange(start.year, start.month)[1] == (i+2):
			
					if j == 11:
						start = start.replace(month=12, day=1)
						end = end.replace(month=12, day=1)
					else:
						start = start.replace(month=(j+2), day=1)
						end = end.replace(month=(j+2), day=1)
					break;
		start = start.replace(year=2020+x+1, month=1)
		end = end.replace(year=2020+x+1, month=1)

