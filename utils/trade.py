import requests

# Sample to get account information to place trades through Alpaca

BASE_URL = "https://paper-api.alpaca.markets"
ACCOUNT_URL = "{}/v2/account".format(BASE_URL)

API_KEY = ""
SECRET_KEY = ""

r = requests.get(ACCOUNT_URL, headers={'APCA-API-KEY-ID': API_KEY, 'APCA-SECRET-KEY': SECRET_KEY})

print(r.content)
