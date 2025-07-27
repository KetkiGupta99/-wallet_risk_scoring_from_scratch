from os import read
import csv
import requests
import json
import requests
from dotenv import load_dotenv
import os
import time

with open(r"D:\Credit_Score_Scratch\Wallet id.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    wallet_addresses = [line.strip().lower() for line in f if line.strip()]
    #print(wallet_addresses)
all_wallet_data = {}

# Fetching Transaction History from compound V2 or V3
'''
for i, wallet in enumerate(wallet_addresses, 1):
    print(f"Fetching {i}/{len(wallet_addresses)}: {wallet}")
query = """
    {
      account(id: "%s") {
        id
        tokens {
          symbol
          cTokenBalance
          totalUnderlyingSupplied
          totalUnderlyingRedeemed
          totalUnderlyingBorrowed
          totalUnderlyingRepaid
        }
      }
    }
    """ % wallet
response = requests.post(
        'https://api.studio.thegraph.com/query/32021/compound-v3-usdc/version/latest',
        json={'query': query}
    )

data = response.json()
all_wallet_data[wallet] = data

#print(all_wallet_data)
'''
    
# Fetch Transaction History from Etherscan API Key
load_dotenv() 
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

def get_etherscan_transactions(address, api_key, retries=3):
    base_url = "https://api.etherscan.io/api"
    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "asc",
        "apikey": api_key
    }
    for attempt in range(retries):
      try:
          response = requests.get(base_url, params=params)
          response.raise_for_status()  
          data = response.json()
          if data["status"] == "1":
              return data["result"]
          else:
              print(f"Etherscan API error for address {address}: {data['message']} (Attempt {attempt+1})")
              time.sleep(1)
              return []
      except requests.exceptions.RequestException as e:
          print(f"Request error for address {address}: {e} (Attempt {attempt+1})")
          time.sleep(1)
    return []
all_wallet_transactions = {}

for i, wallet in enumerate(wallet_addresses, 1):
    print(f"Fetching transactions for {i}/{len(wallet_addresses)}: {wallet}")
    transactions = get_etherscan_transactions(wallet, ETHERSCAN_API_KEY)
    all_wallet_transactions[wallet] = transactions
    time.sleep(0.3)

print("\nFinished fetching transactions.")

if all_wallet_transactions:
    first_wallet = list(all_wallet_transactions.keys())[0]
    print(f"Data for {first_wallet}:")
    print(all_wallet_transactions[first_wallet][:5]) 
else:
    print("No transaction data fetched.")

with open("compound_wallet_transactions.json", "w") as f:
    json.dump(all_wallet_transactions, f, indent=2)

print("Transactions saved to 'compound_wallet_transactions.json'")



   