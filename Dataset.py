import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

import pandas as pd
import re
from tqdm import tqdm
import time
from Utils import *
import json
import numpy as np

def GetAddressRelatedTransactions(type='tornado_cash', json_path='Dataset/tornado_neighbor_addresses.json'):
    if type == 'neighbor':
        with open(json_path, 'r') as f:
            addresses = json.load(f)
        relatedTransactionPath = "Dataset/TornadoNeighborTransactions/"
        max_attempts=3
    elif type == 'tornado_cash':
        addresses = [
            '0x722122df12d4e14e13ac3b6895a86e84145b6967'
        ]
        relatedTransactionPath = "Dataset/TornadoRelatedTransactions/"
        max_attempts=10000
    
    data = DataSource()
    
    if not os.path.exists(relatedTransactionPath):
        os.makedirs(relatedTransactionPath)
    
    for directory in ['Normal', 'Internal', 'ERC20']:
        path = os.path.join(relatedTransactionPath, directory)
        if not os.path.exists(path):
            os.makedirs(path)
    
    for address in tqdm(addresses, desc="Processing addresses"):
        for tt in ['Normal/', 'Internal/', 'ERC20/']:
            file_path = f"{relatedTransactionPath}{tt}{address}.csv"
            if not (os.path.exists(file_path)):
                try:
                    data.getTotalDatafromScan(address, tt, f"{relatedTransactionPath}{tt}",max_attempts=max_attempts)
                    time.sleep(data.timeStep)
                except Exception as e:
                    print(f"Error processing {address} for {tt}: {e}")

def getNeighborAddressLabel():
    with open('Dataset/tornado_neighbor_addresses.json', 'r') as f:
        addresses = json.load(f)
    with open('Dataset/tornado_cash_neighbor_address_label.json', 'r') as f:
        tornado_neighbor_address_label = json.load(f)
    
    for address in tqdm(addresses):
        if address not in tornado_neighbor_address_label:
            label, title = getAddressLabelFromEthereumPage(address)
            tornado_neighbor_address_label[address] = {'label': label, 'title': title}
    with open('Dataset/tornado_cash_neighbor_address_label.json', 'w') as f:
        json.dump(tornado_neighbor_address_label, f)

if __name__ == "__main__":
    GetAddressRelatedTransactions(type='neighbor', json_path='Dataset/unique_addrs_train_val.json')