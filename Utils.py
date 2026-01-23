import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import random
import json
import pandas as pd

class DataSource:

    def __init__(self):
        self.apikeys = [
            "",   
            "",
            "",
            ""
        ]
        self.timeStep = 0.01
        self.headers = {
            "content-type": "application/json",
            "user-agent": "",
        }
        self.url_rpc = ""

    def _get_etherscan_data(self, module, action, address, startblock, endblock, page=1, offset=10000, sort="asc", **kwargs):
        params = {
            "chainid": 1,
            "module": module,
            "action": action,
            "address": address,
            "startblock": startblock,
            "endblock": endblock,
            "page": page,
            "offset": offset,
            "sort": sort,
            "apikey": random.choice(self.apikeys)
        }
        
        params.update(kwargs)
        url_params = "&".join([f"{k}={v}" for k, v in params.items()])
        url = f"https://api.etherscan.io/v2/api?{url_params}"
        
        return getDataFromUrl(url, self.headers).json()["result"]

    def getNormalTransactionsbyAddress(self, address, startblock, endblock, page, offset=10000, sort="asc"):
        return self._get_etherscan_data("account", "txlist", address, startblock, endblock, page, offset, sort)

    def getInternalTransactionsbyAddress(self, address, startblock, endblock, page=1, offset=10000, sort="asc"):
        return self._get_etherscan_data("account", "txlistinternal", address, startblock, endblock, page, offset, sort)
    
    def getInternalTransactionsbyTransactionHash(self, txhash):
        params = {
            "chainid": 1,
            "module": "account",
            "action": "txlistinternal",
            "txhash": txhash,
            "apikey": random.choice(self.apikeys)
        }
        url_params = "&".join([f"{k}={v}" for k, v in params.items()])
        url = f"https://api.etherscan.io/v2/api?{url_params}"
        
        return getDataFromUrl(url, self.headers).json()["result"]

    def getERCTokenTransferbyAddress(self, action, address, startblock, endblock, page, offset=10000, contractaddress="", sort="asc"):
        kwargs = {}
        if contractaddress:
            kwargs["contractaddress"] = contractaddress
            
        return self._get_etherscan_data("account", action, address, startblock, endblock, page, offset, sort, **kwargs)

    def getTotalDatafromScan(self, address, ttype, saved_path, max_attempts = 2, start_number=0, end_number=99999999):
        saved_path_address = f"{saved_path}{address}.csv"
        response_list = []
        
        type_method_map = {
            'Normal/': lambda: self.getNormalTransactionsbyAddress(address, start_number, end_number, 1),
            'Internal/': lambda: self.getInternalTransactionsbyAddress(address, start_number, end_number, 1),
            'ERC20/': lambda: self.getERCTokenTransferbyAddress('tokentx', address, start_number, end_number, 1)
        }
        
        attempt_count = 0
        
        while attempt_count < max_attempts:
            if ttype not in type_method_map:
                return False, 0
                
            response = type_method_map[ttype]()
            
            if not response:
                return False, 0
                
            response_list.extend(response)
            
            if len(response) < 10000:
                if response_list:
                    pd.DataFrame(response_list).to_csv(saved_path_address, index=None)
                return True, len(response_list)
                
            start_number = int(response[-1]["blockNumber"])
            attempt_count += 1
            
        if response_list:
            pd.DataFrame(response_list).to_csv(saved_path_address, index=None)
        return True, len(response_list)

    def getTransactionCountfromRPC(self, address):
        payload = {
            "method": "eth_getTransactionCount",
            "params": [address, "latest"],
            "id": 1,
            "jsonrpc": "2.0",
        }
        return int(getDatafromRPC(payload)["result"], 16)
        
    def getBalancefromRPC(self, address):
        payload = {
            "method": "eth_getBalance",
            "params": [address, "latest"],
            "id": 1,
            "jsonrpc": "2.0",
        }
        return int(getDatafromRPC(payload)["result"], 16) / 10**18

def getDataFromUrl(url, headers, data=None, sstype='get', timeout=20):
    retries = Retry(total=10, backoff_factor=0.9)
    
    with requests.Session() as session:
        session.mount("http://", HTTPAdapter(max_retries=retries))
        session.mount("https://", HTTPAdapter(max_retries=retries))
        
        try:
            if sstype == 'get':
                response = session.get(url, headers=headers, data=data, timeout=timeout)
            else:
                response = session.post(url, headers=headers, data=data, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException:
            return None
            
def getAddressLabelFromEthereumPage(address):
    headers = {
        'cookie': '',
        'user-agent': '',
        'origin': 'https://etherscan.io'
    }
    url_address = f'https://etherscan.io/address/{address}'
    
    response = getDataFromUrl(url=url_address, headers=headers)
    if not response:
        return "", ""
        
    soup = BeautifulSoup(response.text, 'html.parser')
    
    target_section = soup.select_one('body main section:nth-of-type(3) div:nth-of-type(1) div:nth-of-type(1)')
    
    span_text = ""
    
    if target_section:
        all_spans = target_section.find_all('span')
        all_span_contents = [span.get_text().strip() for span in all_spans if span.get_text().strip()]
        
        if not span_text and all_span_contents:
            unique_contents = list(dict.fromkeys(all_span_contents))
            span_text = ";".join(unique_contents)
    
    title_text = ""
    if soup.title:
        title_parts = soup.title.string.split("|")
        if len(title_parts) >= 3:
            title_text = title_parts[0].strip().split("\n")[0]
    
    return span_text, title_text

def getDatafromRPC(payload):
    url_rpc = ""
    headers = {
        "content-type": "application/json",
        "user-agent": "",
    }
    response = getDataFromUrl(url_rpc, headers=headers, data=json.dumps(payload), sstype='post')
    return response.json() if response else None

def getTransactionReceipt(hash):
    payload = {
        "method": "eth_getTransactionReceipt",
        "params": [
            hash
        ],
        "id": 1,
        "jsonrpc": "2.0"
    }
    return getDatafromRPC(payload)

def getTransactionByHash(hash):
    payload = {
        "method": "eth_getTransactionByHash",
        "params": [
            hash
        ],
        "id": 1,
        "jsonrpc": "2.0"
    }
    return getDatafromRPC(payload)

def isContract(address):
    payload = {
        "method": "eth_getCode",
        "params": [
        address,
        "latest"
        ],
        "id": 1,
        "jsonrpc": "2.0"
    }
    response = getDatafromRPC(payload)
    if response is None:
        return False
    if 'result' not in response:
        return False
    if response['result'] is not None:
        if response['result'] != '0x':
            return True
        else:
            return False

if __name__ == "__main__":
    address = '0xdAC17F958D2ee523a2206206994597C13D831ec7'
    print(getAddressLabelFromEthereumPage(address))