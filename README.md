# MixTracker

Code for the paper "MixTracker: An Effective Graph-Based Framework for Address Linking in Ethereum Mixing Services"

## Status

*   🚧 The model part of the code has been uploaded, and the verification part of the code will be uploaded soon.
*   Intermediate files, such as the constructed graph data, will be uploaded after the paper is accepted.

## Configuration

Before running the scripts, you need to add your configuration details to `Utils.py`. This includes:

*   **Etherscan API Keys**: Fill in your Etherscan API keys in the `apikeys` list.
*   **User-Agent**: Provide a browser user-agent string.
*   **RPC Endpoint**: Set your Ethereum RPC endpoint URL in `url_rpc`.

## Usage

### 1. Dataset

Download the dataset from [here](https://1drv.ms/f/c/cdc0f83bf736d892/EqB6iL16fZBKhQZiF1tthuEBjJ4UrNdAZFUBpIYD2-8AZQ?e=x2Q5yS) and extract it to the `Dataset/` directory.

Then, execute the `Dataset.py` script to download the relevant transactions:
```bash
python Dataset.py
```

### 2. Graph Construction

Execute the `Graph.py` script to build the transaction graph:
```bash
python Graph.py
```

### 3. Model Training

Execute the `Train.py` script to train the model:
```bash
python Train.py
```