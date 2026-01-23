# MixTracker

Code for the paper "MixTracker: An Effective Graph-Based Framework for Address Linking in Ethereum Mixing Services"

## Status

*   ✅ The complete codebase has been released, including the full pipeline for data preparation, model training, and evaluation.
*   Intermediate files, such as the constructed graph data, will be uploaded after the paper is accepted.

## Configuration

Before running the scripts, you need to add your configuration details to `Utils.py`. This includes:

*   **Etherscan API Keys**: Fill in your Etherscan API keys in the `apikeys` list.
*   **User-Agent**: Provide a browser user-agent string.
*   **RPC Endpoint**: Set your Ethereum RPC endpoint URL in `url_rpc`.

## Usage

### 1. Dataset

Download the dataset from [here](https://1drv.ms/f/c/cdc0f83bf736d892/EqB6iL16fZBKhQZiF1tthuEBjJ4UrNdAZFUBpIYD2-8AZQ?e=x2Q5yS) and extract it to the `Dataset/` directory.

The dataset contains the following files and folders:
*   **AMLValidation/**: Supplementary training data and case study data (e.g., `train_all_all.json`, `val_heist_all.json`)
*   **mixbroker_raw_data/**: Raw transaction data for mixbroker analysis
*   **tornado_raw_data/**: Raw transaction data for mixlinker analysis
*   **TornadoContractTransaction.tar.gz**: Compressed transaction data (needs to be extracted)
*   **TornadoNeighborTransactions.tar.gz**: Compressed neighbor transaction data (needs to be extracted)
*   **TornadoRelatedTransactions.tar.gz**: Compressed related transaction data (needs to be extracted)

**Note**: Extract the `.tar.gz` files to the `Dataset/` directory before running the scripts.

Then, execute the `Dataset.py` script to download additional relevant transactions (if needed):
```bash
python Dataset.py
```

### 2. Graph Construction

Execute the `Graph.py` script to build the transaction graph:
```bash
python Graph.py
```

### 3. Model Training and Evaluation

Execute the `Train.py` script to train the model and evaluate its performance:
```bash
python Train.py --dataset D1 --gpu 0
```

The training script includes:
*   K-fold cross-validation (default: 10-fold)
*   **Address-disjoint splits**: The code strictly enforces address-disjoint splits between training and validation sets to prevent data leakage, as claimed in the paper. This ensures that no address appearing in the validation set appears in the training set, which is critical for realistic evaluation.
*   Comprehensive evaluation metrics: Accuracy, Precision, Recall, F1-score, FPR (False Positive Rate), and FNR (False Negative Rate)
*   Results are automatically saved to CSV files in the `Results/` directory

**Arguments:**
*   `--dataset`: Choose the dataset (D1, D2, or D3)
*   `--gpu`: Specify GPU device ID (default: 0)