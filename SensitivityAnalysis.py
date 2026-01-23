import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch_geometric.data import Batch
import random
import json
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from Model import PairwiseRankingGNN
from Graph import build_address_graph, build_address_graph_by_tx_volume # Assuming this can be modified
import os
import csv
from datetime import datetime

# --- Configuration ---
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
SEED = 1029
NUM_RUNS = 3
NODE_FEATURE_DIM = 19
EDGE_FEATURE_DIM = 5
GROUND_TRUTH_FILES = [
    # 'Dataset/GroundTruth/heist_0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc.json',
    # 'Dataset/GroundTruth/heist_0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936.json',
    # 'Dataset/GroundTruth/heist_0x910cbd523d972eb0a6f4cae4618ad62622b39dbf.json',
    # 'Dataset/GroundTruth/heist_0xa160cdab225685da1d56aa342ad8841c3b53f291.json'
    'Dataset/AMLValidation/train_all_all.json'
]
LEARNING_RATE = 0.001
EPOCHS = 1000 # Reduced epochs for faster analysis, can be increased
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 60
GRAPH_CACHE_DIR = 'graph_cache_sensitivity'
EVAL_BATCH_SIZE = 64
K_FOLD = 10 # Reduced folds for faster analysis
MAX_TIME_DIFF_SECONDS = 90 * 24 * 60 * 60 # 90 days in seconds

# --- Sensitivity Analysis Specific Config ---
# The overlap ratio determines how much of the time window between deposit and withdrawal
# is used to build the withdrawal graph from the past.
# 0.0 = withdrawal graph starts at its own block time (no overlap)
# 1.0 = withdrawal graph starts at deposit block time (full overlap)
OVERLAP_RATIOS = [0.0, 0.2, 0.5, 0.8, 1.0] 
SUPPLEMENTARY_DATA_RATIOS = [0.0, 0.2, 0.5, 0.8, 1.0]
MAX_TXS_VALUES = [100, 1000, 5000, 10000, 20000]

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_pair_features(deposit_pair, candidate_pair):
    time_diff = abs(candidate_pair['withdraw_timeStamp'] - deposit_pair['deposit_timeStamp'])
    scaled_time_diff = np.log1p(time_diff)
    return [scaled_time_diff]

def load_mixbroker_data(data_type='D1', max_txs=None):
    """
    Loads MixBroker data (D1, D2, D3) and prepares it for link prediction.
    This version includes both positive and negative samples.
    Graphs for these pairs will be built without temporal constraints.
    """
    print(f"Loading MixBroker dataset for {data_type} (Positive and Negative Samples)...")
    node_feature_df = pd.read_csv('Baseline/MixBroker-main/Dataset/Graph/node_feature.csv')
    train_pos_edge_df = pd.read_csv('Baseline/MixBroker-main/Dataset/Graph/train_pos_edge_10fold.csv')
    train_neg_edge_df = pd.read_csv('Baseline/MixBroker-main/Dataset/Graph/train_neg_edge_10fold.csv')

    nodeid_to_addr = dict(zip(node_feature_df['nodeid'], node_feature_df['node']))

    # Load positive and negative pairs
    pos_pairs = [(nodeid_to_addr[r['nodeid1']], nodeid_to_addr[r['nodeid2']]) for _, r in train_pos_edge_df.iterrows()]
    neg_pairs = [(nodeid_to_addr[r['nodeid1']], nodeid_to_addr[r['nodeid2']]) for _, r in train_neg_edge_df.iterrows()]
    
    if data_type == 'D3':
        rule2_files = ['Dataset/tornado_raw_data/heuristic2Mixer_0.1ETH.csv', 'Dataset/tornado_raw_data/heuristic2Mixer_1ETH.csv', 'Dataset/tornado_raw_data/heuristic2Mixer_10ETH.csv', 'Dataset/tornado_raw_data/heuristic2Mixer_100ETH.csv']
        rule3_files = ['Dataset/tornado_raw_data/heuristic3Mixer_0.1ETH.csv', 'Dataset/tornado_raw_data/heuristic3Mixer_1ETH.csv', 'Dataset/tornado_raw_data/heuristic3Mixer_10ETH.csv', 'Dataset/tornado_raw_data/heuristic3Mixer_100ETH.csv']
        rule2_df = pd.concat([pd.read_csv(f) for f in rule2_files])
        rule3_df = pd.concat([pd.read_csv(f) for f in rule3_files])
        extra_pairs = set((r['sender'], r['receiver']) for _, r in rule2_df.iterrows())
        extra_pairs.update((r['sender'], r['receiver']) for _, r in rule3_df.iterrows())
        pos_pairs += list(extra_pairs - set(pos_pairs))

    # Create labels (1 for positive, 0 for negative)
    all_pairs = pos_pairs + neg_pairs
    all_labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
    
    if data_type == 'D2':
        print("Sampling 75% of D1 data to create D2 dataset...")
        set_seed(SEED)
        combined_data = list(zip(all_pairs, all_labels))
        random.shuffle(combined_data)
        sample_size = int(len(combined_data) * 0.75)
        sampled_data = combined_data[:sample_size]
        if sampled_data:
            all_pairs, all_labels = zip(*sampled_data)
        else:
            all_pairs, all_labels = [], []
        all_pairs = list(all_pairs)
        all_labels = list(all_labels)

    print(f"Loaded for {data_type} - Positive pairs: {sum(all_labels)}, Negative pairs: {len(all_labels) - sum(all_labels)}")
    
    # --- Graph Building (Full graphs, no temporal constraints) ---
    print("Building or loading full graphs from cache for MixBroker data...")
    os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
    unique_addrs = set(addr for pair in all_pairs for addr in pair)
    
    address_to_graph = {}
    for i, addr in enumerate(list(unique_addrs)):
        if i % 100 == 0:
            print(f"  Processing address {i+1}/{len(unique_addrs)}...")
        cache_path = os.path.join(GRAPH_CACHE_DIR, f"{addr}_full_{max_txs if max_txs else 'all'}.pt")
        if os.path.exists(cache_path):
            address_to_graph[addr] = torch.load(cache_path)
        else:
            graph, _ = build_address_graph(addr, max_txs=max_txs) # Build full graph with a potential tx cap
            if graph: torch.save(graph, cache_path)
            address_to_graph[addr] = graph
    
    packaged_data = []
    for (addr1, addr2), label in zip(all_pairs, all_labels):
        graph1 = address_to_graph.get(addr1)
        graph2 = address_to_graph.get(addr2)
        if graph1 and graph2:
            packaged_data.append({
                'graph1': graph1, 'graph2': graph2,
                'features': torch.tensor([0.0], dtype=torch.float32), # Placeholder
                'label': torch.tensor([float(label)], dtype=torch.float32),
                'addr1': addr1,
                'addr2': addr2
            })
            
    print(f"Total packaged MixBroker pairs: {len(packaged_data)}")
    return packaged_data

def load_supplementary_data_with_temporal_graphs(json_files, overlap_ratio, max_txs=None):
    """
    Loads supplementary data and builds graphs with specific time constraints.
    This function ONLY loads positive pairs from the ground truth files.
    """
    print(f"\n--- Loading supplementary data for Overlap Ratio: {overlap_ratio} ---")
    all_positive_pairs = []
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                pairs = json.load(f)
            if not pairs: continue
            pairs = [p for p in pairs if (p['withdraw_timeStamp'] - p['deposit_timeStamp']) <= MAX_TIME_DIFF_SECONDS]
            all_positive_pairs.extend(pairs)
        except Exception as e:
            print(f"  - Warning: Could not process {file_path}. Error: {e}")

    print(f"Loaded {len(all_positive_pairs)} total positive pairs from supplementary files.")

    # --- Data Preparation (Positive Pairs Only) ---
    all_pairs_with_labels = []
    for pos_pair in all_positive_pairs:
        all_pairs_with_labels.append({'pair': pos_pair, 'label': 1})
    
    # --- Temporal Graph Building ---
    print("Building temporally constrained graphs for supplementary data...")
    
    packaged_data = []
    for i, item in enumerate(all_pairs_with_labels):
        if i % 100 == 0:
            print(f"  Processing pair {i+1}/{len(all_pairs_with_labels)}...")
        
        pair = item['pair']
        label = item['label']
        
        dep_addr, dep_block = pair['deposit_address'], pair['deposit_blockNumber']
        wit_addr, wit_block = pair['withdraw_address'], pair['withdraw_blockNumber']

        # --- Define time windows based on the TX VOLUME logic ---
        
        graph1, graph2 = None, None
        
        if overlap_ratio == 1.0:
            # Full info: Use the entire history for both, with a potential tx cap
            graph1, _ = build_address_graph(dep_addr, max_txs=max_txs)
            graph2, _ = build_address_graph(wit_addr, max_txs=max_txs)
        else:
            # For other ratios, expand graph based on transaction volume, with a potential tx cap
            # Deposit graph expands forward in time
            graph1, _ = build_address_graph_by_tx_volume(
                address=dep_addr,
                reference_timestamp=pair['deposit_timeStamp'],
                mode='deposit',
                ratio=overlap_ratio,
                max_txs=max_txs
            )
            # Withdrawal graph expands backward in time
            graph2, _ = build_address_graph_by_tx_volume(
                address=wit_addr,
                reference_timestamp=pair['withdraw_timeStamp'],
                mode='withdrawal',
                ratio=overlap_ratio,
                max_txs=max_txs
            )

        if graph1 and graph2:
            pair_features = torch.tensor(create_pair_features(pair, pair), dtype=torch.float32)
            packaged_data.append({
                'graph1': graph1, 'graph2': graph2,
                'features': pair_features,
                'label': torch.tensor([float(label)], dtype=torch.float32),
                'addr1': dep_addr,
                'addr2': wit_addr
            })
            
    print(f"Total packaged supplementary data pairs for ratio {overlap_ratio}: {len(packaged_data)}")
    return packaged_data

# --- Training & Evaluation (similar to Train.py) ---

def train_classification_model(model, train_data, optimizer, loss_func, device):
    model.train()
    total_loss = 0
    random.shuffle(train_data)
    batches_processed = 0
    for i in range(0, len(train_data), BATCH_SIZE):
        batch = train_data[i:i+BATCH_SIZE]
        if len(batch) <= 1: continue
        batches_processed += 1
        
        graphs1 = Batch.from_data_list([p['graph1'] for p in batch]).to(device)
        graphs2 = Batch.from_data_list([p['graph2'] for p in batch]).to(device)
        features = torch.stack([p['features'] for p in batch]).to(device)
        labels = torch.stack([p['label'] for p in batch]).to(device).squeeze(-1)

        optimizer.zero_grad()
        scores = model(graphs1, graphs2, features)
        loss = loss_func(scores, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / batches_processed if batches_processed > 0 else 0.0

def evaluate_classification_model(model, val_data, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(val_data), EVAL_BATCH_SIZE):
            batch = val_data[i:i+EVAL_BATCH_SIZE]
            graphs1 = Batch.from_data_list([p['graph1'] for p in batch]).to(device)
            graphs2 = Batch.from_data_list([p['graph2'] for p in batch]).to(device)
            features = torch.stack([p['features'] for p in batch]).to(device)
            labels = torch.stack([p['label'] for p in batch]).squeeze(-1)
            
            logits = model(graphs1, graphs2, features)
            preds = (torch.sigmoid(logits) > 0.5).cpu()
            all_preds.append(preds)
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Calculate FPR and FNR from confusion matrix
    if len(np.unique(all_labels)) == 2: # Ensure both classes are present
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else: # Handle case where only one class is in the validation set
        fpr, fnr = 0.0, 0.0

    return precision, recall, f1, fpr, fnr

def save_results_to_csv(results, params, filename):
    file_exists = os.path.isfile(filename)
    
    # Format metrics for consistency
    formatted_results = {
        'F1': f"{results['avg_f1']:.4f}±{results['std_f1']:.4f}",
        'Precision': f"{results['avg_precision']:.4f}±{results['std_precision']:.4f}",
        'Recall': f"{results['avg_recall']:.4f}±{results['std_recall']:.4f}",
        'FPR': f"{results['avg_fpr']:.4f}±{results['std_fpr']:.4f}",
        'FNR': f"{results['avg_fnr']:.4f}±{results['std_fnr']:.4f}",
        'Best_Run': results['run']
    }

    row_data = {**params, **formatted_results}
    
    fieldnames = ['Dataset', 'Supplementary_Data_Ratio', 'Overlap_Ratio', 'Max_Txs', 'GNN_Type', 'Best_Run', 'F1', 'Precision', 'Recall', 'FPR', 'FNR']
    
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
    print(f"Results saved to {filename}")

def run_analysis(config, device):
    """
    Runs a full experiment for a given configuration, including multiple runs.
    """
    max_txs = config.get('max_txs') # Can be None for default
    # 1. Load base data (D1, D2, or D3) - this is done once per config
    base_data = load_mixbroker_data(config['dataset'], max_txs=max_txs)

    # 2. Load supplementary data with temporally constrained graphs
    supplementary_data = load_supplementary_data_with_temporal_graphs(
        GROUND_TRUTH_FILES, 
        config['overlap_ratio'],
        max_txs=max_txs
    )
    
    # 3. Sample a portion of the supplementary data
    sup_data_ratio = config['supplementary_data_ratio']
    if sup_data_ratio < 1.0:
        if sup_data_ratio == 0.0:
            sampled_supplementary_data = []
        else:
            # Shuffle before sampling to get a random subset
            random.shuffle(supplementary_data)
            sample_size = int(len(supplementary_data) * sup_data_ratio)
            sampled_supplementary_data = supplementary_data[:sample_size]
    else:
        sampled_supplementary_data = supplementary_data
    
    print(f"Using {len(sampled_supplementary_data)} supplementary pairs ({sup_data_ratio*100}%)")

    # 4. Combine datasets
    all_data = base_data + sampled_supplementary_data
    if not all_data:
        print(f"No data generated for this configuration. Skipping.")
        return None

    all_data_np = np.array(all_data)
    
    best_run_avg_f1 = 0.0
    best_run_results = {}

    for run in range(1, NUM_RUNS + 1):
        print(f"\n{'='*20} Starting Run {run}/{NUM_RUNS} {'='*20}")
        # Use a different seed for each run for variability
        set_seed(SEED)

        # 5. K-fold cross-validation
        kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=SEED + run)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(all_data_np)):
            print(f"\n--- Starting Fold {fold+1}/{K_FOLD} ---")
            train_pairs = all_data_np[train_idx].tolist()
            val_pairs = all_data_np[val_idx].tolist()

            # --- Enforce Address-Disjoint Split (Sensitivity Analysis) ---
            # 1. Identify all addresses in the validation set
            val_addrs = set()
            for item in val_pairs:
                val_addrs.add(item['addr1']) # Note: Need to ensure addr1/addr2 are in packaged items
                val_addrs.add(item['addr2']) # load_mixbroker_data and load_supplementary... need to return these keys
            
            # 2. Filter training set: Remove pairs that involve any validation address
            original_train_len = len(train_pairs)
            train_pairs = [
                item for item in train_pairs 
                if item['addr1'] not in val_addrs and item['addr2'] not in val_addrs
            ]
            print(f"  - Address-Disjoint Filter: Removed {original_train_len - len(train_pairs)} pairs from training set.")

            model = PairwiseRankingGNN(
                node_feature_dim=NODE_FEATURE_DIM, edge_feature_dim=EDGE_FEATURE_DIM,
                pair_feature_dim=1, use_graph=True, gnn_type=config['gnn_type']
            ).to(device)
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
            loss_func = torch.nn.BCEWithLogitsLoss()

            best_f1 = 0.0
            epochs_no_improve = 0
            best_fold_metrics = (0,0,0,0,0) # P, R, F1, FPR, FNR

            for epoch in range(1, EPOCHS + 1):
                avg_loss = train_classification_model(model, train_pairs, optimizer, loss_func, device)
                precision, recall, f1, fpr, fnr = evaluate_classification_model(model, val_pairs, device)

                if epoch % 100 == 0:
                    print(f"  Epoch {epoch:03d} | Loss: {avg_loss:.4f} | F1: {f1:.2%}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_fold_metrics = (precision, recall, f1, fpr, fnr)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                        print(f"  Early stopping triggered at epoch {epoch}.")
                        break
            
            print(f"Fold {fold+1} Best F1: {best_fold_metrics[2]:.4f}")
            fold_results.append(best_fold_metrics)

        # --- This Run's Final Results ---
        avg_precision = np.mean([res[0] for res in fold_results])
        avg_recall = np.mean([res[1] for res in fold_results])
        avg_f1 = np.mean([res[2] for res in fold_results])
        avg_fpr = np.mean([res[3] for res in fold_results])
        avg_fnr = np.mean([res[4] for res in fold_results])
        
        std_precision = np.std([res[0] for res in fold_results])
        std_recall = np.std([res[1] for res in fold_results])
        std_f1 = np.std([res[2] for res in fold_results])
        std_fpr = np.std([res[3] for res in fold_results])
        std_fnr = np.std([res[4] for res in fold_results])

        print(f"\n--- Run {run}/{NUM_RUNS} Cross-Validation Summary ---")
        print(f"Average F1-score: {avg_f1:.2%} ± {std_f1:.2%}")
        
        if avg_f1 > best_run_avg_f1:
            best_run_avg_f1 = avg_f1
            best_run_results = {
                'run': run,
                'avg_precision': avg_precision, 'std_precision': std_precision,
                'avg_recall': avg_recall, 'std_recall': std_recall,
                'avg_f1': avg_f1, 'std_f1': std_f1,
                'avg_fpr': avg_fpr, 'std_fpr': std_fpr,
                'avg_fnr': avg_fnr, 'std_fnr': std_fnr,
            }
    
    return best_run_results

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GNN Sensitivity Analysis on Graph Time Windows')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID.')
    parser.add_argument('--dataset', type=str, default='D1', choices=['D1', 'D2', 'D3'],
                        help='Choose the base dataset to use: D1, D2, or D3')
    args = parser.parse_args()

    DEVICE = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    GNN_TYPE = 'GATv2Conv'
    print(f"Using GNN model: {GNN_TYPE}")

    # --- Experiment Group 1: Analyze Supplementary Data Ratio ---
    # Keep Overlap Ratio fixed at its default (1.0)
    print("\n" + "="*50)
    print("Running Experiment Group 1: Analyzing Supplementary Data Ratio")
    print("Overlap Ratio is fixed at 1.0")
    print("="*50)
    for sup_ratio in SUPPLEMENTARY_DATA_RATIOS:
        print(f"\n--- Testing Supplementary Ratio = {sup_ratio} ---")
        config = {
            'dataset': args.dataset,
            'supplementary_data_ratio': sup_ratio,
            'overlap_ratio': 1.0,  # Fixed default
            'max_txs': 20000, # Fixed default
            'gnn_type': GNN_TYPE
        }
        best_run_results = run_analysis(config, DEVICE)
        if best_run_results:
            params = {
                'Dataset': args.dataset,
                'Supplementary_Data_Ratio': sup_ratio,
                'Overlap_Ratio': 1.0,
                'Max_Txs': 20000,
                'GNN_Type': GNN_TYPE,
            }
            save_results_to_csv(best_run_results, params, f'Results/{RUN_TIMESTAMP}_sensitivity_analysis_{args.dataset}.csv')

    # --- Experiment Group 2: Analyze Overlap Ratio ---
    # Keep Supplementary Data Ratio and Max Txs fixed at defaults
    print("\n" + "="*50)
    print("Running Experiment Group 2: Analyzing Overlap Ratio")
    print("Supplementary Data Ratio is fixed at 1.0, Max Txs at 20000")
    print("="*50)
    for ratio in OVERLAP_RATIOS:
        # The case where ratio is 1.0 was run in the first group, so we skip it here.
        if ratio == 1.0:
            continue
        print(f"\n--- Testing Overlap Ratio = {ratio} ---")
        config = {
            'dataset': args.dataset,
            'supplementary_data_ratio': 1.0, # Fixed default
            'overlap_ratio': ratio,
            'max_txs': 20000, # Fixed default
            'gnn_type': GNN_TYPE
        }
        best_run_results = run_analysis(config, DEVICE)
        if best_run_results:
            params = {
                'Dataset': args.dataset,
                'Supplementary_Data_Ratio': 1.0,
                'Overlap_Ratio': ratio,
                'Max_Txs': 20000,
                'GNN_Type': GNN_TYPE,
            }
            save_results_to_csv(best_run_results, params, f'Results/{RUN_TIMESTAMP}_sensitivity_analysis_{args.dataset}.csv')

    # --- Experiment Group 3: Analyze Max Transactions ---
    # Keep Supplementary Data Ratio and Overlap Ratio fixed at defaults
    print("\n" + "="*50)
    print("Running Experiment Group 3: Analyzing Max Transactions")
    print("Supplementary Data Ratio is fixed at 1.0, Overlap Ratio at 1.0")
    print("="*50)
    for tx_limit in MAX_TXS_VALUES:
        # The case where tx_limit is 20000 was run in the first group, so we skip it here.
        if tx_limit == 20000:
            continue
        print(f"\n--- Testing Max Txs = {tx_limit} ---")
        config = {
            'dataset': args.dataset,
            'supplementary_data_ratio': 1.0, # Fixed default
            'overlap_ratio': 1.0, # Fixed default
            'max_txs': tx_limit,
            'gnn_type': GNN_TYPE
        }
        best_run_results = run_analysis(config, DEVICE)
        if best_run_results:
            params = {
                'Dataset': args.dataset,
                'Supplementary_Data_Ratio': 1.0,
                'Overlap_Ratio': 1.0,
                'Max_Txs': tx_limit,
                'GNN_Type': GNN_TYPE,
            }
            save_results_to_csv(best_run_results, params, f'Results/{RUN_TIMESTAMP}_sensitivity_analysis_{args.dataset}.csv')

    print("\n✅ Sensitivity analysis complete.")
