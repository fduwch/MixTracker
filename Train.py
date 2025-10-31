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
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from Model import PairwiseRankingGNN # Updated import
from Graph import build_address_graph
import os
import csv
from datetime import datetime

# Generate timestamp once at program start
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# --- Configuration ---
SEED = 1029
NUM_RUNS = 4
NODE_FEATURE_DIM = 19
EDGE_FEATURE_DIM = 5 # value, gasUsed, gasPrice, timeStamp, tx_type
GROUND_TRUTH_FILES = [
    'Dataset/GroundTruth/heist_0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc.json',
    'Dataset/GroundTruth/heist_0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936.json',
    'Dataset/GroundTruth/heist_0x910cbd523d972eb0a6f4cae4618ad62622b39dbf.json',
    'Dataset/GroundTruth/heist_0xa160cdab225685da1d56aa342ad8841c3b53f291.json'
]
LEARNING_RATE = 0.001
EPOCHS = 1000
BATCH_SIZE = 32 # BPR Loss uses triplets, so effective batch size is smaller
MODEL_SAVE_PATH = 'best_ranking_model.pth' # New model name
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Will be set based on args
EARLY_STOPPING_PATIENCE = 50
MAX_TIME_DIFF_SECONDS = 90 * 24 * 60 * 60 # 90 days in seconds
GRAPH_CACHE_DIR = 'graph_cache'
EVAL_BATCH_SIZE = 64 # Keep this manageable for evaluation
K_FOLD = 10
POS_WEIGHT = 0.5 # Add weight to BCE loss to balance precision/recall

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Loss Function ---
# We are changing from BPR Loss to Binary Cross-Entropy for link prediction
loss_func = torch.nn.BCEWithLogitsLoss()

# --- Data Loading for Training ---
def create_pair_features(deposit_pair, candidate_pair):
    """Calculates pairwise features between a deposit and a candidate."""
    time_diff = abs(candidate_pair['withdraw_timeStamp'] - deposit_pair['deposit_timeStamp'])
    # Apply log transform to scale the feature and prevent numerical instability
    scaled_time_diff = np.log1p(time_diff)
    # More features can be added here, e.g., token type differences
    return [scaled_time_diff]

def load_and_prep_data_for_ranking(json_files):
    """
    Loads data from JSON files and prepares it for binary classification link prediction,
    matching the output format of load_mixbroker_data.
    """
    print("Loading and preparing ground truth data for link prediction...")
    all_positive_pairs = []

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                pairs = json.load(f)
            if not pairs: continue
            
            # Filter pairs based on time difference
            pairs = [p for p in pairs if (p['withdraw_timeStamp'] - p['deposit_timeStamp']) <= MAX_TIME_DIFF_SECONDS]
            all_positive_pairs.extend(pairs)
        except Exception as e:
            print(f"  - Warning: Could not process {file_path}. Error: {e}")
            continue

    print(f"Loaded {len(all_positive_pairs)} total positive pairs.")

    # --- Negative Sampling ---
    all_pairs_with_labels = []
    withdrawal_pool = [p for p in all_positive_pairs] # Pool of all possible withdrawals
    time_window_seconds = 24 * 60 * 60 # 24 hours for hard negative sampling

    for pos_pair in all_positive_pairs:
        # Add the positive pair with label 1
        all_pairs_with_labels.append({'pair': pos_pair, 'label': 1})
        
        # --- Hard Negative Sampling ---
        deposit_time = pos_pair['deposit_timeStamp']
        
        # Find candidates within a time window relative to the true deposit time
        hard_negative_candidates = [
            p for p in withdrawal_pool 
            if abs(p['withdraw_timeStamp'] - deposit_time) < time_window_seconds
            and p['withdraw_address'] != pos_pair['withdraw_address']
        ]
        
        neg_candidate = None
        if hard_negative_candidates:
            # If we found hard negatives, pick one
            neg_candidate = random.choice(hard_negative_candidates)
        else:
            # Fallback to random sampling if no candidates are found in the window
            # Ensure the negative candidate is not the true withdrawal for the same deposit
            while neg_candidate is None or neg_candidate['withdraw_address'] == pos_pair['withdraw_address']:
                neg_candidate = random.choice(withdrawal_pool)
        
        # The negative pair shares the same deposit but has a different withdrawal
        neg_pair = pos_pair.copy()
        neg_pair['withdraw_address'] = neg_candidate['withdraw_address']
        neg_pair['withdraw_blockNumber'] = neg_candidate['withdraw_blockNumber']
        neg_pair['withdraw_timeStamp'] = neg_candidate['withdraw_timeStamp']

        # all_pairs_with_labels.append({'pair': neg_pair, 'label': 0})

    # --- Graph Building ---
    print("Building or loading graphs from cache...")
    os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
    
    unique_addrs = set()
    for item in all_pairs_with_labels:
        p = item['pair']
        unique_addrs.add(p['deposit_address'])
        unique_addrs.add(p['withdraw_address'])

    address_to_graph = {}
    for i, addr in enumerate(list(unique_addrs)):
        # For this dataset, we build the full graph for each address
        cache_path = os.path.join(GRAPH_CACHE_DIR, f"{addr}_full.pt")
        if os.path.exists(cache_path):
            address_to_graph[addr] = torch.load(cache_path)
        else:
            graph, _ = build_address_graph(addr)
            if graph:
                torch.save(graph, cache_path)
            address_to_graph[addr] = graph

    # --- Package Data ---
    packaged_data = []
    for item in all_pairs_with_labels:
        pair = item['pair']
        label = item['label']
        
        addr1 = pair['deposit_address']
        addr2 = pair['withdraw_address']
        graph1 = address_to_graph.get(addr1)
        graph2 = address_to_graph.get(addr2)

        if graph1 and graph2:
            # Unlike MixBroker, here we have real features
            pair_features = torch.tensor(create_pair_features(pair, pair), dtype=torch.float32)
            
            packaged_data.append({
                'graph1': graph1,
                'graph2': graph2,
                'features': pair_features,
                'label': torch.tensor([float(label)], dtype=torch.float32)
            })
            
    print(f"Total packaged data pairs: {len(packaged_data)}")
    return packaged_data


def load_mixbroker_data(data_type='D1'):
    """
    Loads MixBroker data and prepares it for binary classification link prediction.
    """
    print("Loading MixBroker dataset for link prediction...")
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

        # 去重后的添加
        extra_pairs = set((r['sender'], r['receiver']) for _, r in rule2_df.iterrows())
        extra_pairs.update((r['sender'], r['receiver']) for _, r in rule3_df.iterrows())
        pos_pairs += list(extra_pairs - set(pos_pairs))

    print(f"Loaded for {data_type} - Positive pairs: {len(pos_pairs)}, Negative pairs: {len(neg_pairs)}")

    # Create labels (1 for positive, 0 for negative)
    all_pairs = pos_pairs + neg_pairs
    all_labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
    
    if data_type == 'D2':
        print("Sampling 75% of D1 data to create D2 dataset...")
        set_seed(SEED) # Ensure sampling is reproducible
        
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

        num_pos = sum(all_labels)
        num_neg = len(all_labels) - num_pos
        print(f"D2 - Sampled to: {len(all_pairs)} pairs (Pos: {num_pos}, Neg: {num_neg})")

    
    print("Building or loading graphs from cache for MixBroker data...")
    os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
    
    unique_addrs = set(addr for pair in all_pairs for addr in pair)
    
    # with open('Dataset/unique_addrs.json', 'w') as f:
    #     f.write(json.dumps(list(unique_addrs)))

    address_to_graph = {}
    for i, addr in enumerate(list(unique_addrs)):
        if i % 100 == 0:
            print(f"  Processing address {i+1}/{len(unique_addrs)}...")
        
        cache_path = os.path.join(GRAPH_CACHE_DIR, f"{addr}_full.pt")
        if os.path.exists(cache_path):
            address_to_graph[addr] = torch.load(cache_path)
        else:
            graph, _ = build_address_graph(addr)
            if graph:
                torch.save(graph, cache_path)
            address_to_graph[addr] = graph
    
    packaged_data = []
    for (addr1, addr2), label in zip(all_pairs, all_labels):
        graph1 = address_to_graph.get(addr1)
        graph2 = address_to_graph.get(addr2)
        if graph1 and graph2:
            packaged_data.append({
                'graph1': graph1,
                'graph2': graph2,
                'features': torch.tensor([0.0], dtype=torch.float32), # Placeholder
                'label': torch.tensor([float(label)], dtype=torch.float32)
            })
            
    print(f"Total packaged data pairs: {len(packaged_data)}")
    return packaged_data


# --- Training Loop ---
def train_classification_model(model, train_data, optimizer, loss_func, device):
    model.train()
    total_loss = 0
    random.shuffle(train_data)
    batches_processed = 0

    for i in range(0, len(train_data), BATCH_SIZE):
        batch = train_data[i:i+BATCH_SIZE]
        
        # Skip batches with size <= 1, as they cause issues with BatchNorm during training
        if len(batch) <= 1:
            continue
        
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

    if batches_processed > 0:
        return total_loss / batches_processed
    return 0.0

# --- Evaluation Loop ---
def evaluate_classification_model(model, val_data, device):
    model.eval()
    all_preds = []
    all_labels = []

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
            all_labels.append(labels.cpu()) # Ensure labels are on CPU for concatenation
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Calculate FPR and FNR from confusion matrix
    if len(np.unique(all_labels)) == 2: # Ensure we have both classes present
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else: # Handle case where only one class is in the validation set for a small fold
        fpr, fnr = 0.0, 0.0

    return accuracy, precision, recall, f1, fpr, fnr


def save_results_to_csv(results, params, filename='training_results.csv'):
    """Saves training results and parameters to a CSV file."""
    file_exists = os.path.isfile(filename)

    # Format metrics as "mean±std" with 4 decimal places
    formatted_results = {
        'Accuracy': f"{results['avg_accuracy']:.4f}±{results['std_accuracy']:.4f}",
        'Precision': f"{results['avg_precision']:.4f}±{results['std_precision']:.4f}",
        'Recall': f"{results['avg_recall']:.4f}±{results['std_recall']:.4f}",
        'F1': f"{results['avg_f1']:.4f}±{results['std_f1']:.4f}",
        'FPR': f"{results['avg_fpr']:.4f}±{results['std_fpr']:.4f}",
        'FNR': f"{results['avg_fnr']:.4f}±{results['std_fnr']:.4f}",
        'Best_Run': results['run']
    }

    row_data = {**params, **formatted_results}
    
    # Keep a consistent order for columns
    param_keys = ['Dataset', 'Use_Extra_Data', 'Learning_Rate', 'Epochs', 'Batch_Size', 'K_Fold', 'Seed', 'Patience', 'Num_Runs', 'Pos_Weight',
                  'Use_Graph', 'GNN_Type']
    result_keys = ['Best_Run', 'F1', 'Accuracy', 'Precision', 'Recall', 'FPR', 'FNR']
    fieldnames = param_keys + result_keys

    try:
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
        print(f"Results saved to {filename}")
    except IOError as e:
        print(f"Error saving results to {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving to CSV: {e}")


def run_experiment(config, all_data, extra_data, device):
    """Runs a single experiment configuration."""
    print(f"\n{'='*25}")
    print(f"Starting Experiment with Config: {config}")
    print(f"{'='*25}")

    use_extra_data = config['use_extra_data']
    

    best_run_avg_f1 = 0.0
    best_run_results = {}

    for run in range(1, NUM_RUNS + 1):
        print(f"\n{'='*20} Starting Run {run}/{NUM_RUNS} {'='*20}")
        # Use a different seed for each run for variability
        set_seed(SEED)
        
        # 1. Set up K-fold cross-validation
        kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=SEED)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(all_data)):
            print(f"\n--- Starting Fold {fold+1}/{K_FOLD} ---")
            
            train_pairs = all_data[train_idx].tolist()
            val_pairs = all_data[val_idx].tolist()

            # Add the extra data to the training set if specified
            if use_extra_data:
                train_pairs.extend(extra_data)

            print(f"  - Total training pairs for this fold: {len(train_pairs)}")
            print(f"  - Total validation pairs for this fold: {len(val_pairs)}")

            # 2. Initialize model and optimizer for each fold
            PAIR_FEATURE_DIM = 1
            model = PairwiseRankingGNN(
                node_feature_dim=NODE_FEATURE_DIM,
                edge_feature_dim=EDGE_FEATURE_DIM,
                pair_feature_dim=PAIR_FEATURE_DIM,
                use_graph=config['use_graph'],
                gnn_type=config['gnn_type']
            ).to(device)
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

            # Initialize weighted loss function for this fold
            # loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=device))
            loss_func = torch.nn.BCEWithLogitsLoss()

            best_f1 = 0.0
            epochs_no_improve = 0
            
            for epoch in range(1, EPOCHS + 1):
                avg_loss = train_classification_model(model, train_pairs, optimizer, loss_func, device)
                accuracy, precision, recall, f1, fpr, fnr = evaluate_classification_model(model, val_pairs, device)

                if epoch % 10 == 0: # Print every 10 epochs to reduce verbosity
                    print(f"  Epoch {epoch:03d}/{EPOCHS} | Loss: {avg_loss:.4f} | Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%} | FPR: {fpr:.4%} | FNR: {fnr:.4%}")

                if f1 > best_f1:
                    best_f1 = f1
                    # Save best metrics for this fold
                    best_fold_metrics = (accuracy, precision, recall, f1, fpr, fnr)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                        print(f"  Early stopping triggered at epoch {epoch}.")
                        break
            
            print(f"Fold {fold+1}/{K_FOLD} Best F1: {best_fold_metrics[3]:.4f} | Precision: {best_fold_metrics[1]:.2%} | Recall: {best_fold_metrics[2]:.2%} | Accuracy: {best_fold_metrics[0]:.4f} | FPR: {best_fold_metrics[4]:.2%} | FNR: {best_fold_metrics[5]:.2%}")
            fold_results.append(best_fold_metrics)

        # --- This Run's Final Results ---
        print(f"\n--- Run {run}/{NUM_RUNS} Cross-Validation Summary ---")
        avg_accuracy = np.mean([res[0] for res in fold_results])
        avg_precision = np.mean([res[1] for res in fold_results])
        avg_recall = np.mean([res[2] for res in fold_results])
        avg_f1 = np.mean([res[3] for res in fold_results])
        avg_fpr = np.mean([res[4] for res in fold_results])
        avg_fnr = np.mean([res[5] for res in fold_results])

        std_accuracy = np.std([res[0] for res in fold_results])
        std_precision = np.std([res[1] for res in fold_results])
        std_recall = np.std([res[2] for res in fold_results])
        std_f1 = np.std([res[3] for res in fold_results])
        std_fpr = np.std([res[4] for res in fold_results])
        std_fnr = np.std([res[5] for res in fold_results])

        print(f"Run {run} Average F1-score: {avg_f1:.2%} ± {std_f1:.2%}")

        if avg_f1 > best_run_avg_f1:
            best_run_avg_f1 = avg_f1
            best_run_results = {
                'run': run,
                'avg_accuracy': avg_accuracy, 'std_accuracy': std_accuracy,
                'avg_precision': avg_precision, 'std_precision': std_precision,
                'avg_recall': avg_recall, 'std_recall': std_recall,
                'avg_f1': avg_f1, 'std_f1': std_f1,
                'avg_fpr': avg_fpr, 'std_fpr': std_fpr,
                'avg_fnr': avg_fnr, 'std_fnr': std_fnr,
            }
    
    return best_run_results


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GNN for Link Prediction Training')
    parser.add_argument('--dataset', type=str, default='D1', choices=['D1', 'D2', 'D3'],
                        help='Choose the dataset to use: D1, D2, or D3')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use. e.g., 0, 1, 2...')
    args = parser.parse_args()

    # Set device based on user input
    if torch.cuda.is_available():
        DEVICE = torch.device(f'cuda:{args.gpu}')
    else:
        DEVICE = torch.device('cpu')
    
    print(f"Using device: {DEVICE}")
    print(f"Loading data for dataset: {args.dataset}")

    # 1. Load all data once
    base_data = load_mixbroker_data(args.dataset)
    extra_train_data = load_and_prep_data_for_ranking(GROUND_TRUTH_FILES)
    
    base_data_np = np.array(base_data) # Convert to numpy array for easier indexing by KFold

    # 2. Define Ablation Study Configurations
    ablation_configs = []
    for use_graph in [True, False]:
        for use_extra in [True, False]:
            if use_graph:
                for gnn_type in ['GATv2Conv', 'SAGEConv', 'GATConv', 'TransformerConv']:
                    ablation_configs.append({
                        'use_graph': use_graph,
                        'gnn_type': gnn_type,
                        'use_extra_data': use_extra
                    })
            else:
                # If not using graph, GNN type doesn't matter.
                ablation_configs.append({
                    'use_graph': use_graph,
                    'gnn_type': 'None',
                    'use_extra_data': use_extra
                })

    # 3. Run all experiments
    for config in ablation_configs:
        best_run_results = run_experiment(config, base_data_np, extra_train_data, DEVICE)

        # --- Save Best Run Results to CSV ---
        if best_run_results:
            training_params = {
                'Dataset': args.dataset,
                'Use_Extra_Data': config['use_extra_data'],
                'Learning_Rate': LEARNING_RATE,
                'Epochs': EPOCHS,
                'Batch_Size': BATCH_SIZE,
                'K_Fold': K_FOLD,
                'Seed': SEED,
                'Patience': EARLY_STOPPING_PATIENCE,
                'Num_Runs': NUM_RUNS, # Hardcoded from inside run_experiment
                'Pos_Weight': POS_WEIGHT,
                'Use_Graph': config['use_graph'],
                'GNN_Type': config['gnn_type'],
            }
            save_results_to_csv(best_run_results, training_params, f'Results/{RUN_TIMESTAMP}_{args.dataset}.csv')
        else:
            print(f"Experiment with config {config} did not complete successfully.")

    print("\n✅ All ablation experiments complete.")
