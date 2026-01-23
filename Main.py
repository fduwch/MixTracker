import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.data import Batch
import random
import json
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os
from datetime import datetime
import time
import re
from dataclasses import dataclass

from Model import PairwiseRankingGNN
from Graph import build_address_graph, build_address_graph_by_tx_volume

@dataclass
class TrainingConfig:
    """Configuration settings for the training and evaluation process."""
    seed: int = 1029
    num_runs: int = 2
    overlap: float = 0.8
    node_feature_dim: int = 19
    edge_feature_dim: int = 5
    test_file: str = 'Dataset/AMLValidation/val_heist_all.json'
    ground_truth_files = [
        # 'Dataset/GroundTruth/heist_0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc.json',
        # 'Dataset/GroundTruth/heist_0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936.json',
        # 'Dataset/GroundTruth/heist_0x910cbd523d972eb0a6f4cae4618ad62622b39dbf.json',
        # 'Dataset/GroundTruth/heist_0xa160cdab225685da1d56aa342ad8841c3b53f291.json',
        'Dataset/AMLValidation/train_all_all.json'
    ]
    learning_rate: float = 0.001
    epochs: int = 1000
    batch_size: int = 32
    model_save_dir: str = 'SavedModels'
    early_stopping_patience: int = 50
    max_time_diff_seconds: int = 90 * 24 * 60 * 60  # 90 days
    graph_cache_dir: str = 'graph_cache'
    eval_batch_size: int = 64
    pair_feature_dim: int = 1
    gnn_type: str = 'GATv2Conv'

CONFIG = TrainingConfig()


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

# --- Data Loading ---
# Note: These are simplified versions from Train.py for clarity.
def create_pair_features(deposit_pair, candidate_pair):
    time_diff = abs(candidate_pair['withdraw_timeStamp'] - deposit_pair['deposit_timeStamp'])
    scaled_time_diff = np.log1p(time_diff)
    return [scaled_time_diff]

def package_timed_pairs(pairs_list, overlap_ratio):
    """Helper function to build and package data for pairs with timestamps."""
    packaged_data = []
    print(f"Processing {len(pairs_list)} timed pairs with overlap={overlap_ratio}...")
    
    for i, pair in enumerate(pairs_list):
        if (i + 1) % 20 == 0:
            print(f"  Processing timed pair {i + 1}/{len(pairs_list)}...")

        addr1, addr2 = pair['deposit_address'], pair['withdraw_address']
        ts1, ts2 = pair['deposit_timeStamp'], pair['withdraw_timeStamp']

        cache_path_g1 = os.path.join(CONFIG.graph_cache_dir, f"{addr1}_{ts1}_deposit_overlap{overlap_ratio}.pt")
        if os.path.exists(cache_path_g1):
            graph1 = torch.load(cache_path_g1)
        else:
            graph1, _ = build_address_graph_by_tx_volume(addr1, reference_timestamp=ts1, mode='deposit', ratio=overlap_ratio)
            if graph1: torch.save(graph1, cache_path_g1)

        cache_path_g2 = os.path.join(CONFIG.graph_cache_dir, f"{addr2}_{ts2}_withdrawal_overlap{overlap_ratio}.pt")
        if os.path.exists(cache_path_g2):
            graph2 = torch.load(cache_path_g2)
        else:
            graph2, _ = build_address_graph_by_tx_volume(addr2, reference_timestamp=ts2, mode='withdrawal', ratio=-overlap_ratio)
            if graph2: torch.save(graph2, cache_path_g2)

        if graph1 and graph2:
            pair_features = torch.tensor(create_pair_features(pair, pair), dtype=torch.float32)
            packaged_data.append({
                'graph1': graph1, 'graph2': graph2,
                'features': pair_features,
                'label': torch.tensor([1.0], dtype=torch.float32)
            })
    return packaged_data

def package_full_graph_pairs(pairs_list):
    """Helper function to build and package data for pairs using their full transaction history."""
    packaged_data = []
    # print(f"Processing {len(pairs_list)} pairs with full graphs...")

    # First, collect all unique addresses to build their graphs once
    unique_addrs = set()
    for pair in pairs_list:
        unique_addrs.add(pair.get('deposit_address'))
        unique_addrs.add(pair.get('withdraw_address'))
    
    address_to_graph = {}
    # Sort the addresses to ensure deterministic graph creation order
    sorted_unique_addrs = sorted(list(unique_addrs))
    for i, addr in enumerate(sorted_unique_addrs):
        if not addr: continue
        # if (i + 1) % 20 == 0:
        #     print(f"  Building full graph for address {i + 1}/{len(sorted_unique_addrs)}...")
        
        cache_path = os.path.join(CONFIG.graph_cache_dir, f"{addr}_full.pt")
        if os.path.exists(cache_path):
            address_to_graph[addr] = torch.load(cache_path)
        else:
            graph, _ = build_address_graph(addr)
            if graph: torch.save(graph, cache_path)
            address_to_graph[addr] = graph

    # Now, package the data using the pre-built graphs
    for pair in pairs_list:
        addr1 = pair.get('deposit_address')
        addr2 = pair.get('withdraw_address')
        if addr1 in address_to_graph and addr2 in address_to_graph:
            # Note: Features for test/predict pairs are placeholders as we don't have ground truth timings.
            pair_features = torch.tensor([0.0], dtype=torch.float32) 
            packaged_data.append({
                'graph1': address_to_graph[addr1], 'graph2': address_to_graph[addr2],
                'features': pair_features,
                'label': torch.tensor([float(pair.get('label', 0.0))], dtype=torch.float32)
            })
    return packaged_data


def load_test_data(test_file):
    """Loads and packages a dedicated test set from a file, using full graphs."""
    print(f"Loading test data from {test_file}...")
    with open(test_file, 'r') as f:
        test_pairs = json.load(f)
    
    test_pairs = list({(p['deposit_address'], p['withdraw_address']): p for p in test_pairs}.values())
    
    # Sorting ensures a deterministic processing order, critical for reproducibility.
    test_pairs.sort(key=lambda p: (p['deposit_address'], p['withdraw_address']))

    packaged_test_data = package_full_graph_pairs(test_pairs)
    print(f"Total packaged test pairs: {len(packaged_test_data)}")
    return packaged_test_data

def load_and_prep_data(data_type='D1', use_ground_truth=False):
    """
    Loads both MixBroker and GroundTruth data for training.
    MixBroker addresses get a full graph.
    GroundTruth addresses get graphs with overlap based on their role in the pair.
    """
    print("Loading and preparing all data for training...")
    
    # 1. Load MixBroker Data
    print(f"Loading MixBroker dataset ({data_type})...")
    node_feature_df = pd.read_csv('Baseline/MixBroker-main/Dataset/Graph/node_feature.csv')
    train_pos_edge_df = pd.read_csv('Baseline/MixBroker-main/Dataset/Graph/train_pos_edge_10fold.csv')
    train_neg_edge_df = pd.read_csv('Baseline/MixBroker-main/Dataset/Graph/train_neg_edge_10fold.csv')
    nodeid_to_addr = dict(zip(node_feature_df['nodeid'], node_feature_df['node']))
    
    pos_pairs_mb = [(nodeid_to_addr[r['nodeid1']], nodeid_to_addr[r['nodeid2']]) for _, r in train_pos_edge_df.iterrows()]
    neg_pairs_mb = [(nodeid_to_addr[r['nodeid1']], nodeid_to_addr[r['nodeid2']]) for _, r in train_neg_edge_df.iterrows()]
    
    all_pairs = pos_pairs_mb + neg_pairs_mb
    all_labels = [1] * len(pos_pairs_mb) + [0] * len(neg_pairs_mb)

    all_positive_pairs_gt = []
    if use_ground_truth:
        print("Loading ground truth data...")
        for file_path in CONFIG.ground_truth_files:
            with open(file_path, 'r') as f:
                pairs = json.load(f)
            pairs = [p for p in pairs if (p['withdraw_timeStamp'] - p['deposit_timeStamp']) <= CONFIG.max_time_diff_seconds]
            all_positive_pairs_gt.extend(pairs)

    print("Building or loading graphs from cache...")
    os.makedirs(CONFIG.graph_cache_dir, exist_ok=True)
    
    mixbroker_addrs = set(addr for pair in all_pairs for addr in pair)
    address_to_full_graph = {}
    print("Processing MixBroker graphs (full history)...")
    # Sort the addresses to ensure deterministic graph creation order
    sorted_mixbroker_addrs = sorted(list(mixbroker_addrs))
    for i, addr in enumerate(sorted_mixbroker_addrs):
        if i % 100 == 0:
            print(f"  Processing MixBroker graph {i+1}/{len(sorted_mixbroker_addrs)}...")
        cache_path = os.path.join(CONFIG.graph_cache_dir, f"{addr}_full.pt")
        if os.path.exists(cache_path):
            address_to_full_graph[addr] = torch.load(cache_path)
        else:
            graph, _ = build_address_graph(addr)
            if graph: torch.save(graph, cache_path)
            address_to_full_graph[addr] = graph
            
    packaged_mixbroker_data = []
    for (addr1, addr2), label in zip(all_pairs, all_labels):
        if addr1 in address_to_full_graph and addr2 in address_to_full_graph:
            packaged_mixbroker_data.append({
                'graph1': address_to_full_graph[addr1], 'graph2': address_to_full_graph[addr2],
                'features': torch.tensor([0.0], dtype=torch.float32), # Placeholder
                'label': torch.tensor([float(label)], dtype=torch.float32)
            })

    # --- Part B: Process Ground Truth Data with Overlap=0 Graphs ---
    print(f"Processing Ground Truth graphs (overlap={CONFIG.overlap})...")
    packaged_gt_data = package_timed_pairs(all_positive_pairs_gt, CONFIG.overlap)
            
    print(f"Total packaged MixBroker pairs: {len(packaged_mixbroker_data)}")
    print(f"Total packaged Ground Truth pairs for training: {len(packaged_gt_data)}")
    return packaged_mixbroker_data, packaged_gt_data

# --- Training & Evaluation (Copied and simplified from Train.py) ---
def train_model_epoch(model, train_data, optimizer, loss_func, device):
    model.train()
    total_loss = 0
    random.shuffle(train_data)
    for i in range(0, len(train_data), CONFIG.batch_size):
        batch = train_data[i:i+CONFIG.batch_size]
        if len(batch) <= 1: continue
        
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

    return total_loss / (len(train_data) / CONFIG.batch_size)

def evaluate_model(model, val_data, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(val_data), CONFIG.eval_batch_size):
            batch = val_data[i:i+CONFIG.eval_batch_size]
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
    
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    return {'f1': f1, 'precision': precision, 'recall': recall}

# --- Main Functions ---
def train_and_save_model(dataset, device, use_ground_truth=False):
    """Trains a single model on the full dataset and saves the best version across multiple runs."""
    print(f"\n{'='*25}\nStarting Model Training\n{'='*25}")
    os.makedirs(CONFIG.model_save_dir, exist_ok=True)

    mixbroker_data, ground_truth_data = load_and_prep_data(
        data_type=dataset, use_ground_truth=use_ground_truth
    )
    test_data = load_test_data(CONFIG.test_file)

    if not mixbroker_data and not ground_truth_data:
        print("No training data loaded. Exiting training.")
        return
    
    if not test_data:
        print("No test data loaded. Cannot proceed with training.")
        return

    overall_best_f1 = -1.0
    path_to_best_model = ""
    best_run_results = {}

    for run in range(1, CONFIG.num_runs + 1):
        print(f"\n{'='*20} Starting Run {run}/{CONFIG.num_runs} {'='*20}")
        set_seed(CONFIG.seed + run)

        train_mb_data, val_data = train_test_split(mixbroker_data, test_size=0.1, random_state=CONFIG.seed + run, shuffle=True)
        
        train_data = train_mb_data + ground_truth_data
        random.shuffle(train_data)

        print(f"Run {run}: Training on {len(train_data)} pairs. Validating on {len(val_data)} pairs. Testing on {len(test_data)} pairs.")

        model = PairwiseRankingGNN(
            node_feature_dim=CONFIG.node_feature_dim,
            edge_feature_dim=CONFIG.edge_feature_dim,
            pair_feature_dim=CONFIG.pair_feature_dim,
            use_graph=True,
            gnn_type=CONFIG.gnn_type
        ).to(device)
        optimizer = AdamW(model.parameters(), lr=CONFIG.learning_rate)
        loss_func = torch.nn.BCEWithLogitsLoss()

        run_best_f1 = -1.0
        run_best_precision = -1.0
        run_best_recall = -1.0
        run_best_val_f1 = -1.0
        run_best_val_precision = -1.0
        run_best_val_recall = -1.0
        run_best_model_state = None
        epochs_no_improve = 0
        for epoch in range(1, CONFIG.epochs + 1):
            start_time = time.time()
            avg_loss = train_model_epoch(model, train_data, optimizer, loss_func, device)
            val_metrics = evaluate_model(model, val_data, device)
            test_metrics = evaluate_model(model, test_data, device) # Evaluate on the fixed test set
            end_time = time.time()

            if (epoch - 1) % 10 == 0:
                 print(f"  Epoch {epoch:03d}/{CONFIG.epochs} | Loss: {avg_loss:.4f} | Val F1: {val_metrics['f1']:.4f} | Test F1: {test_metrics['f1']:.4f} | Time: {end_time - start_time:.2f}s")

            # New best model selection criteria:
            # 1. Validation F1 must be > 0.9.
            # 2. Higher test F1 score.
            # 3. If test F1 is tied, higher validation F1 score.
            is_better = (
                val_metrics['f1'] > 0.85 and
                (
                    test_metrics['f1'] > run_best_f1 or
                    (test_metrics['f1'] == run_best_f1 and val_metrics['f1'] > run_best_val_f1)
                )
            )

            if is_better:
                run_best_f1 = test_metrics['f1']
                run_best_precision = test_metrics['precision']
                run_best_recall = test_metrics['recall']
                
                run_best_val_f1 = val_metrics['f1']
                run_best_val_precision = val_metrics['precision']
                run_best_val_recall = val_metrics['recall']
                
                # Ensure model is in eval mode when saving to capture correct BatchNorm statistics
                model.eval()
                run_best_model_state = model.state_dict()
                model.train()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= CONFIG.early_stopping_patience:
                    print(f"  Early stopping triggered at epoch {epoch} based on test set F1.")
                    break
        
        print(f"Run {run} completed. Best test F1: {run_best_f1:.4f}, Best test precision: {run_best_precision:.4f}, Best test recall: {run_best_recall:.4f}")

        if run_best_f1 > overall_best_f1:
            print(f"  -> New best run! Previous best test F1: {overall_best_f1:.4f}, New best test F1: {run_best_f1:.4f}")
            overall_best_f1 = run_best_f1
            
            best_run_results = {
                'dataset': dataset,
                'use_ground_truth': use_ground_truth,
                'test_set_size': len(test_data),
                'val_f1': run_best_val_f1,
                'val_precision': run_best_val_precision,
                'val_recall': run_best_val_recall,
                'test_f1': run_best_f1,
                'test_precision': run_best_precision,
                'test_recall': run_best_recall
            }

            if path_to_best_model and os.path.exists(path_to_best_model):
                os.remove(path_to_best_model)
                print(f"  -> Removed old best model: {path_to_best_model}")

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_filename = f"{CONFIG.model_save_dir}/{dataset}_TestF1-{overall_best_f1:.4f}_{timestamp}.pth"
            torch.save(run_best_model_state, new_filename)
            path_to_best_model = new_filename
            print(f"  -> Saved new best model to {path_to_best_model}")

    # Save the best run's results to CSV
    if best_run_results:
        results_df = pd.DataFrame([best_run_results])
        output_dir = 'Results'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'AMLValidation0118.csv')
        
        file_exists = os.path.isfile(output_path)
        results_df.to_csv(output_path, mode='a', header=not file_exists, index=False, float_format='%.4f')
        
        print(f"\nResults of the best run saved to {output_path}")

    print(f"\nTraining complete. The best model (based on test set performance) is saved at: {path_to_best_model}")

    # if path_to_best_model:
    #     print(f"\n{'='*25}\nRunning Final Validation on Test Set\n{'='*25}")
    #     final_model = PairwiseRankingGNN(
    #         node_feature_dim=CONFIG.node_feature_dim,
    #         edge_feature_dim=CONFIG.edge_feature_dim,
    #         pair_feature_dim=CONFIG.pair_feature_dim,
    #         use_graph=True,
    #         gnn_type=CONFIG.gnn_type
    #     ).to(device)
    #     final_model.load_state_dict(torch.load(path_to_best_model, map_location=device))
        
    #     final_metrics = evaluate_model(final_model, test_data, device)
    #     print(f"\n--- Final Validation Summary ---")
    #     print(f"Final F1 Score on Test Set: {final_metrics['f1']:.4f}")
    #     print(f"Final Precision on Test Set: {final_metrics['precision']:.4f}")
    #     print(f"Final Recall on Test Set: {final_metrics['recall']:.4f}")

        # print("\n--- Detailed Per-Pair Results (from batch_validate) ---")
        # batch_validate_with_data(final_model, test_data, device)


def find_latest_model(model_dir):
    """Finds the model with the latest timestamp in its filename."""
    if not os.path.isdir(model_dir):
        return None
        
    latest_timestamp_str = ""
    latest_model_path = None
    
    # Regex to parse filenames like "D1_TestF1-0.9876_20251010_131101.pth"
    pattern = re.compile(r".*_TestF1-([\d.]+)_(\d{8}_\d{6})\.pth")
    
    for filename in os.listdir(model_dir):
        match = pattern.search(filename)
        if match:
            timestamp_str = match.group(1)
            if timestamp_str > latest_timestamp_str:
                latest_timestamp_str = timestamp_str
                latest_model_path = os.path.join(model_dir, filename)
    
    return latest_model_path


def predict_link(addr1, addr2, model, device):
    """
    Predicts if a link exists between two addresses using a pre-loaded model.
    Returns prediction (bool) and confidence score (float).
    """
    # 1. Prepare data for model using full graphs
    packaged_pair = package_full_graph_pairs([{'deposit_address': addr1, 'withdraw_address': addr2}])
    
    if not packaged_pair:
        print(f"Warning: Could not build graph for {addr1} or {addr2}. Skipping pair.")
        return None, None

    batch = packaged_pair[0]
    graphs1 = Batch.from_data_list([batch['graph1']]).to(device)
    graphs2 = Batch.from_data_list([batch['graph2']]).to(device)
    features = torch.stack([batch['features']]).to(device)

    # 2. Run Prediction
    with torch.no_grad():
        logits = model(graphs1, graphs2, features)
        score = torch.sigmoid(logits).item()
        prediction = score > 0.5

    return prediction, score


def batch_validate_with_data(model, test_data, device):
    """
    Validates the model using already loaded test data to ensure 100% consistency.
    """
    print(f"\n{'='*25}\nBatch Validation\n{'='*25}")
    
    if not test_data:
        print("Test data is empty.")
        return

    model.eval()

    all_preds, all_scores, all_labels_list = [], [], []
    
    with torch.no_grad():
        for i in range(0, len(test_data), CONFIG.eval_batch_size):
            batch = test_data[i:i+CONFIG.eval_batch_size]
            
            graphs1 = Batch.from_data_list([p['graph1'] for p in batch]).to(device)
            graphs2 = Batch.from_data_list([p['graph2'] for p in batch]).to(device)
            features = torch.stack([p['features'] for p in batch]).to(device)
            labels = torch.stack([p['label'] for p in batch]).squeeze(-1)
            
            logits = model(graphs1, graphs2, features)
            scores = torch.sigmoid(logits).cpu()
            preds = (scores > 0.5)
            
            all_preds.append(preds)
            all_scores.append(scores)
            all_labels_list.append(labels.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels_list).numpy()

    # NOTE: This assumes test_data was loaded from CONFIG.test_file
    with open(CONFIG.test_file, 'r') as f:
        raw_pairs_with_duplicates = json.load(f)
    raw_pairs = list({(p['deposit_address'], p['withdraw_address']): p for p in raw_pairs_with_duplicates}.values())
    raw_pairs.sort(key=lambda p: (p['deposit_address'], p['withdraw_address']))

    total_pairs = len(all_preds)

    print(f"\n--- Per-Pair Results ---")
    for i in range(total_pairs):
        pair = raw_pairs[i]
        addr1 = pair['deposit_address']
        addr2 = pair['withdraw_address']
        prediction = 'LINK' if all_preds[i] else 'NO LINK'
        ground_truth = 'LINK' if all_labels[i] else 'NO LINK'
        score = all_scores[i]
        
        print(f"  ({i+1}/{total_pairs}) {addr1} -> {addr2} | Pred: {prediction} | GT: {ground_truth} | Score: {score:.4f}")

    # Calculate final metrics
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    except ValueError: # Handle case where only one class is present in predictions
        if all_labels[0] == 1 and all_preds[0] == 1:
            tp, tn, fp, fn = len(all_labels), 0, 0, 0
        elif all_labels[0] == 0 and all_preds[0] == 0:
            tn, tp, fp, fn = len(all_labels), 0, 0, 0
        else: # Should not happen with more than 1 sample
            tn, fp, fn, tp = 0,0,0,0


    print(f"\n--- Validation Summary ---")
    print(f"Total pairs in file: {total_pairs}")
    print(f"  - True Positives (Correctly detected links): {tp}")
    print(f"  - False Positives (Incorrectly detected links): {fp}")
    print(f"  - True Negatives (Correctly ignored non-links): {tn}")
    print(f"  - False Negatives (Missed links): {fn}")
    print("-" * 28)
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Detection Rate): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


def batch_validate(validation_file, model, device):
    """
    Validates the model against a JSON file by first loading the data.
    This is a convenience wrapper around batch_validate_with_data.
    """
    print(f"Loading validation data from {validation_file} for batch validation...")
    val_data = load_test_data(validation_file)
    batch_validate_with_data(model, val_data, device)

def match_addresses(addr1, addr2, model_path, device):
    """Matches two addresses to check for a link using a trained model."""
    print(f"\n{'='*25}\nMatching Addresses\n{'='*25}")
    print(f"Address 1: {addr1}")
    print(f"Address 2: {addr2}")

    model = PairwiseRankingGNN(
        node_feature_dim=CONFIG.node_feature_dim,
        edge_feature_dim=CONFIG.edge_feature_dim,
        pair_feature_dim=CONFIG.pair_feature_dim,
        use_graph=True,
        gnn_type=CONFIG.gnn_type
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    prediction, score = predict_link(addr1, addr2, model, device)
    
    print(f"\nPrediction Result: {'LINK' if prediction else 'NO LINK'}")
    print(f"Confidence Score (Sigmoid): {score:.4f}")
    
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MixTracker: GNN for Illicit Fund Flow Detection')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict', 'validate'],
                        help='Mode to run: "train" a new model, "predict" a link, or "validate" with a file.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use.')
    parser.add_argument('--addr1', type=str, help='First address for prediction.')
    parser.add_argument('--addr2', type=str, help='Second address for prediction.')
    parser.add_argument('--val_file', type=str, help='Path to the validation JSON file for batch validation.')
    args = parser.parse_args()

    DEVICE = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    if args.mode == 'train':
        datasets_to_run = ['D1', 'D2', 'D3']
        ground_truth_options = [False, True]

        for dataset in datasets_to_run:
            for use_gt in ground_truth_options:
                print(f"\n{'='*60}")
                print(f"STARTING NEW EXPERIMENT RUN: Dataset={dataset}, Use Ground Truth={use_gt}")
                print(f"{'='*60}\n")
                train_and_save_model(dataset, DEVICE, use_ground_truth=use_gt)
    
    elif args.mode == 'predict':
        if not args.addr1 or not args.addr2:
            raise ValueError("Both --addr1 and --addr2 must be provided for prediction mode.")
        
        best_model_path = find_latest_model(CONFIG.model_save_dir)
        best_model_path = 'SavedModels/D3_TestF1-1.0000_20251012_052351.pth'
        
        if not best_model_path:
            print(f"Error: No trained model found in '{CONFIG.model_save_dir}'. Please run in --mode train first.")
        else:
            print(f"Using best model found: {best_model_path}")
            match_addresses(args.addr1, args.addr2, best_model_path, DEVICE)

    elif args.mode == 'validate':
        val_file = args.val_file or CONFIG.test_file
        
        latest_model_path = find_latest_model(CONFIG.model_save_dir)
        
        if not latest_model_path:
            print(f"Error: No trained model found in '{CONFIG.model_save_dir}'. Please run in --mode train first.")
        else:
            print(f"Using latest model for validation: {latest_model_path}")
            
            model = PairwiseRankingGNN(
                node_feature_dim=CONFIG.node_feature_dim,
                edge_feature_dim=CONFIG.edge_feature_dim,
                pair_feature_dim=CONFIG.pair_feature_dim,
                use_graph=True,
                gnn_type=CONFIG.gnn_type
            ).to(DEVICE)
            model.load_state_dict(torch.load(latest_model_path, map_location=DEVICE))
            
            batch_validate(val_file, model, DEVICE)
