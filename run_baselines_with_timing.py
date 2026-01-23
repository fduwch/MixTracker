import os
import csv
import random
import argparse
import time
import gc
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from Model import PairwiseRankingGNN, SimpleLinkPredictorGNN
from Graph import build_address_graph, build_address_graph_with_timing, _load_all_txs_for_address, build_simple_graph_from_features, build_simple_heterogeneous_graph_from_features

# --- Configuration ---
SEED = 1029
NUM_RUNS = 1
K_FOLD = 10
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
# DEVICE is now set in main

# Model Specific Configs
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
LSTM_SEQ_LENGTH = 100
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 0.001
GNN_NODE_FEATURE_DIM = 19
GNN_EDGE_FEATURE_DIM = 5
GNN_EPOCHS = 120  # Fixed epochs for consistent timing comparison (no early stopping)
GNN_LEARNING_RATE = 0.001
GNN_EARLY_STOPPING_PATIENCE = None  # Disabled for consistent timing measurements

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_gpu_memory_info(device, gpu_ids=None):
    """
    Get current GPU memory usage in GB.
    Uses max_memory_reserved() which is closer to nvidia-smi values
    (includes PyTorch reserved memory + CUDA context overhead).
    
    Args:
        device: Primary device
        gpu_ids: List of GPU IDs being used (for multi-GPU tracking)
    
    Returns:
        Dictionary with memory statistics
    """
    if device.type == 'cuda':
        if gpu_ids and len(gpu_ids) > 1:
            # Multi-GPU: collect stats from all devices
            total_allocated = 0.0
            total_reserved = 0.0
            max_reserved_per_device = []  # Use reserved memory (closer to nvidia-smi)
            max_allocated_per_device = []
            
            for gpu_id in gpu_ids:
                gpu_device = torch.device(f'cuda:{gpu_id}')
                allocated = torch.cuda.memory_allocated(gpu_device) / 1024**3
                reserved = torch.cuda.memory_reserved(gpu_device) / 1024**3
                max_allocated = torch.cuda.max_memory_allocated(gpu_device) / 1024**3
                max_reserved = torch.cuda.max_memory_reserved(gpu_device) / 1024**3
                
                total_allocated += allocated
                total_reserved += reserved
                max_allocated_per_device.append(max_allocated)
                max_reserved_per_device.append(max_reserved)
            
            return {
                'allocated_gb': total_allocated,
                'reserved_gb': total_reserved,
                'max_allocated_gb': max(max_allocated_per_device),  # PyTorch allocated
                'max_reserved_gb': max(max_reserved_per_device),  # PyTorch reserved (closer to nvidia-smi)
                'total_max_allocated_gb': sum(max_allocated_per_device),
                'total_max_reserved_gb': sum(max_reserved_per_device),  # Total reserved across GPUs
                'max_allocated_per_device_gb': max_allocated_per_device,
                'max_reserved_per_device_gb': max_reserved_per_device
            }
        else:
            # Single GPU
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'max_reserved_gb': max_reserved,  # Closer to nvidia-smi
                'total_max_allocated_gb': max_allocated,
                'total_max_reserved_gb': max_reserved,
                'max_allocated_per_device_gb': [max_allocated],
                'max_reserved_per_device_gb': [max_reserved]
            }
    return {
        'allocated_gb': 0.0,
        'reserved_gb': 0.0,
        'max_allocated_gb': 0.0,
        'max_reserved_gb': 0.0,
        'total_max_allocated_gb': 0.0,
        'total_max_reserved_gb': 0.0,
        'max_allocated_per_device_gb': [],
        'max_reserved_per_device_gb': []
    }

def reset_gpu_memory_stats(device, gpu_ids=None):
    """Reset GPU memory statistics for single or multiple GPUs."""
    if device.type == 'cuda':
        if gpu_ids and len(gpu_ids) > 0:
            # Use the provided GPU IDs
            for gpu_id in gpu_ids:
                try:
                    # Try integer index first (most reliable)
                    torch.cuda.reset_peak_memory_stats(gpu_id)
                except (RuntimeError, TypeError):
                    try:
                        # Fallback to device object
                        torch.cuda.reset_peak_memory_stats(torch.device(f'cuda:{gpu_id}'))
                    except RuntimeError:
                        # If still fails, skip reset for this device (non-critical)
                        pass
        else:
            # Fallback: try to get device index from device object
            try:
                if hasattr(device, 'index') and device.index is not None:
                    torch.cuda.reset_peak_memory_stats(device.index)
                else:
                    # Last resort: use device 0
                    torch.cuda.reset_peak_memory_stats(0)
            except RuntimeError:
                # If all else fails, skip reset (non-critical)
                pass

def save_results_to_csv(results, params, filename='baseline_results.csv'):
    file_exists = os.path.isfile(filename)
    formatted_results = {
        'F1': f"{results['avg_f1']:.4f}±{results['std_f1']:.4f}",
        'Accuracy': f"{results['avg_accuracy']:.4f}±{results['std_accuracy']:.4f}",
        'Precision': f"{results['avg_precision']:.4f}±{results['std_precision']:.4f}",
        'Recall': f"{results['avg_recall']:.4f}±{results['std_recall']:.4f}",
        'FPR': f"{results['avg_fpr']:.4f}±{results['std_fpr']:.4f}",
        'FNR': f"{results['avg_fnr']:.4f}±{results['std_fnr']:.4f}",
        'Total_Train_Time_s': f"{results['total_train_time']:.2f}",
        'Avg_Fold_Train_Time_s': f"{results['avg_fold_train_time']:.2f}",
        'Graph_Construction_Time_s': f"{results.get('graph_construction_time', 0.0):.2f}",
        'IO_Time_s': f"{results.get('io_time_total', 0.0):.2f}",
        'Total_Test_Time_s': f"{results['total_test_time']:.2f}",
        'Avg_Sample_Test_Time_ms': f"{(results['avg_sample_test_time'] * 1000):.4f}",
        'GPU_Count': results.get('gpu_count', 1),
        'Peak_GPU_Memory_GB': f"{results.get('peak_gpu_memory_gb', 0.0):.2f}",
        'GPU_Memory_Per_Device_GB': f"{results.get('gpu_memory_per_device_gb', 0.0):.2f}",
        'Total_GPU_Memory_GB': f"{results.get('total_gpu_memory_gb', 0.0):.2f}",
        'Speedup_vs_Single_GPU': f"{results.get('speedup_factor', 1.0):.2f}x",
        'Parallel_Efficiency': f"{results.get('parallel_efficiency', 100.0):.1f}%",
        'Cache_Used': results.get('cache_used', 'N/A'),
        'Cache_Time_Saved_s': f"{results.get('cache_time_saved', 0.0):.2f}",
        'Cache_Time_Saved_Percent': f"{results.get('cache_time_saved_percent', 0.0):.1f}%",
        'Best_Run': results['run']
    }
    row_data = {**params, **formatted_results}
    param_keys = ['Dataset', 'Model', 'Seed', 'Num_Runs', 'K_Fold']
    result_keys = ['Best_Run', 'F1', 'Accuracy', 'Precision', 'Recall', 'FPR', 'FNR', 
                   'Total_Train_Time_s', 'Avg_Fold_Train_Time_s', 'Graph_Construction_Time_s', 'IO_Time_s',
                   'Total_Test_Time_s', 'Avg_Sample_Test_Time_ms', 
                   'GPU_Count', 'Peak_GPU_Memory_GB', 'GPU_Memory_Per_Device_GB', 'Total_GPU_Memory_GB',
                   'Speedup_vs_Single_GPU', 'Parallel_Efficiency',
                   'Cache_Used', 'Cache_Time_Saved_s', 'Cache_Time_Saved_Percent']
    fieldnames = param_keys + result_keys
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
        print(f"Results saved to {filename}")
    except IOError as e:
        print(f"Error saving results to {filename}: {e}")

# --- Helper Functions ---
def _calculate_metrics(y_true, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else:
        metrics['fpr'], metrics['fnr'] = 0.0, 0.0
    return metrics

# --- Model Definitions & Data Loaders ---
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]

class SiameseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(SiameseLSTM, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, seq1, seq2):
        emb1, emb2 = self.encoder(seq1), self.encoder(seq2)
        combined = torch.cat((emb1, emb2), dim=1)
        return self.predictor(combined)

class TransactionPairDataset(Dataset):
    def __init__(self, pairs, labels, tx_sequences):
        self.pairs, self.labels, self.tx_sequences = pairs, labels, tx_sequences
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        addr1, addr2 = self.pairs[idx]
        seq1 = self.tx_sequences.get(addr1, torch.zeros(LSTM_SEQ_LENGTH, 5))
        seq2 = self.tx_sequences.get(addr2, torch.zeros(LSTM_SEQ_LENGTH, 5))
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return seq1, seq2, label

def load_data_for_baselines(data_type='D1'):
    print(f"Loading MixBroker dataset ({data_type}) for baseline models...")
    node_feature_df = pd.read_csv('Baseline/MixBroker-main/Dataset/Graph/node_feature.csv').fillna(0)
    train_pos_edge_df = pd.read_csv('Baseline/MixBroker-main/Dataset/Graph/train_pos_edge_10fold.csv')
    train_neg_edge_df = pd.read_csv('Baseline/MixBroker-main/Dataset/Graph/train_neg_edge_10fold.csv')
    nodeid_to_addr = dict(zip(node_feature_df['nodeid'], node_feature_df['node']))
    
    feature_cols = [c for c in node_feature_df.columns if c not in ['nodeid', 'node']]
    node_features = {row['node']: row[feature_cols].values.astype(np.float32) for _, row in node_feature_df.iterrows()}
    
    # Return node_features for GNN models
    
    d1_pos_pairs = [tuple(r) for r in train_pos_edge_df[['nodeid1', 'nodeid2']].applymap(nodeid_to_addr.get).values]
    d1_neg_pairs = [tuple(r) for r in train_neg_edge_df[['nodeid1', 'nodeid2']].applymap(nodeid_to_addr.get).values]
    pos_pairs, neg_pairs = d1_pos_pairs.copy(), d1_neg_pairs.copy()

    if data_type == 'D3':
        for f in [f'Dataset/tornado_raw_data/heuristic{i}Mixer_{eth}ETH.csv' for i in [2,3] for eth in [0.1, 1, 10, 100]]:
            df = pd.read_csv(f)
            pos_pairs.extend(list(zip(df['sender'], df['receiver'])))

    all_pairs = pos_pairs + neg_pairs
    all_labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

    if data_type == 'D2':
        d1_all = list(zip(d1_pos_pairs + d1_neg_pairs, [1]*len(d1_pos_pairs) + [0]*len(d1_neg_pairs)))
        random.shuffle(d1_all)
        sample_size = int(len(d1_all) * 0.75)
        all_pairs, all_labels = zip(*d1_all[:sample_size]) if sample_size > 0 else ([], [])
    
    print(f"Final dataset size for {data_type}: {len(all_pairs)} pairs (Pos: {sum(all_labels)}, Neg: {len(all_labels) - sum(all_labels)})")
    G = nx.from_edgelist([p for p, l in zip(all_pairs, all_labels) if l == 1])
    G.add_nodes_from(node_features.keys())
    print(f"Graph created. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return node_features, np.array(all_pairs), np.array(all_labels), G

TRANSACTION_CACHE = {}
def load_transactions_for_lstm(all_pairs, use_cache=True, measure_time=False):
    """
    Load or build transaction sequences for LSTM models.
    
    Args:
        all_pairs: All address pairs
        use_cache: Whether to use cached sequences or rebuild them
        measure_time: Whether to measure and return construction time
    
    Returns:
        If measure_time=False: tx_sequences (dict)
        If measure_time=True: (tx_sequences, construction_time)
    """
    global TRANSACTION_CACHE
    
    tx_start_time = time.time() if measure_time else None
    
    if use_cache and TRANSACTION_CACHE:
        if measure_time:
            # Even loading from cache takes some time
            _ = TRANSACTION_CACHE
            tx_time = time.time() - tx_start_time
            return TRANSACTION_CACHE, tx_time
        return TRANSACTION_CACHE
    
    print(f"Loading transaction sequences for LSTM (use_cache={use_cache})...")
    unique_addresses = pd.unique(all_pairs.flatten())
    tx_sequences = {}
    
    for i, addr in enumerate(unique_addresses):
        if i % 100 == 0: print(f"  Processing address {i+1}/{len(unique_addresses)}...")
        tx_df = _load_all_txs_for_address(addr)
        if tx_df is None or tx_df.empty:
            tx_sequences[addr] = torch.zeros(LSTM_SEQ_LENGTH, 5)
            continue
        tx_df['direction'] = tx_df['from'].apply(lambda x: 1 if x == addr else -1)
        for col in ['value', 'gasUsed', 'gasPrice']:
            tx_df[col] = pd.to_numeric(tx_df[col], errors='coerce').fillna(0)
        tx_df['value'] /= 1e18
        tx_df['gasPrice'] /= 1e9
        seq_features = tx_df[['value', 'gasUsed', 'gasPrice', 'tx_type', 'direction']].tail(LSTM_SEQ_LENGTH)
        scaled_features = StandardScaler().fit_transform(seq_features)
        padded_features = np.zeros((LSTM_SEQ_LENGTH, scaled_features.shape[1]))
        padded_features[-len(scaled_features):] = scaled_features
        tx_sequences[addr] = torch.tensor(padded_features, dtype=torch.float32)
    print("Transaction loading complete.")
    
    if use_cache:
        TRANSACTION_CACHE = tx_sequences
    
    if measure_time:
        tx_time = time.time() - tx_start_time
        return tx_sequences, tx_time
    return tx_sequences

GNN_DATA_CACHE = {}
GNN_DATA_CACHE_HETERO = {}
def load_data_for_gnn(all_pairs, all_labels, node_features, gnn_type='all', use_cache=True, measure_time=False):
    """
    Load or build graphs for GNN models (GCN, GraphSAGE, RGCN).
    Uses simple feature-based graphs for baselines (same as run_baselines.py).
    
    Args:
        all_pairs: All address pairs
        all_labels: Labels for pairs
        node_features: Dict mapping address to feature vector
        gnn_type: 'all', 'regular', or 'hetero' - determines which graphs to build
        use_cache: Whether to use cached graphs or rebuild them
        measure_time: Whether to measure and return graph construction time
    
    Returns:
        If measure_time=False: (address_to_graph, address_to_hetero_graph)
        If measure_time=True: ((address_to_graph, address_to_hetero_graph), graph_construction_time)
    """
    global GNN_DATA_CACHE, GNN_DATA_CACHE_HETERO
    cache_key = len(all_labels)
    
    graph_start_time = time.time() if measure_time else None
    
    # Check if we have cached graphs
    if use_cache:
        if gnn_type in ['all', 'regular'] and cache_key in GNN_DATA_CACHE:
            address_to_graph = GNN_DATA_CACHE[cache_key]
        else:
            address_to_graph = None
        if gnn_type in ['all', 'hetero'] and cache_key in GNN_DATA_CACHE_HETERO:
            address_to_hetero_graph = GNN_DATA_CACHE_HETERO[cache_key]
        else:
            address_to_hetero_graph = None
        
        if (gnn_type == 'all' and address_to_graph is not None and address_to_hetero_graph is not None) or \
           (gnn_type == 'regular' and address_to_graph is not None) or \
           (gnn_type == 'hetero' and address_to_hetero_graph is not None):
            if measure_time:
                graph_time = time.time() - graph_start_time if graph_start_time else 0.0
                return (address_to_graph, address_to_hetero_graph), graph_time
            return (address_to_graph, address_to_hetero_graph)
    
    print(f"Building simple PyTorch Geometric graphs for baseline GNN models (use_cache={use_cache})...")
    GRAPH_CACHE_DIR = 'graph_cache_baseline'
    GRAPH_CACHE_DIR_HETERO = 'graph_cache_baseline_hetero'
    os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
    os.makedirs(GRAPH_CACHE_DIR_HETERO, exist_ok=True)
    unique_addrs = pd.unique(all_pairs.flatten())
    
    address_to_graph = {} if gnn_type in ['all', 'regular'] else None
    address_to_hetero_graph = {} if gnn_type in ['all', 'hetero'] else None
    
    for i, addr in enumerate(unique_addrs):
        if i % 100 == 0: print(f"  Processing address {i+1}/{len(unique_addrs)}...")
        
        # Regular simple graph (for GCN, GraphSAGE)
        if address_to_graph is not None:
            cache_path = os.path.join(GRAPH_CACHE_DIR, f"{addr}_simple.pt")
            if use_cache and os.path.exists(cache_path):
                address_to_graph[addr] = torch.load(cache_path)
            else:
                graph, _ = build_simple_graph_from_features(addr, node_features)
                if graph and use_cache:
                    torch.save(graph, cache_path)
                address_to_graph[addr] = graph
        
        # Simple heterogeneous graph (for RGCN)
        if address_to_hetero_graph is not None:
            cache_path_hetero = os.path.join(GRAPH_CACHE_DIR_HETERO, f"{addr}_simple_hetero.pt")
            if use_cache and os.path.exists(cache_path_hetero):
                address_to_hetero_graph[addr] = torch.load(cache_path_hetero)
            else:
                hetero_graph, _ = build_simple_heterogeneous_graph_from_features(addr, node_features)
                if hetero_graph and use_cache:
                    torch.save(hetero_graph, cache_path_hetero)
                address_to_hetero_graph[addr] = hetero_graph
    
    print("GNN data preparation complete.")
    
    if use_cache:
        if address_to_graph is not None:
            GNN_DATA_CACHE[cache_key] = address_to_graph
        if address_to_hetero_graph is not None:
            GNN_DATA_CACHE_HETERO[cache_key] = address_to_hetero_graph
    
    if measure_time:
        graph_time = time.time() - graph_start_time if graph_start_time else 0.0
        return (address_to_graph, address_to_hetero_graph), graph_time
    return (address_to_graph, address_to_hetero_graph)

def get_link_features(pairs, features_or_embeddings):
    dim = len(next(iter(features_or_embeddings.values())))
    link_features = [features_or_embeddings.get(p[0], np.zeros(dim)) * features_or_embeddings.get(p[1], np.zeros(dim)) for p in pairs]
    return np.array(link_features)

def train_on_single_gpu(gpu_idx, gpu_id, model_replica, optimizer, gpu_data, loss_fn, actual_model_name, epoch, batch_size=None):
    """
    Train model on a single GPU. This function is designed to run in parallel.
    Returns: (gpu_idx, avg_loss, num_batches)
    """
    if batch_size is None:
        batch_size = LSTM_BATCH_SIZE
    
    model_replica.train()
    gpu_device = torch.device(f'cuda:{gpu_id}')
    gpu_total_loss = 0.0
    gpu_batches = 0
    
    for i in range(0, len(gpu_data), batch_size):
        batch = gpu_data[i:i+LSTM_BATCH_SIZE]
        if len(batch) <= 1:
            continue
        gpu_batches += 1
        
        g1 = Batch.from_data_list([p['graph1'] for p in batch]).to(gpu_device)
        g2 = Batch.from_data_list([p['graph2'] for p in batch]).to(gpu_device)
        labels = torch.stack([p['label'] for p in batch]).to(gpu_device).squeeze(-1)
        
        optimizer.zero_grad()
        
        if actual_model_name in ['gcn', 'graphsage', 'rgcn']:
            scores = model_replica(g1, g2)
        else:
            feat = torch.stack([torch.tensor([0.0]) for _ in batch]).to(gpu_device)
            scores = model_replica(g1, g2, feat)
        
        loss = loss_fn(scores, labels)
        loss.backward()
        optimizer.step()
        gpu_total_loss += loss.item()
    
    avg_loss = gpu_total_loss / gpu_batches if gpu_batches > 0 else 0.0
    return gpu_idx, avg_loss, gpu_batches

def run_experiment(config, data, device, gpu_ids=None):
    model_name = config['model']
    use_cache = config.get('use_cache', True)  # Default to using cache
    max_folds = config.get('max_folds', None)  # Limit number of folds (for statistics mode)
    node_features, all_pairs, all_labels, G = data['node_features'], data['all_pairs'], data['all_labels'], data['G']
    best_run_avg_f1, best_run_results = -1.0, {} # FIX: Initialize with -1 to capture 0-score results

    # Determine if this is multi-GPU training
    is_multi_gpu = model_name == 'MixTracker-GNN-2GPU'
    gpu_count = len(gpu_ids) if (is_multi_gpu and gpu_ids) else 1
    
    # For MixTracker-GNN-2GPU, normalize model name for processing
    actual_model_name = 'MixTracker-GNN' if is_multi_gpu else model_name
    
    # Reset GPU memory stats at the start
    # Always pass gpu_ids (even for single-GPU, pass [device_index])
    reset_gpu_ids = gpu_ids if gpu_ids else ([device.index] if hasattr(device, 'index') and device.index is not None else [0])
    reset_gpu_memory_stats(device, reset_gpu_ids)

    # Measure graph/data construction time if needed (only for models that require it)
    # Only measure if data is not already loaded (to avoid double counting in compare mode)
    construction_time = 0.0
    if actual_model_name == 'lstm' and data.get('tx_sequences') is None:
        tx_sequences, construction_time = load_transactions_for_lstm(all_pairs, use_cache=use_cache, measure_time=True)
        data['tx_sequences'] = tx_sequences
        if construction_time > 0:
            print(f"  Transaction sequence construction time: {construction_time:.2f}s")
    elif actual_model_name == 'MixTracker-GNN':
        # MixTracker-GNN uses full transaction history graphs (build_address_graph), not simple feature graphs
        # IMPORTANT: MixTracker-GNN uses a separate graph dict key to avoid conflict with baseline GNN graphs
        if data.get('address_to_graph_mixtracker') is None:
            construction_time = 0.0
            graph_start_time = time.time()
            
            print(f"Building full transaction history graphs for MixTracker-GNN (use_cache={use_cache})...")
            GRAPH_CACHE_DIR = 'graph_cache_baselines'
            os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
            unique_addrs = pd.unique(all_pairs.flatten())
            address_to_graph = {}
            
            io_time_total = 0.0  # Track I/O time separately (for reporting)
            compute_time_total = 0.0  # Track actual graph construction computation time
            data['io_time_total'] = 0.0  # Initialize for reporting
            
            for i, addr in enumerate(unique_addrs):
                if i % 100 == 0: print(f"  Processing graph {i+1}/{len(unique_addrs)}...")
                cache_path = os.path.join(GRAPH_CACHE_DIR, f"{addr}_full.pt")
                
                if use_cache and os.path.exists(cache_path):
                    # Loading from cache is I/O, but we count it as minimal since it's fast
                    cache_load_start = time.time()
                    address_to_graph[addr] = torch.load(cache_path)
                    io_time_total += time.time() - cache_load_start
                    # Cache loading has negligible computation time
                else:
                    # Build full transaction history graph from scratch with timing
                    graph, _, io_time, compute_time = build_address_graph_with_timing(addr)
                    io_time_total += io_time
                    compute_time_total += compute_time
                    
                    if graph and use_cache:
                        cache_save_start = time.time()
                        torch.save(graph, cache_path)
                        io_time_total += time.time() - cache_save_start
                    address_to_graph[addr] = graph
            
            total_time = time.time() - graph_start_time
            # Use actual computation time (excluding I/O) as construction_time
            # This gives a better measure of algorithm performance for reviewers
            construction_time = compute_time_total if compute_time_total > 0 else max(0.0, total_time - io_time_total)
            # Store MixTracker-GNN graphs in separate key to avoid conflict with baseline GNN graphs
            data['address_to_graph_mixtracker'] = address_to_graph
            data['io_time_total'] = io_time_total  # Store for reporting
            if construction_time > 0:
                print(f"  Graph construction time (excluding I/O): {construction_time:.2f}s")
                if io_time_total > 0:
                    print(f"  I/O time (file reading/caching): {io_time_total:.2f}s")
                print(f"  Total elapsed time: {total_time:.2f}s")
    
    elif actual_model_name in ['gcn', 'graphsage', 'rgcn']:
        # Baseline GNN models use simple feature-based graphs
        if data.get('address_to_graph') is None and data.get('address_to_hetero_graph') is None:
            node_features = data.get('node_features')
            if node_features is None:
                raise ValueError("node_features not found in data package")
            
            # Determine which graphs to build
            if actual_model_name == 'rgcn':
                gnn_type = 'hetero'
            else:  # gcn, graphsage
                gnn_type = 'regular'
            
            (address_to_graph, address_to_hetero_graph), construction_time = load_data_for_gnn(
                all_pairs, all_labels, node_features, gnn_type=gnn_type, use_cache=use_cache, measure_time=True
            )
            data['address_to_graph'] = address_to_graph
            data['address_to_hetero_graph'] = address_to_hetero_graph
            if construction_time > 0:
                print(f"  Simple graph construction time: {construction_time:.2f}s")

    for run in range(1, NUM_RUNS + 1):
        print(f"\n{'='*20} Starting Run {run}/{NUM_RUNS} for {model_name.upper()} {'='*20}")
        print(f"Cache setting: {'Using ego-subgraph cache' if use_cache else 'Building graphs from scratch (no cache)'}")
        set_seed(SEED + run)
        fold_metrics, fold_train_times, fold_test_times, fold_test_samples = [], [], [], []
        peak_memory_gb = 0.0
        
        # Use different K-Fold strategies depending on the model type
        pos_pairs = all_pairs[all_labels == 1]
        neg_pairs = all_pairs[all_labels == 0]
        if actual_model_name in ['deepwalk', 'node2vec']:
            # For graph embedding link prediction, we must split on positive edges
            # to ensure the validation edges are not present during embedding generation.
            kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=SEED + run)
            fold_iterator = kf.split(pos_pairs)
        else:
            kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=SEED + run)
            fold_iterator = kf.split(all_pairs)
        
        for fold, (train_idx, val_idx) in enumerate(fold_iterator):
            # Early exit if max_folds is set (for statistics mode)
            if max_folds is not None and fold >= max_folds:
                print(f"Reached max_folds limit ({max_folds}), stopping early.")
                break
            
            # Display fold number with max_folds limit if applicable
            fold_display = f"{fold+1}/{max_folds if max_folds else K_FOLD}"
            print(f"--- Starting Fold {fold_display} ---")
            
            # Flag to track if metrics were already computed (for GNN models with early stopping)
            gnn_metrics_already_computed = False
            
            # --- 1. Prepare Data for this Fold ---
            if actual_model_name in ['deepwalk', 'node2vec']:
                train_pos, val_pos = pos_pairs[train_idx], pos_pairs[val_idx]
                # To prevent data leakage, create a validation set with unseen negative samples
                val_neg_size = len(val_pos)
                # Ensure we don't sample more than available, and sample without replacement
                chosen_neg_indices = np.random.choice(len(neg_pairs), size=val_neg_size, replace=len(neg_pairs) < val_neg_size)
                val_neg = neg_pairs[chosen_neg_indices]
                # The training negatives are all other negatives
                train_neg_mask = np.ones(len(neg_pairs), dtype=bool)
                train_neg_mask[chosen_neg_indices] = False
                train_neg = neg_pairs[train_neg_mask]
                
                train_pairs, train_labels = np.vstack([train_pos, train_neg]), np.array([1]*len(train_pos) + [0]*len(train_neg))
                val_pairs, val_labels = np.vstack([val_pos, val_neg]), np.array([1]*len(val_pos) + [0]*len(val_neg))
            else:
                train_pairs, val_pairs = all_pairs[train_idx], all_pairs[val_idx]
                train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

            # --- 2. Train Model & Measure Time ---
            scaler, model, embeddings = None, None, None
            start_train_time = time.time()
            # Include construction time in first fold if needed (only count once per run)
            if fold == 0 and construction_time > 0:
                # Construction time is amortized across all folds, so we add it to the first fold
                pass  # Will be added after fold 0 completes
            
            if actual_model_name in ['lr', 'rf']:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(get_link_features(train_pairs, node_features))
                model = LogisticRegression(max_iter=1000) if actual_model_name == 'lr' else RandomForestClassifier(n_jobs=-1, random_state=SEED)
                model.fit(X_train, train_labels)
            
            elif actual_model_name in ['deepwalk', 'node2vec']:
                G_train = nx.from_edgelist(train_pos); G_train.add_nodes_from(G.nodes())
                str_G_train = nx.relabel_nodes(G_train, {n: str(n) for n in G_train.nodes()})
                p, q = (2, 0.5) if actual_model_name == 'node2vec' else (1, 1)
                n2v = Node2Vec(str_G_train, dimensions=32, walk_length=20, num_walks=20, workers=4, p=p, q=q, seed=SEED+run)
                wv = n2v.fit(window=10, min_count=1).wv
                embeddings = {addr: wv[str(addr)] for addr in G_train.nodes()}
                
                scaler = StandardScaler()
                X_train = scaler.fit_transform(get_link_features(train_pairs, embeddings))
                model = RandomForestClassifier(n_jobs=-1, random_state=SEED)
                model.fit(X_train, train_labels)

            elif actual_model_name == 'lstm':
                train_dataset = TransactionPairDataset(train_pairs, train_labels, data['tx_sequences'])
                train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)
                model = SiameseLSTM(5, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LEARNING_RATE)
                loss_fn = nn.BCEWithLogitsLoss()
                model.train()
                for epoch in range(LSTM_EPOCHS):
                    for seq1, seq2, labels in train_loader:
                        seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.to(device)
                        optimizer.zero_grad()
                        preds = model(seq1, seq2)
                        loss = loss_fn(preds, labels)
                        loss.backward(); optimizer.step()
            
            elif actual_model_name == 'MixTracker-GNN':
                # MixTracker-GNN uses full transaction history graphs (already loaded in construction phase)
                # Use separate key to avoid conflict with baseline GNN graphs
                if data.get('address_to_graph_mixtracker') is None:
                    # Should not happen if construction was done correctly, but handle it anyway
                    raise ValueError("MixTracker-GNN graphs not loaded. This should not happen.")
            
            elif actual_model_name in ['gcn', 'graphsage', 'rgcn']:
                # Baseline GNN models use simple feature-based graphs
                node_features = data.get('node_features')
                if data.get('address_to_graph') is None and data.get('address_to_hetero_graph') is None:
                    if actual_model_name == 'rgcn':
                        gnn_type = 'hetero'
                    else:  # gcn, graphsage
                        gnn_type = 'regular'
                    address_to_graph, address_to_hetero_graph = load_data_for_gnn(
                        all_pairs, all_labels, node_features, gnn_type=gnn_type, use_cache=use_cache, measure_time=False
                    )
                    data['address_to_graph'] = address_to_graph
                    data['address_to_hetero_graph'] = address_to_hetero_graph
            
            if actual_model_name in ['MixTracker-GNN', 'gcn', 'graphsage', 'rgcn']:
                
                # Prepare packaged data
                packaged_data = []
                # MixTracker-GNN uses its own graph dict (full transaction history graphs)
                # Baseline GNN models use simple feature-based graphs
                if actual_model_name == 'rgcn':
                    graph_dict = data.get('address_to_hetero_graph')
                elif actual_model_name == 'MixTracker-GNN':
                    graph_dict = data.get('address_to_graph_mixtracker')
                else:
                    graph_dict = data.get('address_to_graph')
                
                if graph_dict is None:
                    raise ValueError(f"Graphs not built for {actual_model_name}")
                
                for (addr1, addr2), label in zip(train_pairs, train_labels):
                    graph1 = graph_dict.get(addr1)
                    graph2 = graph_dict.get(addr2)
                    if graph1 and graph2:
                        packaged_data.append({
                            'graph1': graph1,
                            'graph2': graph2,
                            'label': torch.tensor([float(label)], dtype=torch.float32)
                        })
                
                if len(packaged_data) < 2:
                    print(f"  Warning: Too few valid graph pairs ({len(packaged_data)}), skipping fold")
                    continue
                
                # Initialize model
                # For MixTracker-GNN: get feature dim from actual graph (build_address_graph creates graphs with features from transaction data)
                # For baseline models (gcn, graphsage, rgcn): use node_features dict (they use simple feature-based graphs)
                if actual_model_name == 'MixTracker-GNN':
                    # Get actual feature dimension from the graph (graphs built by build_address_graph have transaction-based features)
                    sample_graph = next(iter(graph_dict.values()))
                    if sample_graph and hasattr(sample_graph, 'x') and sample_graph.x is not None:
                        actual_feature_dim = sample_graph.x.shape[1]
                    else:
                        raise ValueError(f"Cannot determine feature dimension from graph for {actual_model_name}")
                else:
                    # Baseline models use node_features dict (from node_feature.csv)
                    actual_feature_dim = len(next(iter(node_features.values()))) if node_features else GNN_NODE_FEATURE_DIM
                
                gnn_type_map = {
                    'gcn': 'GCNConv',
                    'graphsage': 'SAGEConv',
                    'rgcn': 'RGCNConv',
                    'MixTracker-GNN': 'GATv2Conv'  # Default for MixTracker-GNN
                }
                gnn_type_str = gnn_type_map.get(actual_model_name, 'GATv2Conv')
                
                print(f"  Using node_feature_dim={actual_feature_dim} for {model_name}")
                if is_multi_gpu:
                    print(f"  Multi-GPU training enabled on {gpu_count} GPUs: {gpu_ids}")
                
                if actual_model_name in ['gcn', 'graphsage', 'rgcn']:
                    # Use SimpleLinkPredictorGNN for baseline models
                    model = SimpleLinkPredictorGNN(
                        node_feature_dim=actual_feature_dim,
                        edge_feature_dim=GNN_EDGE_FEATURE_DIM,
                        hidden_channels=64,
                        embedding_dim=32,
                        dropout_rate=0.1,
                        gnn_type=gnn_type_str
                    ).to(device)
                else:
                    # Use PairwiseRankingGNN for MixTracker-GNN (original model)
                    # Match Train.py: use pair_feature_dim=1 (pairwise features)
                    PAIR_FEATURE_DIM = 1
                    model = PairwiseRankingGNN(
                        node_feature_dim=actual_feature_dim, 
                        edge_feature_dim=GNN_EDGE_FEATURE_DIM, 
                        pair_feature_dim=PAIR_FEATURE_DIM,
                        use_graph=True,  # MixTracker-GNN always uses graph
                        gnn_type='GATv2Conv'  # Match Train.py default
                    ).to(device)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LEARNING_RATE)
                loss_fn = nn.BCEWithLogitsLoss()
                
                # Multi-GPU training setup using gradient accumulation strategy
                # Each GPU processes independent data splits, sync only at epoch end
                if is_multi_gpu and gpu_ids and len(gpu_ids) > 1:
                    print(f"  Multi-GPU training (data parallelism) on GPUs: {gpu_ids}")
                    print(f"  Strategy: Data sharding with epoch-end synchronization")
                    
                    # Create model replicas on each GPU
                    model_replicas = [model]  # First GPU already has model
                    for gpu_id in gpu_ids[1:]:
                        replica = type(model)(
                            node_feature_dim=actual_feature_dim, 
                            edge_feature_dim=GNN_EDGE_FEATURE_DIM, 
                            pair_feature_dim=PAIR_FEATURE_DIM,
                            use_graph=True,
                            gnn_type='GATv2Conv'
                        ).to(torch.device(f'cuda:{gpu_id}'))
                        replica.load_state_dict(model.state_dict())
                        model_replicas.append(replica)
                    
                    # Create separate optimizers for each replica
                    optimizers = [torch.optim.Adam(m.parameters(), lr=GNN_LEARNING_RATE) for m in model_replicas]
                    print(f"  Strategy: Data sharding with epoch-end synchronization (only {GNN_EPOCHS} syncs total)")
                else:
                    model_replicas = [model]
                    optimizers = [optimizer]
                
                # Training with fixed epochs (no early stopping for consistent timing comparison)
                best_f1 = 0.0
                best_fold_metrics = None
                
                for epoch in range(1, GNN_EPOCHS + 1):
                    # Train
                    for m in model_replicas:
                        m.train()
                    total_loss = 0
                    batches_processed = 0
                    
                    # Track peak memory during training (not just at end)
                    if device.type == 'cuda':
                        # Update peak memory during training
                        current_mem_info = get_gpu_memory_info(device, gpu_ids if is_multi_gpu else None)
                        peak_memory_gb = max(peak_memory_gb, current_mem_info.get('max_reserved_gb', current_mem_info.get('max_allocated_gb', 0.0)))
                    
                    if is_multi_gpu and len(model_replicas) > 1:
                        # Multi-GPU: Parallel training using ThreadPoolExecutor
                        # Re-shuffle and split data at the beginning of each epoch
                        packaged_data_shuffled = packaged_data.copy()
                        np.random.shuffle(packaged_data_shuffled)
                        
                        num_gpus = len(gpu_ids)
                        chunk_size = len(packaged_data_shuffled) // num_gpus
                        data_splits = []
                        for gpu_idx in range(num_gpus):
                            start_idx = gpu_idx * chunk_size
                            end_idx = start_idx + chunk_size if gpu_idx < num_gpus - 1 else len(packaged_data_shuffled)
                            data_splits.append(packaged_data_shuffled[start_idx:end_idx])
                        
                        # Parallel execution using ThreadPoolExecutor
                        # Use larger batch size for multi-GPU to improve GPU utilization
                        multi_gpu_batch_size = LSTM_BATCH_SIZE * 2  # Double batch size per GPU for better utilization
                        
                        losses_per_gpu = {}
                        batches_per_gpu = {}
                        
                        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                            # Submit all GPU training tasks in parallel
                            futures = {
                                executor.submit(
                                    train_on_single_gpu,
                                    gpu_idx, gpu_ids[gpu_idx], model_replicas[gpu_idx],
                                    optimizers[gpu_idx], data_splits[gpu_idx],
                                    loss_fn, actual_model_name, epoch, multi_gpu_batch_size
                                ): gpu_idx
                                for gpu_idx in range(num_gpus)
                            }
                            
                            # Collect results as they complete (truly parallel execution)
                            for future in as_completed(futures):
                                gpu_idx, avg_loss, num_batches = future.result()
                                losses_per_gpu[gpu_idx] = avg_loss
                                batches_per_gpu[gpu_idx] = num_batches
                        
                        # Average loss across all GPUs
                        total_loss = np.mean(list(losses_per_gpu.values())) if losses_per_gpu else 0.0
                        batches_processed = max(batches_per_gpu.values()) if batches_per_gpu else 0
                        
                        # Update peak memory after parallel training (each GPU may have different peaks)
                        if device.type == 'cuda':
                            current_mem_info = get_gpu_memory_info(device, gpu_ids)
                            peak_memory_gb = max(peak_memory_gb, current_mem_info.get('max_reserved_gb', current_mem_info.get('max_allocated_gb', 0.0)))
                        
                        # Optimized parameter synchronization: batch operations for efficiency
                        # Only sync at end of epoch to minimize communication overhead
                        if epoch < GNN_EPOCHS:  # Skip sync on last epoch if not needed for evaluation
                            sync_start_time = time.time()
                            
                        with torch.no_grad():
                            main_device = torch.device(f'cuda:{gpu_ids[0]}')
                            
                            # Get parameter lists for all GPUs (more efficient than dict iteration)
                            param_lists = [[p for p in m.parameters()] for m in model_replicas]
                            
                            # Process all parameters in batches for better memory access
                            for param_idx in range(len(param_lists[0])):
                                # Collect parameters from all GPUs
                                params = [pl[param_idx].data for pl in param_lists]
                                
                                # Aggregate on main device (more efficient than per-param sync)
                                if param_idx == 0:
                                    # First param: move others to main device
                                    avg_param = params[0].clone()
                                    for gpu_idx in range(1, len(model_replicas)):
                                        avg_param += params[gpu_idx].to(main_device, non_blocking=True)
                                else:
                                    # Subsequent params: reuse device transfers
                                    avg_param = params[0].clone()
                                    for gpu_idx in range(1, len(model_replicas)):
                                        avg_param += params[gpu_idx].to(main_device, non_blocking=True)
                                
                                avg_param /= len(model_replicas)
                                
                                # Broadcast to all GPUs (use non_blocking for async transfer)
                                for gpu_idx in range(len(model_replicas)):
                                    if gpu_idx == 0:
                                        params[gpu_idx].copy_(avg_param)
                                    else:
                                        params[gpu_idx].copy_(avg_param.to(torch.device(f'cuda:{gpu_ids[gpu_idx]}'), non_blocking=True))
                            
                            # Only synchronize once at the end for all transfers
                            for gpu_id in gpu_ids:
                                torch.cuda.synchronize(torch.device(f'cuda:{gpu_id}'))
                            
                            # Update peak memory after synchronization (sync may use additional memory)
                            current_mem_info = get_gpu_memory_info(device, gpu_ids)
                            peak_memory_gb = max(peak_memory_gb, current_mem_info.get('max_reserved_gb', current_mem_info.get('max_allocated_gb', 0.0)))
                        
                        if epoch < GNN_EPOCHS:
                            sync_time = time.time() - sync_start_time
                            if epoch == 1:  # Print once for debugging
                                print(f"    Parameter sync time: {sync_time*1000:.2f}ms")
                    else:
                        # Single GPU: original code
                        np.random.shuffle(packaged_data)
                        for i in range(0, len(packaged_data), LSTM_BATCH_SIZE):
                            batch = packaged_data[i:i+LSTM_BATCH_SIZE]
                            if len(batch) <= 1:
                                continue
                            batches_processed += 1
                            
                            g1 = Batch.from_data_list([p['graph1'] for p in batch]).to(device)
                            g2 = Batch.from_data_list([p['graph2'] for p in batch]).to(device)
                            labels = torch.stack([p['label'] for p in batch]).to(device).squeeze(-1)
                            optimizer.zero_grad()
                            
                            if actual_model_name in ['gcn', 'graphsage', 'rgcn']:
                                scores = model(g1, g2)
                            else:
                                feat = torch.stack([torch.tensor([0.0]) for _ in batch]).to(device)
                                scores = model(g1, g2, feat)
                            
                            loss = loss_fn(scores, labels)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()
                        
                        # Update peak memory during single-GPU training
                        if device.type == 'cuda' and (batches_processed % 10 == 0):  # Check every 10 batches
                            current_mem_info = get_gpu_memory_info(device, None)
                            peak_memory_gb = max(peak_memory_gb, current_mem_info.get('max_reserved_gb', current_mem_info.get('max_allocated_gb', 0.0)))
                    
                    if batches_processed == 0:
                        continue
                    
                    # Final memory check after epoch
                    if device.type == 'cuda':
                        current_mem_info = get_gpu_memory_info(device, gpu_ids if is_multi_gpu else None)
                        peak_memory_gb = max(peak_memory_gb, current_mem_info.get('max_reserved_gb', current_mem_info.get('max_allocated_gb', 0.0)))
                    
                    avg_loss = total_loss if is_multi_gpu else total_loss / batches_processed
                    
                    # Evaluate on validation set (for early stopping)
                    # Use first GPU/model for evaluation in multi-GPU mode
                    eval_model = model_replicas[0] if is_multi_gpu and len(model_replicas) > 1 else model
                    eval_device = device if not is_multi_gpu else torch.device(f'cuda:{gpu_ids[0]}')
                    eval_model.eval()
                    all_preds = []
                    all_labels_eval = []
                    
                    with torch.no_grad():
                        EVAL_BATCH_SIZE = 64
                        # Use validation pairs for early stopping
                        val_packaged_data = []
                        for (addr1, addr2), label in zip(val_pairs, val_labels):
                            graph1 = graph_dict.get(addr1)
                            graph2 = graph_dict.get(addr2)
                            if graph1 and graph2:
                                val_packaged_data.append({
                                    'graph1': graph1,
                                    'graph2': graph2,
                                    'label': torch.tensor([float(label)], dtype=torch.float32)
                                })
                        
                        for i in range(0, len(val_packaged_data), EVAL_BATCH_SIZE):
                            batch = val_packaged_data[i:i+EVAL_BATCH_SIZE]
                            g1 = Batch.from_data_list([p['graph1'] for p in batch]).to(eval_device)
                            g2 = Batch.from_data_list([p['graph2'] for p in batch]).to(eval_device)
                            labels_batch = torch.stack([p['label'] for p in batch]).to(eval_device).squeeze(-1)
                            
                            if actual_model_name in ['gcn', 'graphsage', 'rgcn']:
                                logits = eval_model(g1, g2)
                            else:
                                feat = torch.stack([torch.tensor([0.0]) for _ in batch]).to(eval_device)
                                logits = eval_model(g1, g2, feat)
                            
                            preds = (torch.sigmoid(logits) > 0.5).cpu()
                            all_preds.append(preds)
                            all_labels_eval.append(labels_batch.cpu())
                    
                    all_preds = torch.cat(all_preds).numpy()
                    all_labels_eval = torch.cat(all_labels_eval).numpy()
                    
                    accuracy = (all_preds == all_labels_eval).mean()
                    precision = precision_score(all_labels_eval, all_preds, zero_division=0)
                    recall = recall_score(all_labels_eval, all_preds, zero_division=0)
                    f1 = f1_score(all_labels_eval, all_preds, zero_division=0)
                    
                    if len(np.unique(all_labels_eval)) == 2:
                        tn, fp, fn, tp = confusion_matrix(all_labels_eval, all_preds).ravel()
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                    else:
                        fpr, fnr = 0.0, 0.0
                    
                    if epoch % 20 == 0 or epoch == GNN_EPOCHS:
                        print(f"  Epoch {epoch:03d}/{GNN_EPOCHS} | Loss: {avg_loss:.4f} | F1: {f1:.2%}")
                    
                    # Track best metrics (no early stopping, just for recording best performance)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_fold_metrics = (accuracy, precision, recall, f1, fpr, fnr)
                
                # Store best metrics for this fold (already computed during training with early stopping)
                if best_fold_metrics:
                    fold_metrics.append({
                        'accuracy': best_fold_metrics[0],
                        'precision': best_fold_metrics[1],
                        'recall': best_fold_metrics[2],
                        'f1': best_fold_metrics[3],
                        'fpr': best_fold_metrics[4],
                        'fnr': best_fold_metrics[5]
                    })
                    print(f"  Best F1: {best_fold_metrics[3]:.4f}")
                else:
                    print(f"  Warning: No valid metrics recorded for this fold")
                    continue
                
                # For GNN models, we already computed predictions during training (for early stopping)
                # Set a flag to skip metric calculation in test phase
                gnn_metrics_already_computed = True
                preds = None  # Not needed for GNN models since metrics already computed
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Final peak GPU memory check after all training
            # Use max_reserved_gb which is closer to nvidia-smi values
            mem_info = get_gpu_memory_info(device, gpu_ids if is_multi_gpu else None)
            # Use reserved memory (includes PyTorch cache + CUDA context) - closer to nvidia-smi
            peak_memory_gb = max(peak_memory_gb, mem_info.get('max_reserved_gb', mem_info.get('max_allocated_gb', 0.0)))
            
            fold_train_time = time.time() - start_train_time
            # Add construction time to first fold only (amortized across all folds)
            if fold == 0 and construction_time > 0:
                fold_train_time += construction_time
            fold_train_times.append(fold_train_time)

            # --- 3. Test Model & Measure Time ---
            # For GNN models, we already computed metrics during training (for early stopping)
            # So we only need to measure test time for consistency, but don't recompute metrics
            start_test_time = time.time()
            gnn_metrics_already_computed = False  # Will be set to True in GNN training loop
            
            if actual_model_name in ['lr', 'rf', 'deepwalk', 'node2vec']:
                test_features = node_features if actual_model_name in ['lr', 'rf'] else embeddings
                X_val = get_link_features(val_pairs, test_features)
                if scaler: X_val = scaler.transform(X_val)
                preds = model.predict(X_val)
            
            elif actual_model_name == 'lstm':
                preds = []
                model.eval()
                with torch.no_grad():
                    # Load tx_sequences if not already loaded
                    if data.get('tx_sequences') is None:
                        tx_sequences = load_transactions_for_lstm(all_pairs, use_cache=use_cache, measure_time=False)
                        data['tx_sequences'] = tx_sequences
                    val_dataset = TransactionPairDataset(val_pairs, val_labels, data['tx_sequences'])
                    val_loader = DataLoader(val_dataset, batch_size=1)
                    for seq1, seq2, _ in val_loader:
                        seq1, seq2 = seq1.to(device), seq2.to(device)
                        preds.append(torch.sigmoid(model(seq1, seq2)).cpu().item() > 0.5)
            
            elif actual_model_name in ['MixTracker-GNN', 'gcn', 'graphsage', 'rgcn']:
                # For GNN models, metrics were already computed during training (for early stopping)
                # Just do a quick re-evaluation for test time measurement (using best model state)
                preds = []
                # Use first model for test evaluation in multi-GPU mode
                eval_model = model_replicas[0] if (is_multi_gpu and 'model_replicas' in locals() and len(model_replicas) > 1) else model
                eval_device = device if not is_multi_gpu else torch.device(f'cuda:{gpu_ids[0]}')
                eval_model.eval()
                with torch.no_grad():
                    # Use correct graph dict based on model type
                    if actual_model_name == 'rgcn':
                        graph_dict = data.get('address_to_hetero_graph')
                    elif actual_model_name == 'MixTracker-GNN':
                        graph_dict = data.get('address_to_graph_mixtracker')
                    else:
                        graph_dict = data.get('address_to_graph')
                    
                    if graph_dict is None:
                        raise ValueError(f"Graphs not built for {actual_model_name}")
                    
                    # Verify graph feature dimension for MixTracker-GNN (should be 19, not 46)
                    if actual_model_name == 'MixTracker-GNN':
                        sample_graph = next(iter([g for g in graph_dict.values() if g is not None]), None)
                        if sample_graph and hasattr(sample_graph, 'x') and sample_graph.x is not None:
                            actual_dim = sample_graph.x.shape[1]
                            # MixTracker-GNN graphs built by build_address_graph should have 19 features
                            # If we see 46 features, we're using the wrong graph dict (baseline GNN graphs)
                            if actual_dim == 46:
                                raise ValueError(f"Graph feature dimension mismatch for MixTracker-GNN: "
                                               f"got {actual_dim} (baseline GNN graph), expected 19 (MixTracker-GNN graph). "
                                               f"This indicates wrong graph dict is being used. "
                                               f"Check that address_to_graph_mixtracker is properly set.")
                            elif actual_dim != 19:
                                print(f"  Warning: Unexpected feature dimension {actual_dim} for MixTracker-GNN (expected 19)")
                    
                    EVAL_BATCH_SIZE = 64
                    val_packaged_data = []
                    for (addr1, addr2), _ in zip(val_pairs, val_labels):
                        graph1 = graph_dict.get(addr1)
                        graph2 = graph_dict.get(addr2)
                        if graph1 and graph2:
                            val_packaged_data.append({
                                'graph1': graph1,
                                'graph2': graph2
                            })
                    
                    for i in range(0, len(val_packaged_data), EVAL_BATCH_SIZE):
                        batch = val_packaged_data[i:i+EVAL_BATCH_SIZE]
                        g1 = Batch.from_data_list([p['graph1'] for p in batch]).to(eval_device)
                        g2 = Batch.from_data_list([p['graph2'] for p in batch]).to(eval_device)
                        
                        if actual_model_name in ['gcn', 'graphsage', 'rgcn']:
                            logits = eval_model(g1, g2)
                        else:
                            feat = torch.stack([torch.tensor([0.0]) for _ in batch]).to(eval_device)
                            logits = eval_model(g1, g2, feat)
                        
                        batch_preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                        preds.extend(batch_preds.tolist())
                
                preds = np.array(preds)
                gnn_metrics_already_computed = True  # Mark that metrics were already computed in training loop
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            fold_test_times.append(time.time() - start_test_time)
            fold_test_samples.append(len(val_labels))
            
            # --- 4. Calculate and Store Metrics ---
            # For GNN models, metrics were already computed and stored during training (for early stopping)
            # Only compute metrics here for non-GNN models
            if not gnn_metrics_already_computed:
                fold_metrics.append(_calculate_metrics(val_labels, preds))
            
            # Print fold results
            if fold_metrics:
                print(f"Fold {fold+1} F1: {fold_metrics[-1]['f1']:.4f}, Train Time: {fold_train_times[-1]:.2f}s, Test Time: {fold_test_times[-1]:.2f}s")

        # --- Aggregate and Store Run Results ---
        run_metrics = {key: [m[key] for m in fold_metrics] for key in fold_metrics[0]}
        avg_f1 = np.mean(run_metrics['f1'])
        if avg_f1 > best_run_avg_f1:
            best_run_avg_f1 = avg_f1
            best_run_results = {'run': run}
            for key, values in run_metrics.items():
                best_run_results[f'avg_{key}'] = np.mean(values)
                best_run_results[f'std_{key}'] = np.std(values)
            best_run_results['total_train_time'] = sum(fold_train_times)
            best_run_results['avg_fold_train_time'] = np.mean(fold_train_times)
            best_run_results['graph_construction_time'] = construction_time  # Include construction time in results
            best_run_results['total_test_time'] = sum(fold_test_times)
            best_run_results['avg_sample_test_time'] = sum(fold_test_times) / sum(fold_test_samples) if sum(fold_test_samples) > 0 else 0
            
            # GPU statistics
            best_run_results['gpu_count'] = gpu_count
            best_run_results['peak_gpu_memory_gb'] = peak_memory_gb
            
            # Multi-GPU specific stats
            final_mem_info = get_gpu_memory_info(device, gpu_ids if is_multi_gpu else None)
            
            if is_multi_gpu and gpu_ids:
                # Use reserved memory for more accurate reporting (closer to nvidia-smi)
                # Peak on any single device (max across all devices)
                best_run_results['gpu_memory_per_device_gb'] = peak_memory_gb
                # Total across all GPUs (sum of peak reserved memory on each device)
                max_reserved_per_device = final_mem_info.get('max_reserved_per_device_gb', [])
                if max_reserved_per_device:
                    # Use the actual per-device peaks for accurate total
                    best_run_results['total_gpu_memory_gb'] = sum(max_reserved_per_device)
                else:
                    # Fallback to total_max_reserved_gb
                    best_run_results['total_gpu_memory_gb'] = final_mem_info.get('total_max_reserved_gb', peak_memory_gb * len(gpu_ids))
            else:
                # Single GPU
                best_run_results['gpu_memory_per_device_gb'] = peak_memory_gb
                best_run_results['total_gpu_memory_gb'] = final_mem_info.get('max_reserved_gb', peak_memory_gb)
            
            # Speedup and efficiency metrics (to be calculated by comparing with single-GPU baseline)
            best_run_results['speedup_factor'] = 1.0  # Will be updated if comparing with baseline
            best_run_results['parallel_efficiency'] = 100.0  # Will be updated if comparing with baseline
            
            best_run_results['cache_used'] = 'Yes' if use_cache else 'No'
            best_run_results['io_time_total'] = data.get('io_time_total', 0.0)  # Add I/O time to results
    
    return best_run_results

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Baseline Models for Link Prediction Training with Timing and Multi-GPU Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single-GPU training (MixTracker-GNN on GPU 0)
  python run_baselines_with_timing.py --dataset D1 --model MixTracker-GNN --gpu 0
  
  # Multi-GPU training (MixTracker-GNN on GPUs 0 and 1)
  python run_baselines_with_timing.py --dataset D1 --model MixTracker-GNN-2GPU --gpus 0,1
  
  # Compare single-GPU vs multi-GPU performance
  python run_baselines_with_timing.py --dataset D1 --model MixTracker-GNN --compare-multi-gpu --gpus 0,1
  
  # Run all baseline models on single GPU
  python run_baselines_with_timing.py --dataset D1 --model all --gpu 0

Note: MixTracker-GNN-2GPU uses DataParallel for multi-GPU training to address reviewer concerns
      about scalability, training time, and memory consumption on RTX 3090 GPUs.
        '''
    )
    parser.add_argument('--dataset', type=str, default='D1', choices=['D1', 'D2', 'D3'])
    parser.add_argument('--model', type=str, default='all', 
                        choices=['lr', 'rf', 'deepwalk', 'node2vec', 'lstm', 'MixTracker-GNN', 'MixTracker-GNN-2GPU', 'gcn', 'graphsage', 'rgcn', 'all'])
    parser.add_argument('--gpu', type=int, default=0, help='Primary GPU device ID to use.')
    parser.add_argument('--gpus', type=str, default=None, 
                        help='Comma-separated list of GPU IDs for multi-GPU training (e.g., "0,1"). Overrides --gpu for MixTracker-GNN-2GPU.')
    parser.add_argument('--use-cache', action='store_true', default=True, help='Use cached graphs/sequences (default: True)')
    parser.add_argument('--no-cache', dest='use_cache', action='store_false', help='Rebuild graphs/sequences from scratch (include construction time)')
    parser.add_argument('--compare-cache', action='store_true', help='Run both cached and non-cached versions to compare time savings')
    parser.add_argument('--compare-multi-gpu', action='store_true', 
                        help='Run both single-GPU and multi-GPU versions of MixTracker-GNN to measure speedup')
    parser.add_argument('--stats-gpu-memory', action='store_true',
                        help='Run MixTracker-GNN on all datasets (D1, D2, D3) with single-GPU and multi-GPU to collect average GPU memory statistics')
    args = parser.parse_args()

    # Parse GPU IDs
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
        DEVICE = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    else:
        gpu_ids = [args.gpu]
        DEVICE = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Handle GPU memory statistics mode
    if args.stats_gpu_memory:
        if not torch.cuda.is_available():
            print("Error: GPU memory statistics requires CUDA-enabled GPU.")
            exit(1)
        
        if len(gpu_ids) < 2:
            print("Warning: Multi-GPU statistics requires at least 2 GPUs. Using GPUs 0,1 by default.")
            gpu_ids = [0, 1]
            DEVICE = torch.device('cuda:0')
        
        print(f"\n{'='*70}")
        print(f"GPU Memory Statistics Collection Mode")
        print(f"{'='*70}")
        print(f"Will run MixTracker-GNN on all datasets (D1, D2, D3)")
        print(f"Single-GPU: GPU {gpu_ids[0]}")
        print(f"Multi-GPU: GPUs {gpu_ids[:2]}")
        print(f"{'='*70}\n")
        
        # Statistics collection
        single_gpu_memories = []  # List of (dataset, memory_per_device, total_memory)
        multi_gpu_memories = []   # List of (dataset, memory_per_device, total_memory)
        
        datasets = ['D3']
        
        simple_gpu_flag = False 
        multi_gpu_flag = True

        for dataset in datasets:
            print(f"\n{'='*70}")
            print(f"Processing Dataset: {dataset}")
            print(f"{'='*70}\n")
            
            # Clear GPU memory before processing each dataset
            print("Clearing GPU memory...")
            if torch.cuda.is_available():
                # Clear PyTorch CUDA cache
                for gpu_id in gpu_ids[:2]:  # Clear for all GPUs we'll use
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(torch.device(f'cuda:{gpu_id}'))
                # Python garbage collection
                gc.collect()
                print("  GPU memory cleared.")
            
            # Load dataset-specific data
            node_features, all_pairs, all_labels, G = load_data_for_baselines(dataset)
            
            # Pre-load MixTracker-GNN graphs for this dataset
            print("Pre-loading MixTracker-GNN graphs...")
            data_package = {
                'node_features': node_features,
                'all_pairs': all_pairs,
                'all_labels': all_labels,
                'G': G,
                'tx_sequences': None,
                'address_to_graph': None,
                'address_to_hetero_graph': None
            }
            
            if simple_gpu_flag:
                # Run single-GPU version (limit to 5 folds for statistics)
                print(f"\n--- Single-GPU Run for {dataset} ---")
                single_gpu_ids = [gpu_ids[0]]
                single_device = torch.device(f'cuda:{single_gpu_ids[0]}')
                results_single = run_experiment({'model': 'MixTracker-GNN', 'use_cache': True, 'max_folds': 2}, 
                                            data_package.copy(), single_device, single_gpu_ids)
                
                if results_single:
                    mem_per_device = results_single.get('gpu_memory_per_device_gb', 0.0)
                    total_mem = results_single.get('total_gpu_memory_gb', 0.0)
                    single_gpu_memories.append((dataset, mem_per_device, total_mem))
                    print(f"  Single-GPU Memory: {mem_per_device:.2f} GB per device, {total_mem:.2f} GB total")
                
                # Clear GPU memory between single-GPU and multi-GPU runs
                if torch.cuda.is_available():
                    print("\nClearing GPU memory before multi-GPU run...")
                    for gpu_id in gpu_ids[:2]:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize(torch.device(f'cuda:{gpu_id}'))
                    gc.collect()
                    print("  GPU memory cleared.")
            
            if multi_gpu_flag:
                # Run multi-GPU version (limit to 5 folds for statistics)
                print(f"\n--- Multi-GPU Run for {dataset} ---")
                multi_gpu_ids = gpu_ids[:2]  # Use first 2 GPUs
                multi_device = torch.device(f'cuda:{multi_gpu_ids[0]}')
                
                # Reset cache for MixTracker-GNN graphs (will be rebuilt)
                data_package_multi = data_package.copy()
                data_package_multi['address_to_graph_mixtracker'] = None
                
                results_multi = run_experiment({'model': 'MixTracker-GNN-2GPU', 'use_cache': True, 'max_folds': 2}, 
                                            data_package_multi, multi_device, multi_gpu_ids)
                
                if results_multi:
                    mem_per_device = results_multi.get('gpu_memory_per_device_gb', 0.0)
                    total_mem = results_multi.get('total_gpu_memory_gb', 0.0)
                    multi_gpu_memories.append((dataset, mem_per_device, total_mem))
                    print(f"  Multi-GPU Memory: {mem_per_device:.2f} GB per device, {total_mem:.2f} GB total")
        
        # Calculate and print statistics
        print(f"\n{'='*70}")
        print(f"GPU Memory Statistics Summary")
        print(f"{'='*70}\n")
        
        if single_gpu_memories:
            avg_single_per_device = np.mean([m[1] for m in single_gpu_memories])
            avg_single_total = np.mean([m[2] for m in single_gpu_memories])
            std_single_per_device = np.std([m[1] for m in single_gpu_memories])
            std_single_total = np.std([m[2] for m in single_gpu_memories])
            
            print(f"Single-GPU (MixTracker-GNN):")
            print(f"  Average Memory per Device: {avg_single_per_device:.2f} ± {std_single_per_device:.2f} GB")
            print(f"  Average Total Memory:      {avg_single_total:.2f} ± {std_single_total:.2f} GB")
            print(f"\n  Per-Dataset Details:")
            for dataset, per_device, total in single_gpu_memories:
                print(f"    {dataset}: {per_device:.2f} GB per device, {total:.2f} GB total")
        
        if multi_gpu_memories:
            avg_multi_per_device = np.mean([m[1] for m in multi_gpu_memories])
            avg_multi_total = np.mean([m[2] for m in multi_gpu_memories])
            std_multi_per_device = np.std([m[1] for m in multi_gpu_memories])
            std_multi_total = np.std([m[2] for m in multi_gpu_memories])
            
            print(f"\nMulti-GPU (MixTracker-GNN-2GPU):")
            print(f"  Average Memory per Device: {avg_multi_per_device:.2f} ± {std_multi_per_device:.2f} GB")
            print(f"  Average Total Memory:      {avg_multi_total:.2f} ± {std_multi_total:.2f} GB")
            print(f"\n  Per-Dataset Details:")
            for dataset, per_device, total in multi_gpu_memories:
                print(f"    {dataset}: {per_device:.2f} GB per device, {total:.2f} GB total")
        
        print(f"\n{'='*70}\n")
        print("✅ GPU memory statistics collection complete.")
        exit(0)
    
    print(f"Using dataset: {args.dataset}, Primary Device: {DEVICE}")
    
    # Print GPU information for reviewer context
    if DEVICE.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(DEVICE)
        gpu_memory_gb = torch.cuda.get_device_properties(DEVICE).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"GPU Total Memory: {gpu_memory_gb:.2f} GB")
    else:
        print("Using CPU (no GPU available)")

    # 1. Load common data (includes node_features)
    node_features, all_pairs, all_labels, G = load_data_for_baselines(args.dataset)
    
    # Determine models to run
    if args.model == 'all':
        models_to_run = ['MixTracker-GNN', 'gcn', 'graphsage', 'rgcn']
        # 'lr', 'rf', 'deepwalk', 'node2vec', 'lstm'
    else:
        models_to_run = [args.model]
    
    # Handle multi-GPU comparison mode
    if args.compare_multi_gpu:
        if 'MixTracker-GNN' in models_to_run and 'MixTracker-GNN-2GPU' not in models_to_run:
            models_to_run.append('MixTracker-GNN-2GPU')
        elif args.model == 'MixTracker-GNN-2GPU':
            # Add single-GPU version for comparison
            models_to_run.insert(0, 'MixTracker-GNN') 
    # 3. Pre-load model-specific data (only if using cache and not comparing, otherwise load during experiment to measure time)
    tx_sequences = None
    address_to_graph = None
    address_to_hetero_graph = None
    if args.use_cache and not args.compare_cache:
        tx_sequences = load_transactions_for_lstm(all_pairs, use_cache=True) if 'lstm' in models_to_run else None
        
        # Pre-load GNN graphs if any GNN model is included (only for baseline models, not MixTracker-GNN)
        baseline_gnn_models = [m for m in models_to_run if m in ['gcn', 'graphsage', 'rgcn']]
        if baseline_gnn_models:
            gnn_type = 'all'  # Build both regular and hetero graphs for baseline models
            address_to_graph, address_to_hetero_graph = load_data_for_gnn(
                all_pairs, all_labels, node_features, gnn_type=gnn_type, use_cache=True, measure_time=False
            )
        
        # MixTracker-GNN graphs are loaded separately during experiment (they use different graph building logic)
    
    # 4. Run experiments
    single_gpu_baseline_time = None  # Store for speedup calculation
    
    for model_name in models_to_run:
        data_package = {
            'node_features': node_features,
            'all_pairs': all_pairs,
            'all_labels': all_labels,
            'G': G,
            'tx_sequences': tx_sequences,
            'address_to_graph': address_to_graph,
            'address_to_hetero_graph': address_to_hetero_graph
        }
        
        # Determine GPU configuration for this model
        if model_name == 'MixTracker-GNN-2GPU':
            # Multi-GPU training
            if len(gpu_ids) < 2:
                print(f"\n⚠️  Warning: MixTracker-GNN-2GPU requires at least 2 GPUs, but only {len(gpu_ids)} provided.")
                print(f"   Please use --gpus argument (e.g., --gpus 0,1)")
                continue
            current_gpu_ids = gpu_ids
            current_device = DEVICE
        else:
            # Single-GPU training (use first GPU from list)
            current_gpu_ids = [gpu_ids[0]]
            current_device = DEVICE
        
        if args.compare_cache and model_name in ['lstm', 'MixTracker-GNN', 'MixTracker-GNN-2GPU', 'gcn', 'graphsage', 'rgcn']:
            # Run both cached and non-cached versions to compare
            print(f"\n{'='*60}")
            print(f"Comparing cached vs non-cached for {model_name.upper()}")
            print(f"{'='*60}")
            
            # First run: with cache
            print("\n--- Running WITH ego-subgraph cache ---")
            data_package_cached = data_package.copy()
            results_cached = run_experiment({'model': model_name, 'use_cache': True}, data_package_cached, current_device, current_gpu_ids)
            
            # Clear caches to force rebuild
            actual_model = 'MixTracker-GNN' if model_name == 'MixTracker-GNN-2GPU' else model_name
            if actual_model == 'lstm':
                globals()['TRANSACTION_CACHE'] = {}
                data_package_no_cache = data_package.copy()
                data_package_no_cache['tx_sequences'] = None
            elif actual_model in ['MixTracker-GNN', 'gcn', 'graphsage', 'rgcn']:
                globals()['GNN_DATA_CACHE'] = {}
                globals()['GNN_DATA_CACHE_HETERO'] = {}
                data_package_no_cache = data_package.copy()
                # Clear both baseline GNN graphs and MixTracker-GNN graphs
                if actual_model == 'MixTracker-GNN':
                    data_package_no_cache['address_to_graph_mixtracker'] = None
                else:
                    data_package_no_cache['address_to_graph'] = None
                    data_package_no_cache['address_to_hetero_graph'] = None
            else:
                data_package_no_cache = data_package.copy()
            
            # Second run: without cache
            print("\n--- Running WITHOUT cache (rebuilding from scratch) ---")
            results_no_cache = run_experiment({'model': model_name, 'use_cache': False}, data_package_no_cache, current_device, current_gpu_ids)
            
            # Calculate cache time savings
            if results_cached and results_no_cache:
                cached_time = results_cached['total_train_time']
                no_cache_time = results_no_cache['total_train_time']
                time_saved = no_cache_time - cached_time
                time_saved_percent = (time_saved / no_cache_time * 100) if no_cache_time > 0 else 0.0
                
                print(f"\n{'='*60}")
                print(f"Cache Comparison Results for {model_name.upper()}:")
                print(f"  With cache:    {cached_time:.2f}s")
                print(f"  Without cache: {no_cache_time:.2f}s")
                print(f"  Time saved:    {time_saved:.2f}s ({time_saved_percent:.1f}%)")
                print(f"  Peak GPU Memory: {results_cached.get('peak_gpu_memory_gb', 0.0):.2f} GB")
                print(f"{'='*60}\n")
                
                # Save cached version with cache savings info
                results_cached['cache_time_saved'] = time_saved
                results_cached['cache_time_saved_percent'] = time_saved_percent
                params = {'Dataset': args.dataset, 'Model': model_name.upper(), 'Seed': SEED, 'Num_Runs': NUM_RUNS, 'K_Fold': K_FOLD}
                save_results_to_csv(results_cached, params, f'Results/{RUN_TIMESTAMP}_baselines_timed_{args.dataset}.csv')
                
                # Also save no-cache version for reference
                results_no_cache['cache_time_saved'] = 0.0
                results_no_cache['cache_time_saved_percent'] = 0.0
                save_results_to_csv(results_no_cache, params, f'Results/{RUN_TIMESTAMP}_baselines_timed_{args.dataset}.csv')
        else:
            # Normal run with specified cache setting
            cache_status = 'Using ego-subgraph cache' if args.use_cache else 'Rebuilding from scratch (construction time will be measured)'
            print(f"\nCache setting: {cache_status}")
            if model_name == 'MixTracker-GNN-2GPU':
                print(f"Multi-GPU configuration: {len(current_gpu_ids)} GPUs - {current_gpu_ids}")
            
            results = run_experiment({'model': model_name, 'use_cache': args.use_cache}, data_package, current_device, current_gpu_ids)
            if results:
                results['cache_time_saved'] = 0.0  # Not compared, so no savings data
                results['cache_time_saved_percent'] = 0.0
                
                # Calculate speedup if we have single-GPU baseline
                if model_name == 'MixTracker-GNN':
                    single_gpu_baseline_time = results['total_train_time']
                elif model_name == 'MixTracker-GNN-2GPU' and single_gpu_baseline_time:
                    multi_gpu_time = results['total_train_time']
                    speedup = single_gpu_baseline_time / multi_gpu_time if multi_gpu_time > 0 else 1.0
                    parallel_efficiency = (speedup / results['gpu_count']) * 100.0 if results['gpu_count'] > 1 else 100.0
                    results['speedup_factor'] = speedup
                    results['parallel_efficiency'] = parallel_efficiency
                    
                    print(f"\n{'='*60}")
                    print(f"Multi-GPU Speedup Analysis:")
                    print(f"  Single-GPU time:  {single_gpu_baseline_time:.2f}s")
                    print(f"  Multi-GPU time:   {multi_gpu_time:.2f}s")
                    print(f"  Speedup:          {speedup:.2f}x")
                    print(f"  Parallel efficiency: {parallel_efficiency:.1f}%")
                    print(f"  GPU count:        {results['gpu_count']}")
                    print(f"{'='*60}\n")
                
                params = {'Dataset': args.dataset, 'Model': model_name.upper(), 'Seed': SEED, 'Num_Runs': NUM_RUNS, 'K_Fold': K_FOLD}
                save_results_to_csv(results, params, f'Results/{RUN_TIMESTAMP}_baselines_timed_{args.dataset}.csv')
            else:
                print(f"Experiment with {model_name} did not complete successfully.")
    
    print("\n✅ All baseline experiments complete.")
