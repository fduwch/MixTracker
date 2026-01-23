import os
import csv
import random
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# PyTorch and PyTorch Geometric for GCN and GraphSAGE (following Train.py)
import torch
from torch.optim import AdamW
from torch_geometric.data import Batch
from Model import PairwiseRankingGNN, SimpleLinkPredictorGNN
from Graph import build_address_graph, build_simple_graph_from_features, build_simple_heterogeneous_graph_from_features

# --- Configuration ---
SEED = 1029
NUM_RUNS = 1
K_FOLD = 10
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_results_to_csv(results, params, filename='baseline_results.csv'):
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
    param_keys = ['Dataset', 'Model', 'Seed', 'Num_Runs', 'K_Fold']
    result_keys = ['Best_Run', 'F1', 'Accuracy', 'Precision', 'Recall', 'FPR', 'FNR']
    fieldnames = param_keys + result_keys

    # Create directory if it doesn't exist
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
    except Exception as e:
        print(f"An unexpected error occurred while saving to CSV: {e}")


def load_data_for_baselines(data_type='D1', build_graphs_for_gnn=False):
    """
    Loads MixBroker data and prepares it for classical ML models.
    Args:
        data_type: 'D1', 'D2', or 'D3'
        build_graphs_for_gnn: If True, also build PyG graphs for each address (for GCN/GraphSAGE)
    Returns:
        - node_features: A dict mapping node address to its feature vector.
        - all_pairs: A numpy array of (node1, node2) tuples.
        - all_labels: A numpy array of corresponding labels (0 or 1).
        - G: A networkx graph of all nodes and positive edges from the final dataset.
        - address_to_graph: (optional) dict mapping address to PyG Data object, if build_graphs_for_gnn=True
    """
    print(f"Loading MixBroker dataset ({data_type}) for baseline models...")
    node_feature_df = pd.read_csv('Baseline/MixBroker-main/Dataset/Graph/node_feature.csv')
    
    # Preprocess data to handle missing values, which cause errors in scikit-learn
    node_feature_df.fillna(0, inplace=True)
    
    train_pos_edge_df = pd.read_csv('Baseline/MixBroker-main/Dataset/Graph/train_pos_edge_10fold.csv')
    train_neg_edge_df = pd.read_csv('Baseline/MixBroker-main/Dataset/Graph/train_neg_edge_10fold.csv')

    nodeid_to_addr = dict(zip(node_feature_df['nodeid'], node_feature_df['node']))
    
    # Create a mapping from address to its features
    node_features = {}
    feature_cols = [col for col in node_feature_df.columns if col not in ['nodeid', 'node']]
    for _, row in node_feature_df.iterrows():
        node_features[row['node']] = row[feature_cols].values.astype(np.float32)
    
    # Get actual feature dimension (for model initialization)
    actual_feature_dim = len(feature_cols)
    print(f"Node feature dimension: {actual_feature_dim}")

    # Load base D1 positive and negative pairs
    d1_pos_pairs = [(nodeid_to_addr[r['nodeid1']], nodeid_to_addr[r['nodeid2']]) for _, r in train_pos_edge_df.iterrows()]
    d1_neg_pairs = [(nodeid_to_addr[r['nodeid1']], nodeid_to_addr[r['nodeid2']]) for _, r in train_neg_edge_df.iterrows()]
    
    pos_pairs, neg_pairs = d1_pos_pairs.copy(), d1_neg_pairs.copy()

    if data_type == 'D3':
        print("Loading extra data for D3...")
        rule2_files = ['Dataset/tornado_raw_data/heuristic2Mixer_0.1ETH.csv', 'Dataset/tornado_raw_data/heuristic2Mixer_1ETH.csv', 'Dataset/tornado_raw_data/heuristic2Mixer_10ETH.csv', 'Dataset/tornado_raw_data/heuristic2Mixer_100ETH.csv']
        rule3_files = ['Dataset/tornado_raw_data/heuristic3Mixer_0.1ETH.csv', 'Dataset/tornado_raw_data/heuristic3Mixer_1ETH.csv', 'Dataset/tornado_raw_data/heuristic3Mixer_10ETH.csv', 'Dataset/tornado_raw_data/heuristic3Mixer_100ETH.csv']

        rule2_df = pd.concat([pd.read_csv(f) for f in rule2_files])
        rule3_df = pd.concat([pd.read_csv(f) for f in rule3_files])

        pos_pairs += [ (r['sender'], r['receiver']) for _ , r in rule2_df.iterrows()]
        pos_pairs += [ (r['sender'], r['receiver']) for _ , r in rule3_df.iterrows()]

    # Combine pairs and labels
    all_pairs = pos_pairs + neg_pairs
    all_labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

    if data_type == 'D2':
        # D2 is 75% of D1
        print("Sampling 75% of D1 data to create D2 dataset...")
        set_seed(SEED)
        
        d1_all_pairs = d1_pos_pairs + d1_neg_pairs
        d1_all_labels = [1] * len(d1_pos_pairs) + [0] * len(d1_neg_pairs)
        
        combined_data = list(zip(d1_all_pairs, d1_all_labels))
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
    print(f"Final dataset size for {data_type}: {len(all_pairs)} pairs (Pos: {num_pos}, Neg: {num_neg})")

    # Build graph for graph-based models
    G = nx.Graph()
    all_nodes = list(node_features.keys())
    G.add_nodes_from(all_nodes)
    
    # Add ONLY the positive edges to the graph for embedding generation
    final_pos_pairs = [pair for pair, label in zip(all_pairs, all_labels) if label == 1]
    G.add_edges_from(final_pos_pairs)

    print(f"Graph created. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # Build PyG graphs for GNN models (GCN, GraphSAGE, RGCN) - using simple feature-based graphs for baselines
    address_to_graph = None
    address_to_hetero_graph = None  # For RGCN
    if build_graphs_for_gnn:
        print("Building simple PyTorch Geometric graphs for baseline GNN models (based on node features only)...")
        GRAPH_CACHE_DIR = 'graph_cache_baseline'
        GRAPH_CACHE_DIR_HETERO = 'graph_cache_baseline_hetero'  # Separate cache for baseline heterogeneous graphs
        os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
        os.makedirs(GRAPH_CACHE_DIR_HETERO, exist_ok=True)
        
        unique_addrs = set(addr for pair in all_pairs for addr in pair)
        address_to_graph = {}
        address_to_hetero_graph = {}
        for i, addr in enumerate(list(unique_addrs)):
            if i % 100 == 0:
                print(f"  Processing address {i+1}/{len(unique_addrs)}...")
            
            # Regular simple graph (for GCN, GraphSAGE) - based on node features only
            cache_path = os.path.join(GRAPH_CACHE_DIR, f"{addr}_simple.pt")
            if os.path.exists(cache_path):
                address_to_graph[addr] = torch.load(cache_path)
            else:
                graph, _ = build_simple_graph_from_features(addr, node_features)
                if graph:
                    torch.save(graph, cache_path)
                address_to_graph[addr] = graph
            
            # Simple heterogeneous graph (for RGCN) - based on node features only
            cache_path_hetero = os.path.join(GRAPH_CACHE_DIR_HETERO, f"{addr}_simple_hetero.pt")
            if os.path.exists(cache_path_hetero):
                address_to_hetero_graph[addr] = torch.load(cache_path_hetero)
            else:
                hetero_graph, _ = build_simple_heterogeneous_graph_from_features(addr, node_features)
                if hetero_graph:
                    torch.save(hetero_graph, cache_path_hetero)
                address_to_hetero_graph[addr] = hetero_graph

    return node_features, np.array(all_pairs), np.array(all_labels), G, address_to_graph, address_to_hetero_graph, actual_feature_dim


def get_link_features(pairs, features_or_embeddings):
    """
    Creates features for links (pairs of nodes) using the Hadamard product.
    """
    link_features = []
    
    # Get a sample dimension to handle cases where features might be missing
    sample_key = next(iter(features_or_embeddings))
    dim = len(features_or_embeddings[sample_key])

    for pair in pairs:
        node1, node2 = pair[0], pair[1]
        
        feat1 = features_or_embeddings.get(node1)
        feat2 = features_or_embeddings.get(node2)
        
        if feat1 is not None and feat2 is not None:
            link_features.append(feat1 * feat2)
        else:
            # Append a zero vector if any node's feature is missing
            link_features.append(np.zeros(dim))
            
    return np.array(link_features)


def run_experiment(config, data):
    """Runs a single experiment configuration."""
    print(f"--- Preparing for model: {config['model'].upper()} ---")
    
    model_name = config['model']
    node_features = data['node_features']
    all_pairs = data['all_pairs']
    all_labels = data['all_labels']
    G = data['G']

    all_run_results = []

    for run in range(1, NUM_RUNS + 1):
        print(f"\n{'='*20} Starting Run {run}/{NUM_RUNS} for {model_name.upper()} {'='*20}")
        set_seed(SEED)

        fold_results = []

        # --- Standard K-fold for feature-based models ---
        if model_name in ['lr', 'rf']:
            kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=SEED)
            for fold, (train_idx, val_idx) in enumerate(kf.split(all_pairs)):
                print(f"--- Starting Fold {fold+1}/{K_FOLD} ---")
                
                train_pairs, val_pairs = all_pairs[train_idx], all_pairs[val_idx]
                train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

                # --- Enforce Address-Disjoint Split (LR/RF) ---
                val_addrs = set()
                for pair in val_pairs:
                    val_addrs.add(pair[0])
                    val_addrs.add(pair[1])

                train_mask = np.array([
                    pair[0] not in val_addrs and pair[1] not in val_addrs 
                    for pair in train_pairs
                ])
                original_len = len(train_pairs)
                train_pairs = train_pairs[train_mask]
                train_labels = train_labels[train_mask]
                print(f"  - Address-Disjoint Filter: Removed {original_len - len(train_pairs)} pairs from training set.")

                X_train = get_link_features(train_pairs, node_features)
                y_train = train_labels
                X_val = get_link_features(val_pairs, node_features)
                y_val = val_labels

                if model_name == 'lr':
                    model = LogisticRegression(random_state=SEED, max_iter=1000, solver='liblinear')
                elif model_name == 'rf':
                    model = RandomForestClassifier(random_state=SEED, n_estimators=100, n_jobs=-1)
                
                print(f"Training {model.__class__.__name__} on {len(X_train)} samples...")
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                
                # ... (metric calculation remains the same)
                accuracy = accuracy_score(y_val, preds)
                precision = precision_score(y_val, preds, zero_division=0)
                recall = recall_score(y_val, preds, zero_division=0)
                f1 = f1_score(y_val, preds, zero_division=0)
                if len(np.unique(y_val)) == 2:
                    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                else:
                    fpr, fnr = 0.0, 0.0
                print(f"Fold {fold+1} F1: {f1:.4f}")
                fold_results.append((accuracy, precision, recall, f1, fpr, fnr))

        # --- GNN models (GCN, GraphSAGE, RGCN) - following Train.py's approach ---
        elif model_name in ['gcn', 'graphsage', 'rgcn']:
            # Use the same end-to-end training approach as Train.py
            # Convert pairs to Train.py's format
            packaged_data = []
            address_to_graph = data.get('address_to_graph')
            
            if address_to_graph is None:
                raise ValueError(f"Graphs not built for GNN model {model_name}. Please ensure load_data_for_baselines is called with build_graphs_for_gnn=True")
            
            # For RGCN, use heterogeneous graphs; for others, use regular graphs
            graph_dict = data.get('address_to_hetero_graph') if model_name == 'rgcn' else address_to_graph
            
            if model_name == 'rgcn' and graph_dict is None:
                raise ValueError(f"Heterogeneous graphs not built for RGCN model. Please ensure load_data_for_baselines is called with build_graphs_for_gnn=True")
            
            for (addr1, addr2), label in zip(all_pairs, all_labels):
                graph1 = graph_dict.get(addr1) if graph_dict else None
                graph2 = graph_dict.get(addr2) if graph_dict else None
                if graph1 and graph2:
                    packaged_data.append({
                        'graph1': graph1,
                        'graph2': graph2,
                        'label': torch.tensor([float(label)], dtype=torch.float32),
                        'addr1': addr1,
                        'addr2': addr2
                    })
            
            if not packaged_data:
                print(f"  Warning: No valid graph pairs found for {model_name}")
                continue
            
            packaged_data = np.array(packaged_data)
            
            # Standard K-fold cross-validation
            kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=SEED)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(packaged_data)):
                print(f"--- Starting Fold {fold+1}/{K_FOLD} ---")
                
                train_data = packaged_data[train_idx].tolist()
                val_data = packaged_data[val_idx].tolist()
                
                # --- Enforce Address-Disjoint Split (same as Train.py) ---
                val_addrs = set()
                for item in val_data:
                    val_addrs.add(item['addr1'])
                    val_addrs.add(item['addr2'])
                
                original_train_len = len(train_data)
                train_data = [
                    item for item in train_data 
                    if item['addr1'] not in val_addrs and item['addr2'] not in val_addrs
                ]
                print(f"  - Address-Disjoint Filter: Removed {original_train_len - len(train_data)} pairs from training set.")
                
                if len(train_data) < 2:
                    print(f"  Warning: Too few training samples after filtering, skipping fold")
                    continue
                
                # Use device from command line argument (passed via config)
                device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                
                # Map model name to GNN type (matching Train.py's GNN types)
                gnn_type_map = {'gcn': 'GCNConv', 'graphsage': 'SAGEConv', 'rgcn': 'RGCNConv'}
                if model_name not in gnn_type_map:
                    raise ValueError(f"Unknown GNN model: {model_name}")
                gnn_type = gnn_type_map[model_name]
                
                # Initialize model - use actual feature dimension from data
                actual_feature_dim = data.get('node_feature_dim', 19)  # Fallback to 19 if not provided
                NODE_FEATURE_DIM = actual_feature_dim
                EDGE_FEATURE_DIM = 5
                LEARNING_RATE = 0.001
                EPOCHS = 1000
                BATCH_SIZE = 32
                EARLY_STOPPING_PATIENCE = 60
                
                print(f"  Initializing {model_name.upper()} baseline model with node_feature_dim={NODE_FEATURE_DIM}")
                # Use SimpleLinkPredictorGNN instead of PairwiseRankingGNN for baselines
                model = SimpleLinkPredictorGNN(
                    node_feature_dim=NODE_FEATURE_DIM,
                    edge_feature_dim=EDGE_FEATURE_DIM,
                    hidden_channels=64,
                    embedding_dim=32,
                    dropout_rate=0.1,
                    gnn_type=gnn_type
                ).to(device)
                optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
                loss_func = torch.nn.BCEWithLogitsLoss()
                
                # Training loop (same as Train.py)
                best_f1 = 0.0
                epochs_no_improve = 0
                best_fold_metrics = None
                
                for epoch in range(1, EPOCHS + 1):
                    # Train
                    model.train()
                    total_loss = 0
                    random.shuffle(train_data)
                    batches_processed = 0
                    
                    for i in range(0, len(train_data), BATCH_SIZE):
                        batch = train_data[i:i+BATCH_SIZE]
                        if len(batch) <= 1:
                            continue
                        batches_processed += 1
                        
                        graphs1 = Batch.from_data_list([p['graph1'] for p in batch]).to(device)
                        graphs2 = Batch.from_data_list([p['graph2'] for p in batch]).to(device)
                        labels = torch.stack([p['label'] for p in batch]).to(device).squeeze(-1)
                        
                        optimizer.zero_grad()
                        # SimpleLinkPredictorGNN takes only two graphs, no pair features
                        scores = model(graphs1, graphs2)
                        loss = loss_func(scores, labels)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    
                    if batches_processed == 0:
                        continue
                    
                    avg_loss = total_loss / batches_processed
                    
                    # Evaluate
                    model.eval()
                    all_preds = []
                    all_labels_eval = []
                    
                    with torch.no_grad():
                        EVAL_BATCH_SIZE = 64
                        for i in range(0, len(val_data), EVAL_BATCH_SIZE):
                            batch = val_data[i:i+EVAL_BATCH_SIZE]
                            graphs1 = Batch.from_data_list([p['graph1'] for p in batch]).to(device)
                            graphs2 = Batch.from_data_list([p['graph2'] for p in batch]).to(device)
                            labels = torch.stack([p['label'] for p in batch]).to(device).squeeze(-1)
                            
                            # SimpleLinkPredictorGNN takes only two graphs, no pair features
                            logits = model(graphs1, graphs2)
                            preds = (torch.sigmoid(logits) > 0.5).cpu()
                            all_preds.append(preds)
                            all_labels_eval.append(labels.cpu())
                    
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
                    
                    if epoch % 100 == 0:
                        print(f"  Epoch {epoch:03d} | Loss: {avg_loss:.4f} | F1: {f1:.2%}")
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_fold_metrics = (accuracy, precision, recall, f1, fpr, fnr)
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                            print(f"  Early stopping triggered at epoch {epoch}.")
                            break
                
                if best_fold_metrics:
                    print(f"Fold {fold+1} Best F1: {best_fold_metrics[3]:.4f}")
                    fold_results.append(best_fold_metrics)
        
        # --- Link Prediction specific K-fold for graph embedding models ---
        elif model_name in ['deepwalk', 'node2vec']:
            pos_pairs = all_pairs[all_labels == 1]
            neg_pairs = all_pairs[all_labels == 0]

            kf_pos = KFold(n_splits=K_FOLD, shuffle=True, random_state=SEED)

            for fold, (train_pos_idx, val_pos_idx) in enumerate(kf_pos.split(pos_pairs)):
                print(f"--- Starting Fold {fold+1}/{K_FOLD} ---")

                # 1. Create datasets for this fold
                train_pos_edges = pos_pairs[train_pos_idx]
                val_pos_edges = pos_pairs[val_pos_idx]
                
                # Balance the validation set with an equal number of negative samples
                np.random.seed(SEED + fold)  # Reproducible sampling
                if len(neg_pairs) < len(val_pos_edges):
                    val_neg_edges = neg_pairs
                else:
                    val_neg_edges = neg_pairs[np.random.choice(len(neg_pairs), size=len(val_pos_edges), replace=False)]

                # --- Enforce Address-Disjoint Split (Graph Embeddings) ---
                val_addrs = set()
                for pair in val_pos_edges:
                    val_addrs.add(pair[0])
                    val_addrs.add(pair[1])
                for pair in val_neg_edges:
                    val_addrs.add(pair[0])
                    val_addrs.add(pair[1])

                # Filter training positive edges
                train_pos_mask = np.array([
                    pair[0] not in val_addrs and pair[1] not in val_addrs 
                    for pair in train_pos_edges
                ])
                original_pos_len = len(train_pos_edges)
                train_pos_edges = train_pos_edges[train_pos_mask]
                print(f"  - Address-Disjoint Filter: Removed {original_pos_len - len(train_pos_edges)} positive pairs from training set.")

                # Filter training negative edges
                train_neg_mask = np.array([
                    pair[0] not in val_addrs and pair[1] not in val_addrs 
                    for pair in neg_pairs
                ])
                train_neg_edges = neg_pairs[train_neg_mask]
                print(f"  - Address-Disjoint Filter: Available training negative pairs: {len(train_neg_edges)} (filtered from {len(neg_pairs)})")

                # For the classifier, we train on filtered positive and negative edges
                train_pairs = np.vstack([train_pos_edges, train_neg_edges])
                train_labels = np.array([1] * len(train_pos_edges) + [0] * len(train_neg_edges))
                val_pairs = np.vstack([val_pos_edges, val_neg_edges])
                val_labels = np.array([1] * len(val_pos_edges) + [0] * len(val_neg_edges))

                # 2. Build training graph (now much denser) and learn embeddings
                # This is the standard and correct way for link prediction evaluation:
                # The model should learn from a graph containing all known information *except* for the links it needs to predict.
                print(f"Generating embeddings for Fold {fold+1} using {model_name.upper()}...")
                G_train = G.copy()
                G_train.remove_edges_from(val_pos_edges)
                
                str_G_train = nx.relabel_nodes(G_train, {n: str(n) for n in G_train.nodes()})
                
                p_val, q_val, model_seed = 1, 1, SEED + run
                if model_name == 'node2vec':
                    p_val, q_val, model_seed = 2, 0.5, SEED + run + 100

                n2v = Node2Vec(str_G_train, dimensions=32, walk_length=20, num_walks=20, workers=4, p=p_val, q=q_val, seed=model_seed)
                print("Starting to fit the model (this may take a while)...")
                model_wv = n2v.fit(window=10, min_count=1, batch_words=4)
                
                # 3. Create hybrid features
                print("Creating hybrid features...")
                structural_features = {addr: model_wv.wv[str(addr)] for addr in G_train.nodes()}
                content_dim = len(next(iter(node_features.values())))
                structure_dim = len(next(iter(structural_features.values())))
                hybrid_features = {}
                for addr in G.nodes():
                    content = node_features.get(addr, np.zeros(content_dim))
                    structure = structural_features.get(addr, np.zeros(structure_dim))
                    hybrid_features[addr] = np.concatenate([content, structure])
                
                # 4. Train and evaluate downstream classifier
                X_train = get_link_features(train_pairs, hybrid_features)
                X_val = get_link_features(val_pairs, hybrid_features)
                
                # Using a more powerful downstream classifier as requested.
                # RandomForest often performs better than Logistic Regression and provides a fairer comparison.
                model = RandomForestClassifier(random_state=SEED, n_estimators=100, n_jobs=-1)
                print(f"Training {model.__class__.__name__} on {len(X_train)} samples...")
                model.fit(X_train, train_labels)
                preds = model.predict(X_val)

                # ... (metric calculation)
                accuracy = accuracy_score(val_labels, preds)
                precision = precision_score(val_labels, preds, zero_division=0)
                recall = recall_score(val_labels, preds, zero_division=0)
                f1 = f1_score(val_labels, preds, zero_division=0)
                if len(np.unique(val_labels)) == 2:
                    tn, fp, fn, tp = confusion_matrix(val_labels, preds).ravel()
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                else:
                    fpr, fnr = 0.0, 0.0
                print(f"Fold {fold+1} F1: {f1:.4f}")
                fold_results.append((accuracy, precision, recall, f1, fpr, fnr))
        else:
             raise ValueError(f"Model {model_name} not supported.")

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

        all_run_results.append({
            'run': run,
            'avg_accuracy': avg_accuracy, 'std_accuracy': std_accuracy,
            'avg_precision': avg_precision, 'std_precision': std_precision,
            'avg_recall': avg_recall, 'std_recall': std_recall,
            'avg_f1': avg_f1, 'std_f1': std_f1,
            'avg_fpr': avg_fpr, 'std_fpr': std_fpr,
            'avg_fnr': avg_fnr, 'std_fnr': std_fnr,
        })
    
    avg_accuracy = np.mean([r['avg_accuracy'] for r in all_run_results])
    avg_precision = np.mean([r['avg_precision'] for r in all_run_results])
    avg_recall = np.mean([r['avg_recall'] for r in all_run_results])
    avg_f1 = np.mean([r['avg_f1'] for r in all_run_results])
    avg_fpr = np.mean([r['avg_fpr'] for r in all_run_results])
    avg_fnr = np.mean([r['avg_fnr'] for r in all_run_results])
    
    std_accuracy = np.std([r['avg_accuracy'] for r in all_run_results])
    std_precision = np.std([r['avg_precision'] for r in all_run_results])
    std_recall = np.std([r['avg_recall'] for r in all_run_results])
    std_f1 = np.std([r['avg_f1'] for r in all_run_results])
    std_fpr = np.std([r['avg_fpr'] for r in all_run_results])
    std_fnr = np.std([r['avg_fnr'] for r in all_run_results])
    
    best_run_idx = np.argmax([r['avg_f1'] for r in all_run_results])
    
    return {
        'run': all_run_results[best_run_idx]['run'],
        'avg_accuracy': avg_accuracy, 'std_accuracy': std_accuracy,
        'avg_precision': avg_precision, 'std_precision': std_precision,
        'avg_recall': avg_recall, 'std_recall': std_recall,
        'avg_f1': avg_f1, 'std_f1': std_f1,
        'avg_fpr': avg_fpr, 'std_fpr': std_fpr,
        'avg_fnr': avg_fnr, 'std_fnr': std_fnr,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline Models for Link Prediction Training')
    parser.add_argument('--dataset', type=str, default='D1', choices=['D1', 'D2', 'D3'],
                        help='Choose the dataset to use: D1, D2, or D3')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['lr', 'rf', 'deepwalk', 'node2vec', 'gcn', 'graphsage', 'rgcn', 'lstm', 'all'],
                        help='Choose the baseline model to run.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use. e.g., 0, 1, 2...')
    args = parser.parse_args()

    # Set device based on user input (for GNN models)
    if torch.cuda.is_available():
        DEVICE = torch.device(f'cuda:{args.gpu}')
    else:
        DEVICE = torch.device('cpu')
    
    print(f"Using dataset: {args.dataset}")
    print(f"Using device: {DEVICE} (for GNN models: GCN, GraphSAGE, RGCN)")

    # 1. Load data
    # Check if we need to build graphs for GNN models
    models_to_run = []
    if args.model == 'all':
        models_to_run = ['lr', 'rf', 'gcn', 'graphsage', 'rgcn', 'deepwalk', 'node2vec']
    else:
        models_to_run.append(args.model)
    
    # Build graphs if any GNN model is included
    build_graphs = any(m in ['gcn', 'graphsage', 'rgcn'] for m in models_to_run)
    node_features, all_pairs, all_labels, G, address_to_graph, address_to_hetero_graph, node_feature_dim = load_data_for_baselines(args.dataset, build_graphs_for_gnn=build_graphs)

    # 2. Define experiment configurations (already done above)

    # 3. Run experiments
    for model_name in models_to_run:
        print(f"\n{'='*25}")
        print(f"Starting Experiment for Model: {model_name.upper()}")
        print(f"{'='*25}")
        
        config = {
            'model': model_name,
            'device': DEVICE  # Pass device to config for GNN models
        }
        data_package = {
            'node_features': node_features,
            'all_pairs': all_pairs,
            'all_labels': all_labels,
            'G': G,
            'address_to_graph': address_to_graph if model_name in ['gcn', 'graphsage', 'rgcn'] else None,
            'address_to_hetero_graph': address_to_hetero_graph if model_name == 'rgcn' else None,
            'node_feature_dim': node_feature_dim  # Pass actual feature dimension
        }
        results = run_experiment(config, data_package)

        # --- Save results ---
        if results:
            params = {
                'Dataset': args.dataset,
                'Model': model_name.upper(),
                'Seed': SEED,
                'Num_Runs': NUM_RUNS,
                'K_Fold': K_FOLD
            }
            save_results_to_csv(results, params, f'Results/{RUN_TIMESTAMP}_baselines_{args.dataset}.csv')
        else:
            print(f"Experiment with config {config} did not complete successfully.")

    print("\n✅ All baseline experiments complete.")
