import torch
import pandas as pd
import json
import argparse
import os
from tqdm import tqdm
from torch_geometric.data import Batch
from dataclasses import dataclass, fields
from collections import defaultdict
import time
from datetime import datetime

# Correctly import from existing project files instead of duplicating code.
from Graph import build_address_graph
from Model import PairwiseRankingGNN
from Main import package_full_graph_pairs, set_seed

@dataclass
class PredictorConfig:
    """Configuration settings for the link prediction script."""
    source_address_file: str = 'Dataset/AMLValidation/heist_validation_uniq.json'
    # source_address_file: str = 'Dataset/AMLValidation/heist_validation_all_uniq.json'
    model_path: str = 'SavedModels/D3_TestF1-0.9756_20260118_131434.pth'
    # 7， 30， 90， 365, 730
    output_file: str = 'Results/link_predictions_hack_90day_top5_0118.json'
    address_label_file: str = 'Dataset/tornado_cash_neighbor_address_label.json'
    withdraw_tx_dir: str = 'Dataset/TornadoContractTransaction'
    batch_size: int = 64
    time_window_days: int = 90
    top_n: int = 5
    gpu: int = 0

CONFIG = PredictorConfig()

# --- Prediction Logic (adapted from Main.py) ---

def predict_links_in_batch(model, device, source_address, candidate_addresses):
    """
    Predicts links for a single source address against a batch of candidate addresses.
    Returns a list of scores for the candidates.
    """
    if not candidate_addresses:
        return []

    # 1. Prepare data for the model
    # The source address graph will be duplicated for each pair in the batch
    packaged_pairs = package_full_graph_pairs([
        {'deposit_address': source_address, 'withdraw_address': candidate}
        for candidate in candidate_addresses
    ])

    if not packaged_pairs:
        print(f"Warning: Could not build graph for source {source_address} or one of its candidates. Skipping batch.")
        return [0.0] * len(candidate_addresses)

    # 2. Batch the data for the model
    graphs1 = Batch.from_data_list([p['graph1'] for p in packaged_pairs]).to(device)
    graphs2 = Batch.from_data_list([p['graph2'] for p in packaged_pairs]).to(device)
    features = torch.stack([p['features'] for p in packaged_pairs]).to(device)

    # 3. Run Prediction
    with torch.no_grad():
        logits = model(graphs1, graphs2, features)
        scores = torch.sigmoid(logits).cpu().numpy()

    return scores


# --- Main Analysis Function ---

def find_top_linked_addresses(config: PredictorConfig):
    """
    Finds and ranks withdrawal addresses linked to a given list of illicit deposit addresses.
    """
    set_seed(1029)
    DEVICE = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # --- Derived Configs ---
    time_window_seconds = config.time_window_days * 24 * 60 * 60

    # --- Load Model ---
    print(f"Loading model from {config.model_path}...")
    model = PairwiseRankingGNN(
        node_feature_dim=19,
        edge_feature_dim=5,
        pair_feature_dim=1,
        use_graph=True,
        gnn_type='GATv2Conv'
    ).to(DEVICE)
    model.load_state_dict(torch.load(config.model_path, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

    # --- Load Data ---
    print(f"Loading source addresses from {config.source_address_file}...")
    with open(config.source_address_file, 'r') as f:
        source_txs = json.load(f)

    print(f"Loading address labels from {config.address_label_file}...")
    with open(config.address_label_file, 'r') as f:
        address_labels = json.load(f)

    all_results = []
    
    # Process each unique source transaction directly
    for deposit_event in tqdm(source_txs[:], desc="Analyzing source addresses"):
        source_addr = deposit_event['deposit_address']
        contract_addrs = deposit_event['contract_address']
        deposit_ts = deposit_event['deposit_time']
        opt_types = deposit_event['opt_type']

        potential_links = []

        for dep_id, contract_addr in enumerate(contract_addrs):
            opt_type = opt_types[dep_id]
            withdraw_csv_path = os.path.join(config.withdraw_tx_dir, contract_addr, 'withdraw.csv')

            if not os.path.exists(withdraw_csv_path):
                print(f"Warning: Withdraw CSV not found for contract {contract_addr}, skipping.")
                continue
            
            withdraw_df = pd.read_csv(withdraw_csv_path)
            
            # Filter withdrawals by time window
            time_filtered_df = withdraw_df[
                (withdraw_df['timeStamp'] >= deposit_ts) &
                (withdraw_df['timeStamp'] <= deposit_ts + time_window_seconds) &
                (withdraw_df['opt_type'] == opt_type)
            ]
            
            for _, row in time_filtered_df.iterrows():
                potential_links.append({
                    "withdraw_address": row['to'],
                    "withdraw_time": row['timeStamp'],
                    "source_contract": contract_addr
                })

        # Batch predict for all potential links for the current source address
        if potential_links:
            candidate_addrs = [p['withdraw_address'] for p in potential_links]
            
            scores = []
            for i in range(0, len(candidate_addrs), config.batch_size):
                batch_candidates = candidate_addrs[i:i+config.batch_size]
                batch_scores = predict_links_in_batch(model, DEVICE, source_addr, batch_candidates)
                scores.extend(batch_scores)

            for i, score in enumerate(scores):
                potential_links[i]['score'] = score
                potential_links[i]['address_label'] = address_labels.get(potential_links[i]['withdraw_address'], {}).get('label', "")
        
        # Filter, sort, and get top N results
        valid_links = []
        seen = set()
        for p in sorted((x for x in potential_links if x['score'] > 0.5), key=lambda x: x['score'], reverse=True): #  and x['address_label'] != "" and ".eth" not in x['address_label']
            addr = p['withdraw_address']
            if addr not in seen:
                valid_links.append(p)
                seen.add(addr)
        top_links = valid_links[:config.top_n]

        # Structure the output for the current source address
        top_withdrawals_details = []
        for link in top_links:
            withdraw_addr = link['withdraw_address']
            top_withdrawals_details.append({
                "withdraw_address": withdraw_addr,
                "withdraw_address_label": address_labels.get(withdraw_addr, {}).get('label', ""),
                "withdraw_time": datetime.fromtimestamp(link['withdraw_time']).strftime('%Y-%m-%d %H:%M:%S'),
                "score": float(link['score']),
                "source_contract": link['source_contract']
            })
            
        all_results.append({
            "deposit_address": source_addr,
            "deposit_address_label": deposit_event.get('label', 'Unknown'),
            "deposit_times": [datetime.fromtimestamp(deposit_event['deposit_time']).strftime('%Y-%m-%d %H:%M:%S')],
            "deposit_contracts": sorted(list(set(contract_addrs))),
            "top_n_withdraw_addresses": top_withdrawals_details
        })

    # --- Save Results ---
    print(f"\nSaving {len(all_results)} results to {config.output_file}...")
    with open(config.output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Link prediction for illicit addresses.')
    
    # Introspect the dataclass to automatically build arguments
    for field in fields(PredictorConfig):
        parser.add_argument(
            f'--{field.name}',
            type=field.type,
            default=field.default,
            help=f'Default: {field.default}'
        )
        
    args = parser.parse_args()
    config = PredictorConfig(**vars(args))
    
    find_top_linked_addresses(config)
