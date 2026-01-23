import pandas as pd
import os
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import time
import warnings

warnings.filterwarnings("ignore")

TORNADO_CASH_ADDRESSES = {
    '0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc',
    '0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936',
    '0x910cbd523d972eb0a6f4cae4618ad62622b39dbf',
    '0xa160cdab225685da1d56aa342ad8841c3b53f291',
    '0xd90e2f925da726b50c4ed8d0fb90ad053324f31b',
    '0x905b63fff465b9ffbf41dea908ceb12478ec7601',
    '0x722122df12d4e14e13ac3b6895a86e84145b6967',
}

def _load_all_txs_for_address(address):
    """Loads all transaction types for a given address from CSV files."""
    all_txs_dfs = []
    core_columns = {
        'blockNumber': 'str', 'from': 'str', 'to': 'str', 'value': 'str',
        'gasUsed': 'str', 'gasPrice': 'str', 'timeStamp': 'str'
    }
    tx_type_mapping = {'Normal': 0, 'Internal': 1, 'ERC20': 2}

    for tx_type in ['Normal', 'Internal', 'ERC20']:
        file_path = f'Dataset/TornadoNeighborTransactions/{tx_type}/{address}.csv'
        if not os.path.exists(file_path):
            continue
        
        address_tx_df = pd.read_csv(file_path, dtype=str, engine='c')
        standard_df = pd.DataFrame()

        for col in core_columns.keys():
            if col in address_tx_df.columns:
                standard_df[col] = address_tx_df[col]
            else:
                standard_df[col] = '0'
        
        standard_df['tx_type'] = tx_type_mapping[tx_type]
        all_txs_dfs.append(standard_df)

    if not all_txs_dfs:
        return None

    combined_tx_df = pd.concat(all_txs_dfs, ignore_index=True)
    combined_tx_df['blockNumber'] = pd.to_numeric(combined_tx_df['blockNumber'], errors='coerce')
    combined_tx_df['timeStamp'] = pd.to_numeric(combined_tx_df['timeStamp'], errors='coerce')
    combined_tx_df.dropna(subset=['blockNumber', 'timeStamp'], inplace=True)
    return combined_tx_df

def _build_graph_from_df(df):
    """Builds a PyG Data object from a transaction dataframe."""
    if df is None or df.empty:
        return None, None

    all_nodes = pd.unique(df[['from', 'to']].values.ravel('K'))
    addr_to_index = {addr: i for i, addr in enumerate(all_nodes)}

    edge_index, edge_attr = extract_edge_features(df, addr_to_index)
    node_feature_df = extract_node_features(df)

    node_feature_df = node_feature_df.set_index('address')
    ordered_feature_df = node_feature_df.reindex(all_nodes).fillna(0)

    scaler = StandardScaler()
    feature_cols = ordered_feature_df.columns
    log_transformed_features = ordered_feature_df[feature_cols].apply(np.log1p)
    scaled_features = scaler.fit_transform(log_transformed_features)
    ordered_feature_df[feature_cols] = np.nan_to_num(scaled_features)

    x = torch.tensor(ordered_feature_df.values, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, addr_to_index

def build_address_graph(address, start_block=None, end_block=None, max_txs=None):
    combined_tx_df = _load_all_txs_for_address(address)
    if combined_tx_df is None:
        return None, None
    
    if start_block is not None:
        combined_tx_df = combined_tx_df[combined_tx_df['blockNumber'] >= start_block]
    if end_block is not None:
        combined_tx_df = combined_tx_df[combined_tx_df['blockNumber'] <= end_block]
        
    if max_txs is not None and len(combined_tx_df) > max_txs:
        combined_tx_df = combined_tx_df.nlargest(max_txs, 'timeStamp', keep='first')

    return _build_graph_from_df(combined_tx_df)


def build_address_graph_with_timing(address, start_block=None, end_block=None, max_txs=None):
    import time
    
    io_start = time.time()
    combined_tx_df = _load_all_txs_for_address(address)
    io_time = time.time() - io_start
    
    if combined_tx_df is None:
        return None, None, io_time, 0.0
    
    compute_start = time.time()
    
    if start_block is not None:
        combined_tx_df = combined_tx_df[combined_tx_df['blockNumber'] >= start_block]
    if end_block is not None:
        combined_tx_df = combined_tx_df[combined_tx_df['blockNumber'] <= end_block]
        
    if max_txs is not None and len(combined_tx_df) > max_txs:
        combined_tx_df = combined_tx_df.nlargest(max_txs, 'timeStamp', keep='first')

    graph, addr_to_index = _build_graph_from_df(combined_tx_df)
    compute_time = time.time() - compute_start
    
    return graph, addr_to_index, io_time, compute_time

def build_address_graph_by_tx_volume(address, reference_timestamp, mode, ratio, max_txs=None):
    all_txs_df = _load_all_txs_for_address(address)
    if all_txs_df is None:
        return None, None
    
    all_txs_df = all_txs_df.reset_index().sort_values(by=['timeStamp', 'index'], ascending=True).drop(columns=['index'])
    
    total_tx_count = len(all_txs_df)
    num_to_add = int(total_tx_count * ratio)
    final_df = None

    if mode == 'deposit':
        base_df = all_txs_df[all_txs_df['timeStamp'] <= reference_timestamp]
        extra_df_pool = all_txs_df[all_txs_df['timeStamp'] > reference_timestamp]
        final_df = pd.concat([base_df, extra_df_pool.head(num_to_add)])
    elif mode == 'withdrawal':
        base_df = all_txs_df[all_txs_df['timeStamp'] >= reference_timestamp]
        extra_df_pool = all_txs_df[all_txs_df['timeStamp'] < reference_timestamp]
        final_df = pd.concat([extra_df_pool.tail(num_to_add), base_df])
    
    if max_txs is not None and final_df is not None and len(final_df) > max_txs:
        final_df = final_df.nlargest(max_txs, 'timeStamp', keep='first')

    return _build_graph_from_df(final_df)


def build_simple_graph_from_features(address, node_features_dict):
    if address not in node_features_dict:
        return None, None
    
    features = node_features_dict[address]
    
    if isinstance(features, np.ndarray):
        if features.ndim == 1:
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        else:
            x = torch.tensor(features, dtype=torch.float32)
    else:
        x = torch.tensor([features], dtype=torch.float32).unsqueeze(0) if not isinstance(features, torch.Tensor) else features
    
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    edge_attr = torch.zeros((1, 5), dtype=torch.float32)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, {address: 0}


def build_simple_heterogeneous_graph_from_features(address, node_features_dict):
    if address not in node_features_dict:
        return None, None
    
    features = node_features_dict[address]
    
    if isinstance(features, np.ndarray):
        if features.ndim == 1:
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        else:
            x = torch.tensor(features, dtype=torch.float32)
    else:
        x = torch.tensor([features], dtype=torch.float32).unsqueeze(0) if not isinstance(features, torch.Tensor) else features
    
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    edge_attr = torch.zeros((1, 4), dtype=torch.float32)
    edge_type = torch.tensor([0], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)
    return data, {address: 0}


def build_heterogeneous_graph(address, start_block=None, end_block=None, max_txs=None):
    combined_tx_df = _load_all_txs_for_address(address)
    if combined_tx_df is None:
        return None, None
    
    if start_block is not None:
        combined_tx_df = combined_tx_df[combined_tx_df['blockNumber'] >= start_block]
    if end_block is not None:
        combined_tx_df = combined_tx_df[combined_tx_df['blockNumber'] <= end_block]
        
    if max_txs is not None and len(combined_tx_df) > max_txs:
        combined_tx_df = combined_tx_df.nlargest(max_txs, 'timeStamp', keep='first')

    if combined_tx_df.empty:
        return None, None

    all_nodes = pd.unique(combined_tx_df[['from', 'to']].values.ravel('K'))
    addr_to_index = {addr: i for i, addr in enumerate(all_nodes)}
    
    node_feature_df = extract_node_features(combined_tx_df)
    node_feature_df = node_feature_df.set_index('address')
    ordered_feature_df = node_feature_df.reindex(all_nodes).fillna(0)
    
    scaler = StandardScaler()
    feature_cols = ordered_feature_df.columns
    log_transformed_features = ordered_feature_df[feature_cols].apply(np.log1p)
    scaled_features = scaler.fit_transform(log_transformed_features)
    ordered_feature_df[feature_cols] = np.nan_to_num(scaled_features)
    
    x = torch.tensor(ordered_feature_df.values, dtype=torch.float32)
    
    edges_df = combined_tx_df.copy()
    edges_df['from_idx'] = edges_df['from'].map(addr_to_index)
    edges_df['to_idx'] = edges_df['to'].map(addr_to_index)
    edges_df.dropna(subset=['from_idx', 'to_idx'], inplace=True)
    
    edge_index = torch.tensor([edges_df['from_idx'].values, edges_df['to_idx'].values], dtype=torch.long)
    edge_type = torch.tensor(edges_df['tx_type'].values, dtype=torch.long)
    
    edges_df['value'] = pd.to_numeric(edges_df['value'], errors='coerce').fillna(0) / 1e18
    edges_df['gasUsed'] = pd.to_numeric(edges_df['gasUsed'], errors='coerce').fillna(0)
    edges_df['gasPrice'] = pd.to_numeric(edges_df['gasPrice'], errors='coerce').fillna(0) / 1e9
    edges_df['timeStamp'] = pd.to_numeric(edges_df['timeStamp'], errors='coerce').fillna(0)
    
    edge_feature_cols = ['value', 'gasUsed', 'gasPrice', 'timeStamp']
    scaler = StandardScaler()
    numerical_cols = ['value', 'gasUsed', 'gasPrice', 'timeStamp']
    log_transformed_edges = edges_df[numerical_cols].apply(np.log1p)
    scaled_edges = scaler.fit_transform(log_transformed_edges)
    edges_df[numerical_cols] = np.nan_to_num(scaled_edges)
    
    edge_attr = torch.tensor(edges_df[edge_feature_cols].values, dtype=torch.float32)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)
    return data, addr_to_index


def extract_edge_features(df, addr_to_index):
    edges_df = df.copy()
    edges_df['from_idx'] = edges_df['from'].map(addr_to_index)
    edges_df['to_idx'] = edges_df['to'].map(addr_to_index)
    edges_df.dropna(subset=['from_idx', 'to_idx'], inplace=True)

    edge_index = torch.tensor([edges_df['from_idx'].values, edges_df['to_idx'].values], dtype=torch.long)

    edges_df['value'] = pd.to_numeric(edges_df['value'], errors='coerce').fillna(0) / 1e18
    edges_df['gasUsed'] = pd.to_numeric(edges_df['gasUsed'], errors='coerce').fillna(0)
    edges_df['gasPrice'] = pd.to_numeric(edges_df['gasPrice'], errors='coerce').fillna(0) / 1e9
    edges_df['timeStamp'] = pd.to_numeric(edges_df['timeStamp'], errors='coerce').fillna(0)

    edge_feature_cols = ['value', 'gasUsed', 'gasPrice', 'timeStamp', 'tx_type']
    
    scaler = StandardScaler()
    numerical_cols = ['value', 'gasUsed', 'gasPrice', 'timeStamp']
    log_transformed_edges = edges_df[numerical_cols].apply(np.log1p)
    scaled_edges = scaler.fit_transform(log_transformed_edges)
    
    edges_df[numerical_cols] = np.nan_to_num(scaled_edges)

    edge_attr = torch.tensor(edges_df[edge_feature_cols].values, dtype=torch.float32)
    
    return edge_index, edge_attr


def extract_node_features(df):
    df['value_eth'] = pd.to_numeric(df['value'], errors='coerce').fillna(0) / 1e18
    df['gasUsed'] = pd.to_numeric(df['gasUsed'], errors='coerce').fillna(0)
    df['gasPrice_gwei'] = pd.to_numeric(df['gasPrice'], errors='coerce').fillna(0) / 1e9

    out_features = df.groupby('from').agg(
        out_degree=('to', 'count'),
        unique_to_addresses=('to', 'nunique'),
        total_ether_sent=('value_eth', 'sum'),
        avg_ether_sent=('value_eth', 'mean'),
        max_ether_sent=('value_eth', 'max'),
        min_ether_sent=('value_eth', 'min'),
        avg_gas_price=('gasPrice_gwei', 'mean'),
        total_gas_used=('gasUsed', 'sum')
    ).rename_axis('address')

    in_features = df.groupby('to').agg(
        in_degree=('from', 'count'),
        unique_from_addresses=('from', 'nunique'),
        total_ether_received=('value_eth', 'sum'),
        avg_ether_received=('value_eth', 'mean'),
        max_ether_received=('value_eth', 'max'),
        min_ether_received=('value_eth', 'min')
    ).rename_axis('address')

    from_blocks = df[['from', 'blockNumber']].rename(columns={'from': 'address'})
    to_blocks = df[['to', 'blockNumber']].rename(columns={'to': 'address'})
    all_blocks = pd.concat([from_blocks, to_blocks])
    
    lifetime_features = all_blocks.groupby('address').agg(
        min_block=('blockNumber', 'min'),
        max_block=('blockNumber', 'max')
    )
    lifetime_features['lifetime_blocks'] = lifetime_features['max_block'] - lifetime_features['min_block']
    
    tx_to_mixer = df[df['to'].isin(TORNADO_CASH_ADDRESSES)]
    mixer_out_features = tx_to_mixer.groupby('from').agg(
        sent_to_mixer_count=('to', 'count'),
        sent_to_mixer_value=('value_eth', 'sum')
    ).rename_axis('address')

    tx_from_mixer = df[df['from'].isin(TORNADO_CASH_ADDRESSES)]
    mixer_in_features = tx_from_mixer.groupby('to').agg(
        received_from_mixer_count=('from', 'count'),
        received_from_mixer_value=('value_eth', 'sum')
    ).rename_axis('address')

    mixer_features = pd.concat([mixer_out_features, mixer_in_features], axis=1)
    
    all_features = pd.concat([in_features, out_features, lifetime_features['lifetime_blocks'], mixer_features], axis=1)
    
    all_nodes = pd.unique(df[['from', 'to']].values.ravel('K'))
    feature_df = all_features.reindex(all_nodes).fillna(0)
    
    return feature_df.reset_index().rename(columns={'index': 'address'})


if __name__ == '__main__':
    start_time = time.time()
    data, addr_to_index = build_address_graph('0x03236093522cdcbac662ffbebd6a951349082b72')
    print(data)
    print(data.x[0],data.x[-1])
    print(data.edge_attr[0])
    print(data.edge_attr[-1])
    print(f"Time taken: {time.time() - start_time} seconds")