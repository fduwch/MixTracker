import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d
from torch_geometric.nn import SAGEConv, GCNConv, GATv2Conv, GATConv, TransformerConv, RGCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch

class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, embedding_dim, gnn_type='GATv2Conv', heads=4, dropout_rate=0.1):
        super(GraphEncoder, self).__init__()
        self.dropout_rate = dropout_rate
        self.gnn_type = gnn_type

        if gnn_type == 'GATv2Conv':
            self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim)
            self.bn1 = BatchNorm1d(hidden_channels * heads)
            self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=edge_dim)
            self.bn2 = BatchNorm1d(hidden_channels * heads)
            self.conv3 = GATv2Conv(hidden_channels * heads, embedding_dim, heads=1, edge_dim=edge_dim)
        elif gnn_type == 'GATConv':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim)
            self.bn1 = BatchNorm1d(hidden_channels * heads)
            self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=edge_dim)
            self.bn2 = BatchNorm1d(hidden_channels * heads)
            self.conv3 = GATConv(hidden_channels * heads, embedding_dim, heads=1, edge_dim=edge_dim)
        elif gnn_type == 'TransformerConv':
            self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim)
            self.bn1 = BatchNorm1d(hidden_channels * heads)
            self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=edge_dim)
            self.bn2 = BatchNorm1d(hidden_channels * heads)
            self.conv3 = TransformerConv(hidden_channels * heads, embedding_dim, heads=1, edge_dim=edge_dim)
        elif gnn_type == 'SAGEConv':
            self.conv1 = SAGEConv(in_channels, hidden_channels * heads)
            self.bn1 = BatchNorm1d(hidden_channels * heads)
            self.conv2 = SAGEConv(hidden_channels * heads, hidden_channels * heads)
            self.bn2 = BatchNorm1d(hidden_channels * heads)
            self.conv3 = SAGEConv(hidden_channels * heads, embedding_dim)
        elif gnn_type == 'GCNConv':
            self.conv1 = GCNConv(in_channels, hidden_channels * heads)
            self.bn1 = BatchNorm1d(hidden_channels * heads)
            self.conv2 = GCNConv(hidden_channels * heads, hidden_channels * heads)
            self.bn2 = BatchNorm1d(hidden_channels * heads)
            self.conv3 = GCNConv(hidden_channels * heads, embedding_dim)
        elif gnn_type == 'RGCNConv':
            num_relations = 3
            self.conv1 = RGCNConv(in_channels, hidden_channels * heads, num_relations=num_relations)
            self.bn1 = BatchNorm1d(hidden_channels * heads)
            self.conv2 = RGCNConv(hidden_channels * heads, hidden_channels * heads, num_relations=num_relations)
            self.bn2 = BatchNorm1d(hidden_channels * heads)
            self.conv3 = RGCNConv(hidden_channels * heads, embedding_dim, num_relations=num_relations)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

    def forward(self, x, edge_index, edge_attr, batch, batch_obj=None):
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        edge_type = None
        if self.gnn_type == 'RGCNConv':
            if batch_obj is not None and hasattr(batch_obj, 'edge_type'):
                edge_type = batch_obj.edge_type
            elif hasattr(batch, 'edge_type'):
                edge_type = batch.edge_type
                if hasattr(batch, 'batch'):
                    batch = batch.batch
            else:
                edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        
        batch_for_pooling = batch.batch if hasattr(batch, 'batch') else batch
        
        if self.gnn_type in ['GATv2Conv', 'GATConv', 'TransformerConv']:
            x = self.conv1(x, edge_index, edge_attr=edge_attr)
        elif self.gnn_type == 'RGCNConv':
            x = self.conv1(x, edge_index, edge_type=edge_type)
        else:
            x = self.conv1(x, edge_index)

        x = self.bn1(x).relu()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        if self.gnn_type in ['GATv2Conv', 'GATConv', 'TransformerConv']:
            x = self.conv2(x, edge_index, edge_attr=edge_attr)
        elif self.gnn_type == 'RGCNConv':
            x = self.conv2(x, edge_index, edge_type=edge_type)
        else:
            x = self.conv2(x, edge_index)
            
        x = self.bn2(x).relu()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        if self.gnn_type in ['GATv2Conv', 'GATConv', 'TransformerConv']:
            x = self.conv3(x, edge_index, edge_attr=edge_attr)
        elif self.gnn_type == 'RGCNConv':
            x = self.conv3(x, edge_index, edge_type=edge_type)
        else:
            x = self.conv3(x, edge_index)
        
        mean_pool = global_mean_pool(x, batch_for_pooling)
        max_pool = global_max_pool(x, batch_for_pooling)
        
        return torch.cat([mean_pool, max_pool], dim=1)

class SimpleLinkPredictorGNN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_channels=64, embedding_dim=32, dropout_rate=0.1, gnn_type='GATv2Conv'):
        super(SimpleLinkPredictorGNN, self).__init__()
        self.encoder = GraphEncoder(node_feature_dim, edge_feature_dim, hidden_channels, embedding_dim, gnn_type=gnn_type, dropout_rate=dropout_rate)
        
        with torch.no_grad():
            dummy_x = torch.zeros((1, node_feature_dim))
            dummy_edge_index = torch.zeros((2, 0), dtype=torch.long)
            dummy_edge_attr = torch.zeros((0, edge_feature_dim))
            dummy_batch = torch.zeros(1, dtype=torch.long)
            
            self.encoder.eval()
            if gnn_type == 'RGCNConv':
                dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr, edge_type=torch.zeros(0, dtype=torch.long))
                dummy_batch_obj = Batch.from_data_list([dummy_data])
                encoder_output_dim = self.encoder(dummy_x, dummy_edge_index, dummy_edge_attr, dummy_batch_obj.batch, batch_obj=dummy_batch_obj).shape[1]
            else:
                encoder_output_dim = self.encoder(dummy_x, dummy_edge_index, dummy_edge_attr, dummy_batch).shape[1]
            self.encoder.train()
        
        self.predictor = Sequential(
            Linear(encoder_output_dim, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Dropout(p=dropout_rate),
            Linear(hidden_channels, hidden_channels // 2),
            BatchNorm1d(hidden_channels // 2),
            ReLU(),
            Dropout(p=dropout_rate),
            Linear(hidden_channels // 2, 1)
        )
    
    def forward(self, data1, data2):
        if self.encoder.gnn_type == 'RGCNConv':
            h1 = self.encoder(data1.x, data1.edge_index, data1.edge_attr, data1.batch, batch_obj=data1)
            h2 = self.encoder(data2.x, data2.edge_index, data2.edge_attr, data2.batch, batch_obj=data2)
        else:
            h1 = self.encoder(data1.x, data1.edge_index, data1.edge_attr, data1.batch)
            h2 = self.encoder(data2.x, data2.edge_index, data2.edge_attr, data2.batch)
        
        link_features = h1 * h2
        score = self.predictor(link_features)
        return score.squeeze(-1)


class PairwiseRankingGNN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, pair_feature_dim, hidden_channels=64, embedding_dim=32, dropout_rate=0.1,
                 use_graph=True, gnn_type='GATv2Conv'):
        super(PairwiseRankingGNN, self).__init__()
        self.use_graph = use_graph
        
        if self.use_graph:
            self.encoder = GraphEncoder(node_feature_dim, edge_feature_dim, hidden_channels, embedding_dim, gnn_type=gnn_type, dropout_rate=dropout_rate)
            with torch.no_grad():
                dummy_x = torch.zeros((1, node_feature_dim))
                dummy_edge_index = torch.zeros((2, 0), dtype=torch.long)
                dummy_edge_attr = torch.zeros((0, edge_feature_dim))
                dummy_batch = torch.zeros(1, dtype=torch.long)
                
                self.encoder.eval()
                if gnn_type == 'RGCNConv':
                    dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr, edge_type=torch.zeros(0, dtype=torch.long))
                    dummy_batch_obj = Batch.from_data_list([dummy_data])
                    encoder_output_dim = self.encoder(dummy_x, dummy_edge_index, dummy_edge_attr, dummy_batch_obj.batch, batch_obj=dummy_batch_obj).shape[1]
                else:
                    encoder_output_dim = self.encoder(dummy_x, dummy_edge_index, dummy_edge_attr, dummy_batch).shape[1]
                self.encoder.train()
            mlp_input_dim = encoder_output_dim * 2 + pair_feature_dim
        else:
            self.encoder = None
            mlp_input_dim = node_feature_dim * 2 + pair_feature_dim

        self.mlp_head = Sequential(
            Linear(mlp_input_dim, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Dropout(p=dropout_rate),
            Linear(hidden_channels, hidden_channels // 2),
            BatchNorm1d(hidden_channels // 2),
            ReLU(),
            Dropout(p=dropout_rate),
            Linear(hidden_channels // 2, 1)
        )

    def forward(self, data_deposit, data_candidate, pair_features):
        if self.use_graph:
            if self.encoder.gnn_type == 'RGCNConv':
                h_D = self.encoder(data_deposit.x, data_deposit.edge_index, data_deposit.edge_attr, data_deposit.batch, batch_obj=data_deposit)
                h_C = self.encoder(data_candidate.x, data_candidate.edge_index, data_candidate.edge_attr, data_candidate.batch, batch_obj=data_candidate)
            else:
                h_D = self.encoder(data_deposit.x, data_deposit.edge_index, data_deposit.edge_attr, data_deposit.batch)
                h_C = self.encoder(data_candidate.x, data_candidate.edge_index, data_candidate.edge_attr, data_candidate.batch)
        else:
            deposit_ptr = data_deposit.ptr
            candidate_ptr = data_candidate.ptr
            h_D = data_deposit.x[deposit_ptr[:-1]]
            h_C = data_candidate.x[candidate_ptr[:-1]]
        
        combined_vector = torch.cat([h_D, h_C, pair_features], dim=1)
        score = self.mlp_head(combined_vector)
        return score.squeeze(-1)

class PriorityMatcher:
    def __init__(self, gnn_model, time_window_sec=7200, amount_tolerance=0.1, top_n=200):
        self.gnn = gnn_model
        self.time_window = time_window_sec
        self.amount_tolerance = amount_tolerance
        self.top_n = top_n
        
        if self.gnn:
            self.gnn.eval()

    def _stage0_filter(self, deposit, withdrawal_pool):
        filtered_candidates = []
        deposit_time = deposit['timestamp']
        deposit_amount = deposit['amount']
        
        for candidate in withdrawal_pool:
            if abs(candidate['timestamp'] - deposit_time) > self.time_window:
                continue
            
            if candidate['token'] != deposit['token']:
                continue

            if abs(candidate['amount'] - deposit_amount) / deposit_amount > self.amount_tolerance:
                 continue
            
            filtered_candidates.append(candidate)
            
        return filtered_candidates

    def _stage1_heuristic_score(self, deposit, candidates):
        scored_candidates = []
        for candidate in candidates:
            time_diff = abs(candidate['timestamp'] - deposit['timestamp'])
            amount_diff = abs(candidate['amount'] - deposit['amount'])
            
            time_score = 1 / (1 + time_diff) 
            amount_score = 1 / (1 + amount_diff)
            
            heuristic_score = 0.7 * time_score + 0.3 * amount_score
            scored_candidates.append((candidate, heuristic_score))
            
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates[:self.top_n]

    def _stage2_gnn_rank(self, deposit, candidates_with_scores):
        if not self.gnn or not candidates_with_scores:
            return candidates_with_scores
            
        ranked_results = []
        deposit_graph = deposit['graph']
        
        with torch.no_grad():
            for candidate, heuristic_score in candidates_with_scores:
                candidate_graph = candidate['graph']
                
                deposit_batch = Batch.from_data_list([deposit_graph])
                candidate_batch = Batch.from_data_list([candidate_graph])
                
                gnn_distance = self.gnn(deposit_batch, candidate_batch).item()
                gnn_score = 1 / (1 + gnn_distance)
                final_score = 0.4 * heuristic_score + 0.6 * gnn_score
                ranked_results.append((candidate, final_score))
                
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        return ranked_results

    def rank_candidates(self, deposit_event, withdrawal_pool):
        print("--- Stage 0: Coarse Filtering ---")
        stage0_results = self._stage0_filter(deposit_event, withdrawal_pool)
        print(f"Found {len(stage0_results)} candidates after Stage 0.")
        
        print("\n--- Stage 1: Heuristic Scoring & Top-N Selection ---")
        stage1_results = self._stage1_heuristic_score(deposit_event, stage0_results)
        print(f"Top {len(stage1_results)} candidates after Stage 1:")
        for i, (candidate, score) in enumerate(stage1_results[:5]):
            print(f"  {i+1}. Addr: {candidate['address']}, Score: {score:.4f}")

        print("\n--- Stage 2: GNN Ranking ---")
        final_ranking = self._stage2_gnn_rank(deposit_event, stage1_results)
        print(f"Final ranking after GNN:")
        for i, (candidate, score) in enumerate(final_ranking[:10]):
            print(f"  {i+1}. Addr: {candidate['address']}, Final Score: {score:.4f}")
            
        return final_ranking
