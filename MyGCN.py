import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kendalltau
import numpy as np
import random
import os


# --- 1. Fix Random Seed ---
def set_seed(seed):
    """
    Sets the random seed for reproducibility across different libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 2. Network and SIR Data Loading
def Read(name):
    G = nx.Graph()
    f = open(name, "r")
    firstLine = f.readline()
    V = int(firstLine.split(" ")[0])
    for _ in range(0, int((firstLine.split(" ")[1].split("\n"))[0])):
        NewEdge = f.readline().split(" ")
        v1 = int(NewEdge[0])
        v2 = int((NewEdge[1].split("\n")[0]))
        G.add_edge(v1, v2)
    f.close()
    return G


def Read_SIR(file_name):
    res = dict()
    with open(file_name, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            nodev = line.split(":")
            res[int(nodev[0])] = float((nodev[1].split("\n"))[0])
    return res


# 3. Optimized Feature Calculation (MAM, SI, NI, etc.)
def calculate_network_features(G):
    """
    Calculates MAM, SI, NI, Degree, and KShell for each node in the graph efficiently.
    Returns a dictionary of unnormalized feature arrays.
    """
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # K-shell and Degree values
    ks_values_dict = nx.core_number(G)
    dc_values_dict = dict(G.degree())

    ks_values = np.array([ks_values_dict.get(node, 0) for node in nodes], dtype=float)
    dc_values = np.array([dc_values_dict.get(node, 0) for node in nodes], dtype=float)

    # SI calculation (Vectorized)
    si_denominator = np.where(ks_values != 0, ks_values, 1e-9)
    SI = dc_values / si_denominator

    # NI calculation (Vectorized)
    NI = np.zeros(num_nodes)
    edge_index_list = []
    for u, v in G.edges():
        edge_index_list.append([node_to_idx[u], node_to_idx[v]])
        edge_index_list.append([node_to_idx[v], node_to_idx[u]])
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    temp_conv_ni = GCNConv(1, 1, add_self_loops=False, normalize=False)
    with torch.no_grad():
        temp_input = torch.tensor(dc_values, dtype=torch.float32).unsqueeze(1)
        sum_n1_tensor = temp_conv_ni(temp_input, edge_index).squeeze(1)

    ni_denominator = np.where(ks_values != 0, ks_values ** 2, 1e-9)
    NI = sum_n1_tensor.numpy() / ni_denominator

    # MAM calculation
    max_ni = np.max(NI) if NI.size > 0 else 1.0
    max_si = np.max(SI) if SI.size > 0 else 1.0
    max_ks = np.max(ks_values) if ks_values.size > 0 else 1.0

    MAM = np.zeros(num_nodes)
    for i, node_i in enumerate(nodes):
        sum_n2 = 0.0
        for node_j, d_ij in nx.single_source_shortest_path_length(G, source=node_i, cutoff=2).items():
            if node_j == node_i:
                continue

            idx_j = node_to_idx[node_j]

            term_ni = (NI[idx_j] / max_ni) if max_ni != 0 else 0
            term_si = (SI[idx_j] / max_si) if max_si != 0 else 0
            term_ks = (ks_values[idx_j] / max_ks) if max_ks != 0 else 0

            sum_n2 += (term_ni + term_si + term_ks) / (d_ij ** 2) if d_ij != 0 else 0

        MAM[i] = ks_values[i] * sum_n2

    features = {
        'MAM': MAM.astype(np.float32),
        'SI': SI.astype(np.float32),
        'NI': NI.astype(np.float32),
        'Degree': dc_values.astype(np.float32),
        'KShell': ks_values.astype(np.float32)
    }
    return features, node_to_idx


# 4. GCN Model Class
class GCNRanker(nn.Module):
    def __init__(self, num_features, hidden_channels=64, num_layers=2):
        super(GCNRanker, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(num_features, hidden_channels))
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
        self.output_layer = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = torch.relu(x)
        x = self.output_layer(x)
        return x.squeeze(1)


# 5. Optimized Pairwise Ranking Loss
class OptimizedPairwiseRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(OptimizedPairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, predictions, ground_truth):
        ground_truth_expanded = ground_truth.unsqueeze(1)
        pred_diff_matrix = predictions.unsqueeze(1) - predictions.unsqueeze(0)
        gt_comparison_matrix = (ground_truth_expanded > ground_truth_expanded.T).float()

        loss_matrix = torch.relu(self.margin - pred_diff_matrix)
        masked_loss = loss_matrix * gt_comparison_matrix
        total_loss = torch.sum(masked_loss)
        num_relevant_pairs = torch.sum(gt_comparison_matrix)

        return total_loss / (num_relevant_pairs + 1e-9) if num_relevant_pairs > 0 else torch.tensor(0.0,
                                                                                                    device=predictions.device)


# 6. Evaluation Function
def evaluate_kendall_tau(predictions, ground_truth):
    if len(predictions) == 0 or len(ground_truth) == 0:
        return 0.0
    if len(np.unique(predictions)) < 2 or len(np.unique(ground_truth)) < 2:
        return 0.0
    tau, _ = kendalltau(predictions, ground_truth)
    return tau


# --- Experiment Setup ---
# All available networks and their beta values
ALL_NETWORK_CONFIGS = {
    'dolphins': 0.17, 'PolBooks': 0.09, 'USAir97': 0.02, 'netsci': 0.14,
    'WikiVote': 0.06, 'email': 0.06, 'Power': 0.26, 'Router': 0.08,
    'PGP': 0.06, 'AstroPh': 0.02, 'Hamster': 0.03, 'AS': 0.01,
    'CaSandiAuths': 0.27, 'EmailEnron': 0.08, 'Voles': 0.11,
    'EcoStmarks': 0.06, 'CSPhd': 0.26, 'Plant': 0.11, 'Crime': 0.2
}

# Define the FIXED 10 training and 8 testing networks
# (Based on common split strategies, prioritizing larger/diverse graphs for training)
test_networks_names = [
    "USAir97", "Crime", "CaSandiAuths", "EmailEnron", "PolBooks", "dolphins","netsci","WikiVote", "EcoStmarks"
]

train_networks_names = [
    "PGP", "Power", "AS", "email" , "CSPhd", "Voles","Plant", "AstroPh","Router", "Hamster"
]

# Verify no overlap and all networks are valid
if not set(train_networks_names).isdisjoint(set(test_networks_names)):
    raise ValueError("Train and Test network sets overlap!")
for net_name in train_networks_names + test_networks_names:
    if net_name not in ALL_NETWORK_CONFIGS:
        raise ValueError(f"Network '{net_name}' is not defined in ALL_NETWORK_CONFIGS.")

print(f"Fixed Training Networks ({len(train_networks_names)}): {train_networks_names}")
print(f"Fixed Test Networks ({len(test_networks_names)}): {test_networks_names}")

# Hyperparameters
RANDOM_SEED = 42
set_seed(RANDOM_SEED)

EPOCHS_PER_GRAPH_TRAIN = 20  # Epochs to train on each *training* graph during the main training loop
GLOBAL_EPOCHS = 5  # Number of times to iterate through the entire training set of graphs
LEARNING_RATE = 0.001
MARGIN = 1.0
HIDDEN_CHANNELS = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_multi_graph_train_and_test(feature_combination_name, feature_keys):
    print(f"\n{'=' * 50}\n--- Running Scenario: {feature_combination_name} ---")

    # 1. Initialize the GCN model
    num_input_features = len(feature_keys)
    model = GCNRanker(num_input_features, HIDDEN_CHANNELS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = OptimizedPairwiseRankingLoss(margin=MARGIN)

    # 2. Main Training Phase (on 10 networks)
    print("\n--- Starting Main Training Phase ---")
    for global_epoch in range(GLOBAL_EPOCHS):
        print(f"  Global Epoch {global_epoch + 1}/{GLOBAL_EPOCHS}")
        # Shuffle order of training graphs for each global epoch, but the set itself is fixed
        # Create a shuffled copy to avoid modifying the original list if it's used elsewhere
        shuffled_train_networks = list(train_networks_names)
        random.shuffle(shuffled_train_networks)

        current_global_epoch_losses = []

        for net_name in shuffled_train_networks:
            beta = ALL_NETWORK_CONFIGS[net_name]
            G = Read(net_name + '.txt')
            G.remove_edges_from(nx.selfloop_edges(G))
            sir_path = os.path.join("SIR", net_name, f"SIR_{net_name}{beta}.txt")
            SIRDict = Read_SIR(sir_path)

            all_calculated_features, node_to_idx = calculate_network_features(G)

            features_list = [all_calculated_features[key] for key in feature_keys]
            x_numpy = np.stack(features_list, axis=1)
            scaler = MinMaxScaler()
            x_normalized = scaler.fit_transform(x_numpy)
            x = torch.tensor(x_normalized, dtype=torch.float).to(device)

            nodes_in_order = [node for node, idx in sorted(node_to_idx.items(), key=lambda item: item[1])]
            y = torch.tensor([SIRDict.get(node, 0.0) for node in nodes_in_order], dtype=torch.float).to(device)

            edge_index_list = []
            for u, v in G.edges():
                edge_index_list.append([node_to_idx[u], node_to_idx[v]])
                edge_index_list.append([node_to_idx[v], node_to_idx[u]])
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(device)

            data = Data(x=x, edge_index=edge_index, y=y)

            model.train()
            for _ in range(EPOCHS_PER_GRAPH_TRAIN):  # Train multiple epochs on this single graph
                optimizer.zero_grad()
                predictions = model(data.x, data.edge_index)
                loss = criterion(predictions, data.y)
                loss.backward()
                optimizer.step()
                current_global_epoch_losses.append(loss.item())

        print(f"  Average Loss for Global Epoch {global_epoch + 1}: {np.mean(current_global_epoch_losses):.4f}")
    print("Main training phase complete.")

    # 3. Evaluation Phase (on 8 test networks)
    print("\n--- Starting Evaluation Phase on Test Networks ---")
    test_results = {}
    model.eval()  # Set model to evaluation mode once
    with torch.no_grad():
        for net_name in test_networks_names:
            print(f"  Evaluating on test network: {net_name}")
            beta = ALL_NETWORK_CONFIGS[net_name]
            G = Read(net_name + '.txt')
            G.remove_edges_from(nx.selfloop_edges(G))
            sir_path = os.path.join("SIR", net_name, f"SIR_{net_name}{beta}.txt")
            SIRDict = Read_SIR(sir_path)

            all_calculated_features, node_to_idx = calculate_network_features(G)

            features_list = [all_calculated_features[key] for key in feature_keys]
            x_numpy = np.stack(features_list, axis=1)
            scaler = MinMaxScaler()
            x_normalized = scaler.fit_transform(x_numpy)
            x = torch.tensor(x_normalized, dtype=torch.float).to(device)

            nodes_in_order = [node for node, idx in sorted(node_to_idx.items(), key=lambda item: item[1])]
            y = torch.tensor([SIRDict.get(node, 0.0) for node in nodes_in_order], dtype=torch.float).to(device)

            edge_index_list = []
            for u, v in G.edges():
                edge_index_list.append([node_to_idx[u], node_to_idx[v]])
                edge_index_list.append([node_to_idx[v], node_to_idx[u]])
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(device)

            data = Data(x=x, edge_index=edge_index, y=y)

            # Get predictions for the entire test graph
            predictions = model(data.x, data.edge_index)

            # Calculate Kendall's Tau on the entire test graph
            tau_score = evaluate_kendall_tau(predictions, data.y)
            test_results[net_name] = tau_score
            print(f"    Kendall's Tau on {net_name}: {tau_score:.4f}")

    return test_results


# --- Run All Four Scenarios ---
if __name__ == "__main__":
    # Ensure 'SIR' directory exists and contains SIR files for all networks
    # Example: os.makedirs(os.path.join("SIR", "dolphins"), exist_ok=True)
    # You MUST ensure your SIR files are in the correct 'SIR/network_name/SIR_network_name<beta>.txt' path.

    # Scenario 1: GCN with DC + KS
    results_dc_ks = run_multi_graph_train_and_test(
        feature_combination_name="GCN with DC + KS",
        feature_keys=["Degree", "KShell"]
    )
    print("\n--- Summary Results for GCN with DC + KS ---")
    for net, tau in results_dc_ks.items():
        print(f"Network: {net.ljust(15)}, Kendall's Tau: {tau:.4f}")

    # Scenario 2: GCN with DC + KS + MAM
    results_dc_ks_mam = run_multi_graph_train_and_test(
        feature_combination_name="GCN with DC + KS + MAM",
        feature_keys=["Degree", "KShell", "MAM"]
    )
    print("\n--- Summary Results for GCN with DC + KS + MAM ---")
    for net, tau in results_dc_ks_mam.items():
        print(f"Network: {net.ljust(15)}, Kendall's Tau: {tau:.4f}")

    # Scenario 3: GCN with DC + KS + SI + NI
    results_dc_ks_si_ni = run_multi_graph_train_and_test(
        feature_combination_name="GCN with DC + KS + SI + NI",
        feature_keys=["Degree", "KShell", "SI", "NI"]
    )
    print("\n--- Summary Results for GCN with DC + KS + SI + NI ---")
    for net, tau in results_dc_ks_si_ni.items():
        print(f"Network: {net.ljust(15)}, Kendall's Tau: {tau:.4f}")

    # Scenario 4: GCN with DC + KS + SI + NI + MAM (All Features)
    results_all_features = run_multi_graph_train_and_test(
        feature_combination_name="GCN with DC + KS + SI + NI + MAM (All Features)",
        feature_keys=["Degree", "KShell", "SI", "NI", "MAM"]
    )
    print("\n--- Summary Results for GCN with All Features ---")
    for net, tau in results_all_features.items():
        print(f"Network: {net.ljust(15)}, Kendall's Tau: {tau:.4f}")