import argparse
import os
from tqdm import tqdm
from agent.data import AgentGraphDataset
from torch.utils.data import DataLoader

import torch
import json
import numpy as np
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model import G_ADM, reconstruction_loss2
from einops import rearrange
from datetime import datetime
import random


def train(model: G_ADM, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for x, y, adj_matrix, num_nodes in train_loader:
        x, y, adj_matrix, num_nodes = x.to(device), y.to(device), adj_matrix.to(device), num_nodes
        # x, y, edge_index, edge_attr = data.x.to(device), data.y.to(device), data.edge_index.to(
        #     device), data.edge_attr.to(device)
        optimizer.zero_grad()
        x_hat, a_hat, z, g_s, g_s_hat = model(x, adj_matrix=adj_matrix)
        # loss, node_score, attr_err, struct_err, graph_score = reconstruction_loss(x, adj_matrix, x_hat, a_hat, g_s, g_s_hat)
        loss, node_score, attr_err, _, graph_score = reconstruction_loss2(x, adj_matrix, x_hat, a_hat, g_s, g_s_hat)
        # loss = criterion(outputs, y.float().unsqueeze(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # predicted = (torch.sigmoid(outputs) >= 0.5).squeeze()
        # total += y.size(0)
        # correct += (predicted == y).sum().item()

    avg_loss = running_loss / len(train_loader)
    # accuracy = 100 * correct / total

    return avg_loss



def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiments to train GAT")

    parser.add_argument("--dataset_path", type=str,
                        help="Save path of the dataset")
    parser.add_argument("--dataset", type=str, default="msmarco", choices=["mmlu", "msmarco", "InjecAgent", "gsm8k"])
    parser.add_argument("--training_phase", type=str, default="agent", choices=["initMAS", "agent", "communication"])

    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--num_nodes", type=int, default=8)
    parser.add_argument("--num_attackers", type=int, default=3)
    parser.add_argument("--sparsity", type=float, default=0.5,
                        help="Sparsity of the edges (0 to 1), where higher values indicate denser graphs. 1 represents complete graph.")

    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001) #0.001
    parser.add_argument("--weight_decay", type=float, default=0.0002) #0.0002
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--save_dir", type=str, default="./checkpoint")

    args = parser.parse_args()
    args.dataset_path = f"./ModelTrainingSet/{args.dataset}/dataset_size_{args.samples}-num_nodes_{args.num_nodes}-num_attackers_{args.num_attackers}-sparsity_{args.sparsity}_{args.training_phase}.pkl"
    normalized_path = os.path.normpath(args.dataset_path)
    parts = normalized_path.split(os.sep)
    dataset = parts[-2]
    args.save_dir = os.path.join(args.save_dir, dataset)
    filename = f"hiddim_{args.hidden_dim}-heads_{args.num_heads}-layers_{args.num_layers}-epochs_{args.epochs}-lr_{args.lr}-dropout_{args.dropout}-wd_{args.weight_decay}_{args.training_phase}"

    args.save_dir = os.path.join(args.save_dir, filename)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_path = os.path.join(args.save_dir, "G_ADM.pth")
    # args.save_path = os.path.join(args.save_dir, filename)

    return args


@torch.no_grad()
def collect_train_node_scores(
    model,
    trainloader,
    attr_weight: float = 0.5,
    graph_weight: float = 0.5,
    struct_weight: float = 0.1,
    batch_size: int = 64,
    device: str = "cpu",
):
    model.eval()
    model = model.to(device)

    all_scores = []
    graph_scores = []
    for x, y, adj_matrix, num_nodes in trainloader:
        x, y, adj_matrix, num_nodes = x.to(device), y.to(device), adj_matrix.to(device), num_nodes

        x_hat, a_hat, z, g_s, g_s_hat = model(x, adj_matrix=adj_matrix)
        _, node_score, _, _, graph_score = reconstruction_loss2(
            x, adj_matrix, x_hat, a_hat, g_s, g_s_hat,
            attr_weight=attr_weight,
            graph_weight=graph_weight
        )
        all_scores.append(node_score.cpu().numpy())
        graph_scores.append(graph_score.cpu().numpy())

    all_scores = np.concatenate(all_scores, axis=0)   # [B, N]
    graph_scores = np.concatenate(graph_scores, axis=0)   # [B]
    flat_scores = all_scores.reshape(-1)              # [B*N]

    return flat_scores, graph_scores


@torch.no_grad()
def detect_anomalous_agents(
    model,
    dataloder,
    threshold: float,
    threshold_graph: float,
    attr_weight: float = 0.5,
    struct_weight: float = 0.1,
    graph_weight: float = 0.5,
    device: str = "cpu",
):
    model.eval()
    model = model.to(device)
    all_y = []
    all_pred = []
    for x, y, adj_matrix, num_nodes in dataloder:
        x, y, adj_matrix, num_nodes = x.to(device), y.to(device), adj_matrix.to(device), num_nodes

        x_hat, a_hat, z, g_s, g_s_hat = model(x, adj_matrix=adj_matrix)

        _, node_score, attr_err, _, graph_score = reconstruction_loss2(
            x, adj_matrix, x_hat, a_hat, g_s, g_s_hat,
            attr_weight=attr_weight,
            graph_weight=graph_weight
        )
        # print(node_score)
        scores = node_score[0].cpu().numpy()  # [N]

        if graph_score > threshold_graph or np.max(scores) > threshold:
            flags = scores > threshold               # [N]
        else:
            flags = np.zeros(node_score.size(1), dtype=bool)
        print(graph_score)
        print(threshold_graph)


        print(threshold)
        print(y)
        print(flags)
        print(scores)
        print()
        y_true = y[0].detach().cpu().numpy().astype(bool)  # [N]

        all_y.append(y_true)
        all_pred.append(flags)

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)

    TP = ((all_y == 1) & (all_pred == 1)).sum()
    FN = ((all_y == 1) & (all_pred == 0)).sum()

    FP = ((all_y == 0) & (all_pred == 1)).sum()

    acc = (all_y == all_pred).mean()
    recall = TP / (TP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("precision =", precision)
    print("recall =", recall)
    print("f1 =", f1)
    print("overall acc =", acc)
    exit(0)
    return {
        "scores": scores,
        "is_anomaly": flags,
        "attr_err": attr_err[0].cpu().numpy(),
        "struct_err": struct_err[0].cpu().numpy(),
    }


def fit_threshold_mad(train_scores: np.ndarray, k: float = 3.0) -> float:
    med = np.median(train_scores)
    mad = np.median(np.abs(train_scores - med)) + 1e-8
    return float(med + k * 1.4826 * mad)


def fit_threshold(train_scores: np.ndarray, quantile: float = 0.98) -> float:
    if not 0 < quantile < 1:
        raise ValueError("quantile in (0, 1)")
    return float(np.quantile(train_scores, quantile))


def random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    random_seed(42) #42
    args = parse_arguments()

    train_dataset = AgentGraphDataset(args.dataset_path, phase="train", training_phase=args.training_phase)
    val_dataset = AgentGraphDataset(args.dataset_path, phase="val", training_phase=args.training_phase)

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset)
    example = train_dataset[0]

    in_channels = example[0].size(1)
    # edge_dim = example.edge_attr.size()[1:]
    edge_dim = in_channels

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    gnn = G_ADM(in_channels, args.hidden_dim, out_channels=1, heads=args.num_heads, num_layers=args.num_layers,
                edge_dim=edge_dim)
    gnn.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    best_acc = 0.0

    for i in range(args.epochs):
        train_loss = train(gnn, trainloader, criterion, optimizer, device=device)
        # test_loss, test_acc = test(gnn, testloader, criterion, device=device)
        scheduler.step()

        print(f"Epoch {i}/{args.epochs} || Training Loss: {train_loss:.4f}")

    # torch.save(gnn.state_dict(), args.save_path)  # 保存模型

    flat_scores, graph_scores = collect_train_node_scores(gnn, trainloader)
    print(flat_scores)
    # threshold = fit_threshold(flat_scores, quantile=0.98)
    threshold = fit_threshold_mad(flat_scores, k = 6) #3 #PI C -2.0 # MA Init 3 agent 3 c 0.6 TA Init 3 agent 3 c 0.6

    threshold_graph = fit_threshold_mad(graph_scores, k = 6)
    threshold_dict = {"threshold":threshold, "threshold_graph":threshold_graph}

    path = os.path.join(args.save_dir, "threshold.json")
    with open(path, "w") as f:
        json.dump(threshold_dict, f)

    # args.dataset_path = f"./ModelTrainingSet/{args.dataset}/dataset_size_12-num_nodes_{args.num_nodes}-num_attackers_{args.num_attackers}-sparsity_{args.sparsity}_{args.training_phase}.pkl"
    args.dataset_path = f"./ModelTrainingSet/{args.dataset}/dataset_size_{args.samples}-num_nodes_{args.num_nodes}-num_attackers_{args.num_attackers}-sparsity_{args.sparsity}_{args.training_phase}.pkl"

    test_dataset = AgentGraphDataset(args.dataset_path.replace(".pkl", "")+"_test.pkl", phase="test", training_phase=args.training_phase)

    testloader = DataLoader(test_dataset)
    detect_anomalous_agents(gnn, testloader, threshold, threshold_graph)


if __name__ == "__main__":
    main()
