import torch
import pickle
from torch.utils.data import Dataset

class AgentGraphDataset(Dataset): 
    def __init__(self, root, transform=None, phase="train", training_phase="initMAS"):
        super().__init__()

        with open(root, "rb") as f:
            origin_dataset = pickle.load(f)
        origin_dataset_len = len(origin_dataset)

        self.training_phase = training_phase
        if phase == "train": 
            # self.dataset = origin_dataset[:int(origin_dataset_len*0.8)]
            self.dataset = origin_dataset[:40]
        elif phase == "val":
            self.dataset = origin_dataset[int(origin_dataset_len*0.8):]
        elif phase == "test":
            self.dataset = origin_dataset
        else:
            raise Exception(f"Unknown phase {phase}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        origin_data = self.dataset[idx]
        adj_matrix = torch.tensor(origin_data["adj_matrix"])

        x = torch.tensor(origin_data["features"])
        y = torch.tensor(origin_data["labels"], dtype=torch.long)

        num_nodes = len(x)

        return x, y, adj_matrix, num_nodes
    