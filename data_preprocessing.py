import os
import json
import pickle
import numpy as np
import torch
import random
from tqdm import tqdm
from utils import *


def gen_model_initMAS_training_set(language_dataset, embedding_model, save_path):
    dataset = []
    for meta_data in tqdm(language_dataset, desc="Generate training data"):
        adj_matrix = meta_data["adj_matrix"]
        attacker_idxes = meta_data["attacker_idxes"]
        system_prompts = meta_data["system_prompts"]

        adj_matrix_np = np.array(adj_matrix)
        labels = np.array([1 if i in attacker_idxes else 0 for i in range(len(adj_matrix))])

        system_prompts_embedding = []
        for i in range(len(system_prompts)):
            system_prompts_embedding.append(embedding_model.encode(system_prompts[i]))
        system_prompts_embedding = np.array(system_prompts_embedding)

        # edge_embedding
        edge_index = adj_matrix_np.nonzero()
        edge_index = np.array(edge_index)

        data = {}
        data["adj_matrix"] = adj_matrix_np
        data["features"] = system_prompts_embedding
        data["labels"] = labels
        data["edge_index"] = edge_index

        dataset.append(data)

    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


def gen_model_agent_training_set(args, language_dataset, embedding_model, save_path):
    dataset = []
    for meta_data in tqdm(language_dataset, desc="Generate training data"):
        adj_matrix = meta_data["adj_matrix"]
        system_prompts = meta_data["system_prompts"]

        if args.phase == "train":
            attacker_idxes = meta_data["attacker_idxes"]
        else:
            attacker_idxes = random.sample(range(0, len(meta_data["system_prompts"])), 3)
        meta_data["attacker_idxes"] = attacker_idxes

        adj_matrix_np = np.array(adj_matrix)
        labels = np.array([1 if i in attacker_idxes else 0 for i in range(len(adj_matrix))])

        if args.dataset == "mmlu":
            normal = attack["PI"]["normal"]
            attack_data = attack["PI"]["attack"]
        elif args.dataset == "msmarco":
            normal = attack["MA"]["normal"]
            attack_data = attack["MA"]["attack"]
        elif args.dataset == "InjecAgent":
            normal = attack["TA"]["normal"]
            attack_data = attack["TA"]["attack"]
        elif args.dataset == "gsm8k":
            normal = attack["gsm8k"]["normal"]
            attack_data = attack["gsm8k"]["attack"]


        prompts_embedding = []
        for i in range(len(system_prompts)):
            if i in attacker_idxes:
                prompts_embedding.append(embedding_model.encode(system_prompts[i]+attack_data))
            else:
                prompts_embedding.append(embedding_model.encode(system_prompts[i]+normal))
        prompts_embedding = np.array(prompts_embedding)

        # edge_embedding
        edge_index = adj_matrix_np.nonzero()
        edge_index = np.array(edge_index)

        data = {}
        data["adj_matrix"] = adj_matrix_np
        data["features"] = prompts_embedding
        data["labels"] = labels
        data["edge_index"] = edge_index

        dataset.append(data)

    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


def gen_model_communication_training_set(args, language_dataset, embedding_model, save_path):
    dataset = []
    for meta_data in tqdm(language_dataset, desc="Generate training data"):
        adj_matrix = meta_data["adj_matrix"]
        system_prompts = meta_data["system_prompts"]
        communication = meta_data["communication_data"]
        communication_turn = communication[0:args.turn-1]
        communication_on_turn = communication[args.turn-1]

        communication_list = ["" for _ in range(len(system_prompts))]
        for communication in communication_turn:
            for xid, res in enumerate(communication):
                communication_list[xid] += res[1]

        if args.phase == "train":
            attacker_idxes = meta_data["attacker_idxes"]
        else:
            attacker_idxes = random.sample(range(0, len(meta_data["system_prompts"])), 3)
        meta_data["attacker_idxes"] = attacker_idxes

        adj_matrix_np = np.array(adj_matrix)
        labels = np.array([1 if i in attacker_idxes else 0 for i in range(len(adj_matrix))])

        prompts_embedding = []

        for i in range(len(system_prompts)):
            if i in attacker_idxes:
                # prompts_embedding.append(embedding_model.encode(system_prompts[i]+communication_list[i]+attack["CA"][random.randint(0, len(attack["CA"])-1)]))
                prompts_embedding.append(embedding_model.encode(system_prompts[i]+attack["CA"][random.randint(0, len(attack["CA"])-1)]))
            else:
                prompts_embedding.append(embedding_model.encode(system_prompts[i]+communication_on_turn[i][1]))
                # prompts_embedding.append(embedding_model.encode(system_prompts[i]+communication_list[i]))

        communication_embedding = []
        for i in range(len(system_prompts)):
            communication_embedding.append(embedding_model.encode(communication_list[i]))

        prompts_embedding = np.array(prompts_embedding)
        communication_embedding = np.array(communication_embedding)

        # edge_embedding
        edge_index = adj_matrix_np.nonzero()
        edge_index = np.array(edge_index)

        data = {}
        data["adj_matrix"] = adj_matrix_np
        data["features"] = prompts_embedding + communication_embedding
        data["labels"] = labels
        data["edge_index"] = edge_index

        dataset.append(data)

    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


def gen_model_communication_training_set_old(args, language_dataset, embedding_model, save_path):
    dataset = []
    for meta_data in tqdm(language_dataset, desc="Generate training data"):
        adj_matrix = meta_data["adj_matrix"]
        system_prompts = meta_data["system_prompts"]

        communication = meta_data["communication_data"]
        communication_turn = communication[0:args.turn]
        communication_list = ["" for _ in range(len(system_prompts))]
        for communication in communication_turn:
            for xid, res in enumerate(communication):
                communication_list[xid] += res[1]

        if args.phase == "test":
            attacker_idxes = random.sample(range(0, len(system_prompts)), 3)
        else:
            attacker_idxes = meta_data["attacker_idxes"]

        adj_matrix_np = np.array(adj_matrix)
        labels = np.array([1 if i in attacker_idxes else 0 for i in range(len(adj_matrix))])

        prompts_embedding = []

        # for i in range(len(system_prompts)):
        #     if i in attacker_idxes:
        #         # prompts_embedding.append(embedding_model.encode(system_prompts[i]+communication_list[i]+attack["CA"][random.randint(0, len(attack["CA"])-1)]))
        #         prompts_embedding.append(embedding_model.encode(system_prompts[i]+attack["CA"][random.randint(0, len(attack["CA"])-1)]))
        #     else:
        #         prompts_embedding.append(embedding_model.encode(system_prompts[i]))
        #         # prompts_embedding.append(embedding_model.encode(system_prompts[i]+communication_list[i]))

        for i in range(len(system_prompts)):
            if i in attacker_idxes:
                # prompts_embedding.append(embedding_model.encode(system_prompts[i]+communication_list[i]+attack["CA"][random.randint(0, len(attack["CA"])-1)]))
                prompts_embedding.append(
                    embedding_model.encode(system_prompts[i] + attack["CA"][random.randint(0, len(attack["CA"]) - 1)]))
            else:
                prompts_embedding.append(embedding_model.encode(system_prompts[i]))
                # prompts_embedding.append(embedding_model.encode(system_prompts[i]+communication_list[i]))

        communication_embedding = []
        for i in range(len(system_prompts)):
            communication_embedding.append(embedding_model.encode(communication_list[i]))

        prompts_embedding = np.array(prompts_embedding)
        communication_embedding = np.array(communication_embedding)

        # edge_embedding
        edge_index = adj_matrix_np.nonzero()
        edge_index = np.array(edge_index)

        data = {}
        data["adj_matrix"] = adj_matrix_np
        data["features"] = prompts_embedding + communication_embedding
        data["labels"] = labels
        data["edge_index"] = edge_index

        dataset.append(data)

    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiments that generate dataset")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["mmlu", "msmarco", "InjecAgent", "gsm8k"])
    parser.add_argument("--MAS_phase", type=str, default="agent", choices=["initMAS", "agent", "communication"])
    parser.add_argument("--num_attackers", type=int, default=3)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--num_nodes", type=int, default=8)
    parser.add_argument("--sparsity", type=float, default=0.5,
                        help="Sparsity of the edges (0 to 1), where higher values indicate denser graphs. 1 represents complete graph.")
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--turn", type=int, default=1)
    args = parser.parse_args()
    if args.MAS_phase == "initMAS":
        data_dir = f"./agent_graph_dataset/{args.dataset}/{args.phase}/dataset_size_{args.samples}-num_nodes_{args.num_nodes}-num_attackers_{args.num_attackers}-sparsity_{args.sparsity}_initMAS.json"
    elif args.MAS_phase == "agent" or args.MAS_phase == "communication":
        data_dir = f"./agent_graph_dataset/{args.dataset}/train/dataset_size_{args.samples}-num_nodes_{args.num_nodes}-num_attackers_{args.num_attackers}-sparsity_{args.sparsity}.json"

    save_dir = f"./ModelTrainingSet/{args.dataset}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save_path = os.path.join(save_dir, "dataset_size_12-num_nodes_8-num_attackers_3-sparsity_0.5_initMAS_test.pkl")
    if args.phase == "test":
        if args.MAS_phase == "initMAS":
            save_path = os.path.join(save_dir, f"dataset_size_{args.samples}-num_nodes_{args.num_nodes}-num_attackers_{args.num_attackers}-sparsity_{args.sparsity}_{args.MAS_phase}_test.pkl")
        elif args.MAS_phase == "agent" or args.MAS_phase == "communication":
            save_path = os.path.join(save_dir,
                                             f"dataset_size_{args.samples}-num_nodes_{args.num_nodes}-num_attackers_{args.num_attackers}-sparsity_{args.sparsity}_{args.MAS_phase}_test.pkl")
    else:
        if args.MAS_phase == "initMAS":
            save_path = os.path.join(save_dir, f"dataset_size_{args.samples}-num_nodes_{args.num_nodes}-num_attackers_{args.num_attackers}-sparsity_{args.sparsity}_{args.MAS_phase}.pkl")
        elif args.MAS_phase == "agent" or args.MAS_phase == "communication":
            save_path = os.path.join(save_dir,
                                             f"dataset_size_{args.samples}-num_nodes_{args.num_nodes}-num_attackers_{args.num_attackers}-sparsity_{args.sparsity}_{args.MAS_phase}.pkl")

    embedding_model_dir = "./Qwen/Qwen3-Embedding-0.6B"
    embedding_model = SentenceTransformer(embedding_model_dir)

    with open(data_dir, 'r') as file:
        language_dataset = json.load(file)

    if args.MAS_phase == "initMAS":
        gen_model_initMAS_training_set(language_dataset, embedding_model, save_path)
    elif args.MAS_phase == "agent":
        gen_model_agent_training_set(args, language_dataset, embedding_model, save_path)
    elif args.MAS_phase == "communication":
        gen_model_communication_training_set(args, language_dataset, embedding_model, save_path)

    with open(save_path, "rb") as f:
        loaded_dataset = pickle.load(f)
    print(len(loaded_dataset))
    print(loaded_dataset[0])