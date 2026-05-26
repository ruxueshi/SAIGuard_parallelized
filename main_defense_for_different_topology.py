import os
from model.model import G_ADM, reconstruction_loss
from agent.agents import AgentGraphWithDefense, AgentGraph
from tqdm import tqdm
import json
import random
import numpy as np
import torch
from utils import get_sentence_embedding
from einops import rearrange
import argparse
from datetime import datetime
import asyncio
import copy
import time
from utils import *
from sentence_transformers import SentenceTransformer

async def test_initMAS_agent(ag: AgentGraph, qa_data, num_dialogue_turns):
    communication_data = []
    initial_responses = await ag.afirst_generate(qa_data)
    communication_data.append(initial_responses)

    for _ in range(num_dialogue_turns):
        responses = await ag.are_generate()
        communication_data.append(responses)

    return communication_data

async def test_initMAS_agent_ma(ag: AgentGraph, query, context, num_dialogue_turns):
    communication_data = []
    initial_responses = await ag.afirst_generate_ma(query, context)
    communication_data.append(initial_responses)

    for _ in range(num_dialogue_turns):
        responses = await ag.are_generate_ma()
        communication_data.append(responses)

    return communication_data

async def test_initMAS_agent_ta(ag: AgentGraph, case, num_dialogue_turns):
    communication_data = []
    initial_responses = await ag.afirst_generate_ta(case)
    communication_data.append(initial_responses)

    for _ in range(num_dialogue_turns):
        responses = await ag.are_generate_ta()
        communication_data.append(responses)

    return communication_data


async def test_initMAS_agent_gsm8k(ag: AgentGraph, qa_data, num_dialogue_turns):
    communication_data = []
    initial_responses = await ag.afirst_generate_gsm8k(qa_data)
    communication_data.append(initial_responses)

    for _ in range(num_dialogue_turns):
        responses = await ag.are_generate_gsm8k()
        communication_data.append(responses)

    return communication_data


async def test_communication_attack(ag: AgentGraph, qa_data, num_dialogue_turns, attacker_idxes, attack_turn):
    communication_data = []
    if attack_turn == 1:
        initial_responses = await ag.afirst_generate_attack(qa_data, attacker_idxes)
    else:
        initial_responses = await ag.afirst_generate(qa_data)

    communication_data.append(initial_responses)

    for i in range(num_dialogue_turns):
        if i+1 == attack_turn-1:
            responses = await ag.are_generate_continue_attack(attacker_idxes)
        else:
            responses = await ag.are_generate_continue()
        communication_data.append(responses)

    return communication_data

async def test_communication_attack_ma(ag: AgentGraph, query, context, num_dialogue_turns, attacker_idxes, attack_turn):
    communication_data = []
    if attack_turn == 1:
        initial_responses = await ag.afirst_generate_attack_ma(query, context, attacker_idxes)
    else:
        initial_responses = await ag.afirst_generate_ma(query, context)

    communication_data.append(initial_responses)

    for i in range(num_dialogue_turns):
        if i+1 == attack_turn-1:
            responses = await ag.are_generate_continue_attack_ma(attacker_idxes)
        else:
            responses = await ag.are_generate_continue_ma()
        communication_data.append(responses)

    return communication_data

async def test_communication_attack_ta(ag: AgentGraph, case, num_dialogue_turns, attacker_idxes, attack_turn):
    communication_data = []
    if attack_turn == 1:
        initial_responses = await ag.afirst_generate_attack_ta(case, attacker_idxes)
    else:
        initial_responses = await ag.afirst_generate_ta(case)

    communication_data.append(initial_responses)

    for i in range(num_dialogue_turns):
        if i+1 == attack_turn-1:
            responses = await ag.are_generate_continue_attack_ta(attacker_idxes)
        else:
            responses = await ag.are_generate_continue_ta()
        communication_data.append(responses)

    return communication_data


async def test_communication_attack_defense_old(ag: AgentGraph, qa_data, num_dialogue_turns, attacker_idxes, attack_turn, embedding_model, gnn, adj_m, threshold_graph, threshold):
    communication_data = []
    if attack_turn == 1:
        initial_responses = await ag.afirst_generate_attack(qa_data, attacker_idxes)
    else:
        initial_responses = await ag.afirst_generate(qa_data)

    system_prompts = ag.system_prompts

    prompts_embedding = []
    for i in range(len(system_prompts)):
        prompts_embedding.append(embedding_model.encode(system_prompts[i] + initial_responses[i][1]))
    prompts_embedding = np.array(prompts_embedding)

    communication_list = ["" for _ in range(len(system_prompts))]
    for communication in communication_data:
        for xid, res in enumerate(communication):
            communication_list[xid] += res[1]

    communication_embedding = []
    for i in range(len(system_prompts)):
        communication_embedding.append(embedding_model.encode(communication_list[i]))

    communication_embedding = np.array(communication_embedding)

    feature_embeding = prompts_embedding + communication_embedding

    with torch.no_grad():
        x_hat, a_hat, z, g_s, g_s_hat = gnn(torch.tensor(feature_embeding).unsqueeze(0),
                                            torch.tensor(adj_m).unsqueeze(0))

    loss, node_score, attr_err, struct_err, graph_score = reconstruction_loss(
        torch.tensor(feature_embeding).unsqueeze(0), torch.tensor(adj_m).unsqueeze(0), x_hat, a_hat,
        g_s, g_s_hat)

    scores = node_score[0].numpy()  # [N]

    if graph_score > threshold_graph or np.max(scores) > threshold:
        flags = scores > threshold  # [N]
    else:
        flags = np.zeros(node_score.size(1), dtype=bool)

    detect_idx = np.where(flags)[0]
    # print(initial_responses)
    tasks = []
    for idx in detect_idx:
        ag.agents[idx].memory.pop()
        ag.agents[idx].memory.pop()
        tasks.append(asyncio.create_task(ag.afirst_generate_agent(idx, qa_data)))

    agent_responses = await asyncio.gather(*tasks)
    for response in agent_responses:
        initial_responses[response[0]] = (int(response[0]), response[1])
    communication_data.append(initial_responses)

    for i in range(num_dialogue_turns):
        if i+1 == attack_turn-1:
            responses = await ag.are_generate_continue_attack(attacker_idxes)
        else:
            responses = await ag.are_generate_continue()

        # prompts_embedding = []
        # for i in range(len(system_prompts)):
        #     prompts_embedding.append(embedding_model.encode(system_prompts[i] + responses[i][1]))
        # prompts_embedding = np.array(prompts_embedding)
        #
        # communication_list = ["" for _ in range(len(system_prompts))]
        # for communication in communication_data:
        #     for xid, res in enumerate(communication):
        #         communication_list[xid] += res[1]
        #
        # communication_embedding = []
        # for i in range(len(system_prompts)):
        #     communication_embedding.append(embedding_model.encode(communication_list[i]))
        #
        # communication_embedding = np.array(communication_embedding)
        #
        # feature_embeding = prompts_embedding + communication_embedding
        #
        # with torch.no_grad():
        #     x_hat, a_hat, z, g_s, g_s_hat = gnn(torch.tensor(feature_embeding).unsqueeze(0),
        #                                         torch.tensor(adj_m).unsqueeze(0))
        #
        # loss, node_score, attr_err, struct_err, graph_score = reconstruction_loss(
        #     torch.tensor(feature_embeding).unsqueeze(0), torch.tensor(adj_m).unsqueeze(0), x_hat, a_hat,
        #     g_s, g_s_hat)
        #
        # scores = node_score[0].numpy()  # [N]
        #
        # if graph_score > threshold_graph:
        #     flags = scores > threshold  # [N]
        # else:
        #     flags = np.zeros(node_score.size(1), dtype=bool)
        #
        # detect_idx = np.where(flags)[0]
        #
        # for idx in detect_idx:
        #     ag.agents[idx].memory.pop()
        #     ag.agents[idx].memory.pop()
        #
        # agent_responses = await ag.are_generate_continue_defense(detect_idx)
        #
        # for response in agent_responses:
        #     responses[response[0]] = (int(response[0]), response[1])

        communication_data.append(responses)

    return communication_data


async def test_communication_attack_defense(ag: AgentGraph, qa_data, num_dialogue_turns, attacker_idxes, attack_turn, embedding_model, gnn, adj_m, threshold_graph, threshold):
    communication_data = []
    # if attack_turn == 1:
    #     initial_responses = await ag.afirst_generate_attack(qa_data, attacker_idxes)
    # else:
    initial_responses = await ag.afirst_generate(qa_data)

    # system_prompts = ag.system_prompts
    #     #
    #     # prompts_embedding = []
    #     # for i in range(len(system_prompts)):
    #     #     prompts_embedding.append(embedding_model.encode(system_prompts[i] + initial_responses[i][1]))
    #     # prompts_embedding = np.array(prompts_embedding)
    #     #
    #     # communication_list = ["" for _ in range(len(system_prompts))]
    #     # for communication in communication_data:
    #     #     for xid, res in enumerate(communication):
    #     #         communication_list[xid] += res[1]
    #     #
    #     # communication_embedding = []
    #     # for i in range(len(system_prompts)):
    #     #     communication_embedding.append(embedding_model.encode(communication_list[i]))
    #     #
    #     # communication_embedding = np.array(communication_embedding)
    #     #
    #     # feature_embeding = prompts_embedding + communication_embedding
    #     #
    #     # with torch.no_grad():
    #     #     x_hat, a_hat, z, g_s, g_s_hat = gnn(torch.tensor(feature_embeding).unsqueeze(0),
    #     #                                         torch.tensor(adj_m).unsqueeze(0))
    #     #
    #     # loss, node_score, attr_err, struct_err, graph_score = reconstruction_loss(
    #     #     torch.tensor(feature_embeding).unsqueeze(0), torch.tensor(adj_m).unsqueeze(0), x_hat, a_hat,
    #     #     g_s, g_s_hat)
    #     #
    #     # scores = node_score[0].numpy()  # [N]
    #     #
    #     # if graph_score > threshold_graph or np.max(scores) > threshold:
    #     #     flags = scores > threshold  # [N]
    #     # else:
    #     #     flags = np.zeros(node_score.size(1), dtype=bool)
    #     #
    #     # detect_idx = np.where(flags)[0]
    #     # # print(initial_responses)
    #     # tasks = []
    #     # for idx in detect_idx:
    #     #     ag.agents[idx].memory.pop()
    #     #     ag.agents[idx].memory.pop()
    #     #     tasks.append(asyncio.create_task(ag.afirst_generate_agent(idx, qa_data)))
    #     #
    #     # agent_responses = await asyncio.gather(*tasks)
    #     # for response in agent_responses:
    #     #     initial_responses[response[0]] = (int(response[0]), response[1])
    communication_data.append(initial_responses)

    for i in range(num_dialogue_turns):
        # if i+1 == attack_turn-1:
        #     responses = await ag.are_generate_continue_attack(attacker_idxes)
        # else:
        responses = await ag.are_generate_continue()

        # prompts_embedding = []
        # for i in range(len(system_prompts)):
        #     prompts_embedding.append(embedding_model.encode(system_prompts[i] + responses[i][1]))
        # prompts_embedding = np.array(prompts_embedding)
        #
        # communication_list = ["" for _ in range(len(system_prompts))]
        # for communication in communication_data:
        #     for xid, res in enumerate(communication):
        #         communication_list[xid] += res[1]
        #
        # communication_embedding = []
        # for i in range(len(system_prompts)):
        #     communication_embedding.append(embedding_model.encode(communication_list[i]))
        #
        # communication_embedding = np.array(communication_embedding)
        #
        # feature_embeding = prompts_embedding + communication_embedding
        #
        # with torch.no_grad():
        #     x_hat, a_hat, z, g_s, g_s_hat = gnn(torch.tensor(feature_embeding).unsqueeze(0),
        #                                         torch.tensor(adj_m).unsqueeze(0))
        #
        # loss, node_score, attr_err, struct_err, graph_score = reconstruction_loss(
        #     torch.tensor(feature_embeding).unsqueeze(0), torch.tensor(adj_m).unsqueeze(0), x_hat, a_hat,
        #     g_s, g_s_hat)
        #
        # scores = node_score[0].numpy()  # [N]
        #
        # if graph_score > threshold_graph:
        #     flags = scores > threshold  # [N]
        # else:
        #     flags = np.zeros(node_score.size(1), dtype=bool)
        #
        # detect_idx = np.where(flags)[0]
        #
        # for idx in detect_idx:
        #     ag.agents[idx].memory.pop()
        #     ag.agents[idx].memory.pop()
        #
        # agent_responses = await ag.are_generate_continue_defense(detect_idx)
        #
        # for response in agent_responses:
        #     responses[response[0]] = (int(response[0]), response[1])

        communication_data.append(responses)

    return communication_data

async def test_communication_attack_defense_ma(ag: AgentGraph, query, context, num_dialogue_turns, attacker_idxes, attack_turn, embedding_model, gnn, adj_m, threshold_graph, threshold):
    communication_data = []
    if attack_turn == 1:
        initial_responses = await ag.afirst_generate_attack_ma(query, context, attacker_idxes)
    else:
        initial_responses = await ag.afirst_generate_ma(query, context)

    system_prompts = ag.system_prompts

    prompts_embedding = []
    for i in range(len(system_prompts)):
        prompts_embedding.append(embedding_model.encode(system_prompts[i] + initial_responses[i][1]))
    prompts_embedding = np.array(prompts_embedding)

    communication_list = ["" for _ in range(len(system_prompts))]
    for communication in communication_data:
        for xid, res in enumerate(communication):
            communication_list[xid] += res[1]

    communication_embedding = []
    for i in range(len(system_prompts)):
        communication_embedding.append(embedding_model.encode(communication_list[i]))

    communication_embedding = np.array(communication_embedding)

    feature_embeding = prompts_embedding + communication_embedding

    with torch.no_grad():
        x_hat, a_hat, z, g_s, g_s_hat = gnn(torch.tensor(feature_embeding).unsqueeze(0),
                                            torch.tensor(adj_m).unsqueeze(0))

    loss, node_score, attr_err, struct_err, graph_score = reconstruction_loss(
        torch.tensor(feature_embeding).unsqueeze(0), torch.tensor(adj_m).unsqueeze(0), x_hat, a_hat,
        g_s, g_s_hat)

    scores = node_score[0].numpy()  # [N]

    if graph_score > threshold_graph or np.max(scores) > threshold:
        flags = scores > threshold  # [N]
    else:
        flags = np.zeros(node_score.size(1), dtype=bool)

    detect_idx = np.where(flags)[0]
    # print(initial_responses)
    tasks = []
    for idx in detect_idx:
        ag.agents[idx].memory.pop()
        ag.agents[idx].memory.pop()
        tasks.append(asyncio.create_task(ag.afirst_generate_agent_ma(idx, query, context)))

    agent_responses = await asyncio.gather(*tasks)
    for response in agent_responses:
        initial_responses[response[0]] = (int(response[0]), response[1])

    communication_data.append(initial_responses)

    for i in range(num_dialogue_turns):
        if i+1 == attack_turn-1:
            responses = await ag.are_generate_continue_attack_ma(attacker_idxes)
        else:
            responses = await ag.are_generate_continue_ma()

        prompts_embedding = []
        for i in range(len(system_prompts)):
            prompts_embedding.append(embedding_model.encode(system_prompts[i] + responses[i][1]))
        prompts_embedding = np.array(prompts_embedding)

        communication_list = ["" for _ in range(len(system_prompts))]
        for communication in communication_data:
            for xid, res in enumerate(communication):
                communication_list[xid] += res[1]

        communication_embedding = []
        for i in range(len(system_prompts)):
            communication_embedding.append(embedding_model.encode(communication_list[i]))

        communication_embedding = np.array(communication_embedding)

        feature_embeding = prompts_embedding + communication_embedding

        with torch.no_grad():
            x_hat, a_hat, z, g_s, g_s_hat = gnn(torch.tensor(feature_embeding).unsqueeze(0),
                                                torch.tensor(adj_m).unsqueeze(0))

        loss, node_score, attr_err, struct_err, graph_score = reconstruction_loss(
            torch.tensor(feature_embeding).unsqueeze(0), torch.tensor(adj_m).unsqueeze(0), x_hat, a_hat,
            g_s, g_s_hat)

        scores = node_score[0].numpy()  # [N]

        if graph_score > threshold_graph or np.max(scores) > threshold:
            flags = scores > threshold  # [N]
        else:
            flags = np.zeros(node_score.size(1), dtype=bool)

        detect_idx = np.where(flags)[0]

        for idx in detect_idx:
            ag.agents[idx].memory.pop()
            ag.agents[idx].memory.pop()

        agent_responses = await ag.are_generate_continue_defense_ma(detect_idx)

        for response in agent_responses:
            responses[response[0]] = (int(response[0]), response[1])

        communication_data.append(responses)

    return communication_data

async def test_communication_attack_defense_ta(ag: AgentGraph, case, num_dialogue_turns, attacker_idxes, attack_turn, embedding_model, gnn, adj_m, threshold_graph, threshold):
    communication_data = []
    user_instruction, available_tool_names, available_tool_descriptions, agent_scratchpad = case
    if attack_turn == 1:
        initial_responses = await ag.afirst_generate_attack_ta(case, attacker_idxes)
    else:
        initial_responses = await ag.afirst_generate_ta(case)

    system_prompts = ag.system_prompts

    prompts_embedding = []
    for i in range(len(system_prompts)):
        prompts_embedding.append(embedding_model.encode(system_prompts[i] + initial_responses[i][1]))
    prompts_embedding = np.array(prompts_embedding)

    communication_list = ["" for _ in range(len(system_prompts))]
    for communication in communication_data:
        for xid, res in enumerate(communication):
            communication_list[xid] += res[1]

    communication_embedding = []
    for i in range(len(system_prompts)):
        communication_embedding.append(embedding_model.encode(communication_list[i]))

    communication_embedding = np.array(communication_embedding)

    feature_embeding = prompts_embedding + communication_embedding

    with torch.no_grad():
        x_hat, a_hat, z, g_s, g_s_hat = gnn(torch.tensor(feature_embeding).unsqueeze(0),
                                            torch.tensor(adj_m).unsqueeze(0))

    loss, node_score, attr_err, struct_err, graph_score = reconstruction_loss(
        torch.tensor(feature_embeding).unsqueeze(0), torch.tensor(adj_m).unsqueeze(0), x_hat, a_hat,
        g_s, g_s_hat)

    scores = node_score[0].numpy()  # [N]

    if graph_score > threshold_graph or np.max(scores) > threshold:
        flags = scores > threshold  # [N]
    else:
        flags = np.zeros(node_score.size(1), dtype=bool)

    detect_idx = np.where(flags)[0]
    # print(initial_responses)
    tasks = []
    for idx in detect_idx:
        ag.agents[idx].memory.pop()
        ag.agents[idx].memory.pop()
        tasks.append(asyncio.create_task(ag.afirst_generate_agent_ta(idx, USER_PROMPT, ATTACKER_PROMPT, user_instruction, available_tool_names, available_tool_descriptions, agent_scratchpad )))

    agent_responses = await asyncio.gather(*tasks)
    for response in agent_responses:
        initial_responses[response[0]] = (int(response[0]), response[1])

    communication_data.append(initial_responses)

    for i in range(num_dialogue_turns):
        if i+1 == attack_turn-1:
            responses = await ag.are_generate_continue_attack_ta(attacker_idxes)
        else:
            responses = await ag.are_generate_continue_ta()

        prompts_embedding = []
        for i in range(len(system_prompts)):
            prompts_embedding.append(embedding_model.encode(system_prompts[i] + responses[i][1]))
        prompts_embedding = np.array(prompts_embedding)

        communication_list = ["" for _ in range(len(system_prompts))]
        for communication in communication_data:
            for xid, res in enumerate(communication):
                communication_list[xid] += res[1]

        communication_embedding = []
        for i in range(len(system_prompts)):
            communication_embedding.append(embedding_model.encode(communication_list[i]))

        communication_embedding = np.array(communication_embedding)

        feature_embeding = prompts_embedding + communication_embedding

        with torch.no_grad():
            x_hat, a_hat, z, g_s, g_s_hat = gnn(torch.tensor(feature_embeding).unsqueeze(0),
                                                torch.tensor(adj_m).unsqueeze(0))

        loss, node_score, attr_err, struct_err, graph_score = reconstruction_loss(
            torch.tensor(feature_embeding).unsqueeze(0), torch.tensor(adj_m).unsqueeze(0), x_hat, a_hat,
            g_s, g_s_hat)

        scores = node_score[0].numpy()  # [N]

        if graph_score > threshold_graph:
            flags = scores > threshold  # [N]
        else:
            flags = np.zeros(node_score.size(1), dtype=bool)

        detect_idx = np.where(flags)[0]

        for idx in detect_idx:
            ag.agents[idx].memory.pop()
            ag.agents[idx].memory.pop()

        agent_responses = await ag.are_generate_continue_defense_ta(detect_idx)

        for response in agent_responses:
            responses[response[0]] = (int(response[0]), response[1])

        communication_data.append(responses)

    return communication_data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiments to train GAT")

    parser.add_argument("--dataset_path", type=str,
                        help="Save path of the dataset")
    parser.add_argument("--dataset", type=str, default="InjecAgent", choices=["mmlu", "msmarco", "InjecAgent", "gsm8k"])
    parser.add_argument("--training_phase", type=str, default="agent", choices=["initMAS", "agent", "communication"])
    parser.add_argument("--graph_type", type=str, default="chain", choices=["random", "chain", "tree", "star"])
    parser.add_argument("--gnn_checkpoint_path", type=str, default="")
    parser.add_argument("--test_again", type=bool, default=False)
    parser.add_argument("--threshold_path", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="./result")
    parser.add_argument("--model_type", type=str, default="gpt-4o-mini")
    parser.add_argument("--samples", type=int, default=40)
    # parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--num_nodes", type=int, default=8)
    parser.add_argument("--num_attackers", type=int, default=3)
    parser.add_argument("--turn", type=int, default=1)
    parser.add_argument("--sparsity", type=float, default=0.5,
                        help="Sparsity of the edges (0 to 1), where higher values indicate denser graphs. 1 represents complete graph.")


    args = parser.parse_args()
    args.gnn_checkpoint_path = f"./checkpoint/{args.dataset}/hiddim_1024-heads_8-layers_2-epochs_20-lr_0.001-dropout_0.2-wd_0.0002_{args.training_phase}/G_ADM.pth"
    args.threshold_path = f"./checkpoint/{args.dataset}/hiddim_1024-heads_8-layers_2-epochs_20-lr_0.001-dropout_0.2-wd_0.0002_{args.training_phase}/threshold.json"

    if args.training_phase == "initMAS":
        args.dataset_path = f"./agent_graph_dataset/{args.dataset}/test/dataset_size_{args.samples}-num_nodes_{args.num_nodes}-num_attackers_{args.num_attackers}-sparsity_{args.sparsity}_initMAS.json"
    else:
        args.dataset_path = f"./agent_graph_dataset/{args.dataset}/train/dataset_size_{args.samples}-num_nodes_{args.num_nodes}-num_attackers_{args.num_attackers}-sparsity_{args.sparsity}.json"

    args.save_dir = os.path.join(args.save_dir, args.dataset, args.graph_type+f"_{args.training_phase}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_no_defense = f"no_defense-model_type_{args.model_type}.json"
    filename_defense = f"defense-model_type_{args.model_type}.json"
    args.save_path_no_defense = os.path.join(args.save_dir, filename_no_defense)
    args.save_path_with_defense = os.path.join(args.save_dir, filename_defense)

    return args


def random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


async def main():
    random_seed(42)
    args = parse_arguments()
    filepath = args.dataset_path
    graph_type = args.graph_type
    with open(filepath, "r") as f:
        dataset = json.load(f)
    if args.training_phase == "initMAS":
        num_dialogue_turns = 3
    else:
        num_dialogue_turns = len(dataset[0]["communication_data"]) - 1

    gnn = G_ADM(in_channels=1024, hidden_channels=1024, out_channels=1, heads=8) #bert 384
    embedding_model = SentenceTransformer("./base_model/Qwen/Qwen3-Embedding-0.6B")
    gnn.eval()
    state_dict = torch.load(args.gnn_checkpoint_path, map_location=torch.device('cpu'))
    gnn.load_state_dict(state_dict)

    with open(args.threshold_path, "r") as f:
        threshold_dict = json.load(f)
    threshold = threshold_dict["threshold"]
    threshold_graph = threshold_dict["threshold_graph"]

    final_dataset_nd = []
    final_dataset_wd = []
    for d in tqdm(dataset):
        if graph_type == "random":
            adj_m = np.array(d["adj_matrix"])
        elif graph_type in ["chain", "tree", "star"]:
            adj_m = get_adj_matrix(graph_type, len(d["adj_matrix"]))
        else:
            raise Exception(f"Unknown graph type: {graph_type}! Can only be one of [random, chain, tree, star]")

        # attacker_idxes = d["attacker_idxes"]
        if args.training_phase == "initMAS":
            attacker_idxes = d["attacker_idxes"]
        else:
            attacker_idxes = random.sample(range(0, len(d["system_prompts"])), 3)
        d["attacker_idxes"] = attacker_idxes

        d_nd = copy.deepcopy(d)
        d_wd = copy.deepcopy(d)
        # attacker_idxes = d["attacker_idxes"]
        d_nd_system_prompts = d_nd["system_prompts"]
        d_wd_system_prompts = d_wd["system_prompts"]

        if args.dataset == "mmlu":
            qa_data_origin = d["question"], d["correct_answer"], d["wrong_answer"]
            wrong_answer = random.choice(qa_data_origin[2]) if qa_data_origin[2] else None
            d_nd["wrong_answer"] = wrong_answer
            d_wd["wrong_answer"] = wrong_answer
            qa_data = (qa_data_origin[0], qa_data_origin[1], wrong_answer)
        elif args.dataset == "msmarco":
            query = d["query"]
            context = d["adv_texts"]
        elif args.dataset == "InjecAgent":
            user_instruction = d["user_instruction"]
            available_tool_names = d["available_tool_names"]
            available_tool_descriptions = d["available_tool_descriptions"]
            agent_scratchpad = d["agent_scratchpad"]
            case = (user_instruction, available_tool_names, available_tool_descriptions, agent_scratchpad)
        elif args.dataset == "gsm8k":
            qa_data_origin = d["question"], d["correct_answer"], d["wrong_answer"]
            wrong_answer = random.choice(qa_data_origin[2]) if qa_data_origin[2] else None
            qa_data = (qa_data_origin[0], qa_data_origin[1], wrong_answer)
            d_nd["wrong_answer"] = wrong_answer
            d_wd["wrong_answer"] = wrong_answer
        labels = np.array([1 if i in attacker_idxes else 0 for i in range(len(adj_m))])

        try:
        # if True:
            system_prompts_embedding = []
            if args.training_phase == "initMAS":
                if args.test_again:
                    agnd = AgentGraph(adj_m, d_nd_system_prompts, [],
                                      model_type=args.model_type)  # agent graph no defense
                    if args.dataset == "mmlu":
                        communication_data_no_defense = await test_initMAS_agent(agnd, qa_data, num_dialogue_turns)
                    elif args.dataset == "msmarco":
                        communication_data_no_defense = await test_initMAS_agent_ma(agnd, query, context, num_dialogue_turns)
                    elif args.dataset == "InjecAgent":
                        communication_data_no_defense = await test_initMAS_agent_ta(agnd, case,
                                                                                    num_dialogue_turns)

                for i in range(len(d_wd_system_prompts)):
                    system_prompts_embedding.append(embedding_model.encode(d_wd_system_prompts[i]))

                normal_system_prompts = d_wd_system_prompts[labels.tolist().index(0)].replace(str(labels.tolist().index(0)),
                                                                                              "{}", 1)
            elif args.training_phase == "agent":
                if args.test_again:
                    agnd = AgentGraph(adj_m, d_nd_system_prompts, attacker_idxes,
                                      model_type=args.model_type)  # agent graph no defense
                if args.dataset == "mmlu":
                    if args.test_again:
                        communication_data_no_defense = await test_initMAS_agent(agnd, qa_data, num_dialogue_turns)
                    normal = attack["PI"]["normal"]
                    attack_data = attack["PI"]["attack"]
                elif args.dataset == "msmarco":
                    if args.test_again:
                        communication_data_no_defense = await test_initMAS_agent_ma(agnd, query, context,
                                                                                num_dialogue_turns)
                    normal = attack["MA"]["normal"]
                    attack_data = attack["MA"]["attack"]
                elif args.dataset == "InjecAgent":
                    if args.test_again:
                        communication_data_no_defense = await test_initMAS_agent_ta(agnd, case,
                                                                                num_dialogue_turns)
                    normal = attack["TA"]["normal"]
                    attack_data = attack["TA"]["attack"]
                elif args.dataset == "gsm8k":
                    if args.test_again:
                        communication_data_no_defense = await test_initMAS_agent_gsm8k(agnd, qa_data,
                                                                                num_dialogue_turns)
                    normal = attack["gsm8k"]["normal"]
                    attack_data = attack["gsm8k"]["attack"]

                for i in range(len(d_wd_system_prompts)):

                    if i in attacker_idxes:
                        system_prompts_embedding.append(
                            embedding_model.encode(d_wd_system_prompts[i] + attack_data))
                    else:
                        system_prompts_embedding.append(
                            embedding_model.encode(d_wd_system_prompts[i] + normal))
            elif args.training_phase == "communication":
                if args.test_again:
                    agnd = AgentGraph(adj_m, d_nd_system_prompts, [],
                                      model_type=args.model_type)  # agent graph no defense
                    if args.dataset == "mmlu":
                        communication_data_no_defense = await test_communication_attack(agnd, qa_data, num_dialogue_turns, attacker_idxes, args.turn)
                    elif args.dataset == "msmarco":
                        communication_data_no_defense = await test_communication_attack_ma(agnd, query, context, num_dialogue_turns,
                                                                                        attacker_idxes, args.turn)
                    elif args.dataset == "InjecAgent":
                        communication_data_no_defense = await test_communication_attack_ta(agnd, case,
                                                                                           num_dialogue_turns,
                                                                                           attacker_idxes, args.turn)

                agwd = AgentGraph(adj_m, d_nd_system_prompts, [],
                                  model_type=args.model_type)  # agent graph no defense
                if args.dataset == "mmlu":
                    communication_data_defense = await test_communication_attack_defense(agwd, qa_data, num_dialogue_turns,
                                                                              attacker_idxes, args.turn, embedding_model, gnn, adj_m, threshold_graph, threshold)

                elif args.dataset == "msmarco":
                    communication_data_defense = await test_communication_attack_defense_ma(agwd, query, context,
                                                                             num_dialogue_turns,
                                                                             attacker_idxes, args.turn,
                                                                             embedding_model, gnn, adj_m,
                                                                             threshold_graph, threshold)
                elif args.dataset == "InjecAgent":
                    communication_data_defense = await test_communication_attack_defense_ta(agwd, case,
                                                                                            num_dialogue_turns,
                                                                                            attacker_idxes, args.turn,
                                                                                            embedding_model, gnn, adj_m,
                                                                                            threshold_graph, threshold)


            if args.training_phase == "initMAS" or args.training_phase == "agent":
                with torch.no_grad():
                    x_hat, a_hat, z, g_s, g_s_hat = gnn(torch.tensor(system_prompts_embedding).unsqueeze(0),
                                                        torch.tensor(adj_m).unsqueeze(0))

                loss, node_score, attr_err, struct_err, graph_score = reconstruction_loss(
                    torch.tensor(system_prompts_embedding).unsqueeze(0), torch.tensor(adj_m).unsqueeze(0), x_hat, a_hat,
                    g_s, g_s_hat)

                scores = node_score[0].numpy()  # [N]

                if graph_score > threshold_graph or np.max(scores) > threshold:
                    flags = scores > threshold  # [N]
                else:
                    flags = np.zeros(node_score.size(1), dtype=bool)
                detect_idx = np.where(flags)[0]

            if args.training_phase == "initMAS":
                for xid, i in enumerate(flags):
                    if i:
                        d_wd_system_prompts[xid] = normal_system_prompts.format(xid)

                agwd = AgentGraph(adj_m, d_wd_system_prompts, [],
                                  model_type=args.model_type)  # agent graph no defense

                if args.dataset == "mmlu":
                    communication_data_defense = await test_initMAS_agent(agwd, qa_data, num_dialogue_turns)
                elif args.dataset == "msmarco":
                    communication_data_defense = await test_initMAS_agent_ma(agwd, query, context, num_dialogue_turns)
                elif args.dataset == "InjecAgent":
                    communication_data_defense = await test_initMAS_agent_ta(agwd, case, num_dialogue_turns)

            elif args.training_phase == "agent":
                for xid, i in enumerate(flags):
                    if i:
                        d_wd_system_prompts[xid] = normal_system_prompts.format(xid)

                for idx in detect_idx:
                    if idx in attacker_idxes:
                        attacker_idxes.remove(idx)

                agwd = AgentGraph(adj_m, d_wd_system_prompts, attacker_idxes,
                                  model_type=args.model_type)  # agent graph no defense
                if args.dataset == "mmlu":
                    communication_data_defense = await test_initMAS_agent(agwd, qa_data, num_dialogue_turns)
                elif args.dataset == "msmarco":
                    communication_data_defense = await test_initMAS_agent_ma(agwd, query, context, num_dialogue_turns)
                elif args.dataset == "InjecAgent":
                    communication_data_defense = await test_initMAS_agent_ta(agwd, case, num_dialogue_turns)
                elif args.dataset == "gsm8k":
                    communication_data_defense = await test_initMAS_agent_gsm8k(agwd, qa_data, num_dialogue_turns)

        except Exception as e:
            print(e)
            continue

        if args.test_again:
            d_nd["communication_data"] = communication_data_no_defense
        d_wd["communication_data"] = communication_data_defense
        if args.test_again:
            final_dataset_nd.append(d_nd)
        final_dataset_wd.append(d_wd)
    if args.test_again:
        with open(args.save_path_no_defense, "w") as file:
            json.dump(final_dataset_nd, file, indent=None)
    with open(args.save_path_with_defense, "w") as file:
        json.dump(final_dataset_wd, file, indent=None)


if __name__ == "__main__":
    asyncio.run(main())


