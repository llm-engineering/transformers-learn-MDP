import pickle
import torch
import sys
import os
import random
import argparse

sys.path.append('../')

from generate_embeddings_connect4 import get_embeddings_qvalues, min_max_normalization
from probe import ProbeDataset, train_probe, test_probe
from GPT.dataset import EpisodeDataset
from GPT.model import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=0, choices=[0, 1, 2], help='Data Mode (state, action, state-action)')
    parser.add_argument('-s', type=int, default=0, choices=[0, 1, 2], help='Seed')
    args = parser.parse_args()
    if args.m == 0:
        token_to_idx = {(i, j): i * 7 + j + 1 for i in range(6) for j in range(7)}
        vocab_size = 43
    elif args.m == 1:
        token_to_idx = {i: i + 1 for i in range(7)}
        vocab_size = 8
    elif args.m == 2:
        token_to_idx = {(i, j): i * 7 + j + 1 for i in range(6) for j in range(7)} | {i: i + 44 for i in range(7)}
        vocab_size = 51
    token_to_idx['<pad>'] = 0  # Padding token
    block_size = 42 # Honestly this could probably be whatever
    embed_size = 512
    num_heads = 8
    num_layers = 8
    dropout = 0.1
    torch.manual_seed(args.s)
    random.seed(args.s)
    torch.cuda.manual_seed(args.s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    path = ''

    with open(os.path.join(path, rf'training_data\1\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent1 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\2\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent2 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\3\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent3 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\4\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent4 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\5\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent5 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\6\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent6 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\7\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent7 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\8\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent8 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\9\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent9 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\10\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent10 = pickle.load(f)

    with open(os.path.join(path, rf'training_data\1\q_vals_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        qagent1 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\2\q_vals_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        qagent2 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\3\q_vals_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        qagent3 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\4\q_vals_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        qagent4 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\5\q_vals_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        qagent5 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\6\q_vals_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        qagent6 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\7\q_vals_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        qagent7 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\8\q_vals_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        qagent8 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\9\q_vals_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        qagent9 = pickle.load(f)
    with open(os.path.join(path, rf'training_data\10\q_vals_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        qagent10 = pickle.load(f)

    train_ratio = 0.8
    valid_ratio = 0.1

    d1 = len(agent1)
    d2 = len(agent2)
    d3 = len(agent3)
    d4 = len(agent4)
    d5 = len(agent5)
    d6 = len(agent6)
    d7 = len(agent7)
    d8 = len(agent8)
    d9 = len(agent9)
    d10 = len(agent10)

    train1 = agent1[:int(train_ratio * d1)]
    valid1 = agent1[int(train_ratio * d1):int((train_ratio + valid_ratio) * d1) ]
    test1 = agent1[int((train_ratio + valid_ratio) * d1): ]

    train2 = agent2[:int(train_ratio * d2)]
    valid2 = agent2[int(train_ratio * d2):int((train_ratio + valid_ratio) * d2) ]
    test2 = agent2[int((train_ratio + valid_ratio) * d2): ]

    train3 = agent3[:int(train_ratio * d3)]
    valid3 = agent3[int(train_ratio * d3):int((train_ratio + valid_ratio) * d3)]
    test3 = agent3[int((train_ratio + valid_ratio) * d3):]

    train4 = agent4[:int(train_ratio * d4)]
    valid4 = agent4[int(train_ratio * d4):int((train_ratio + valid_ratio) * d4)]
    test4 = agent4[int((train_ratio + valid_ratio) * d4):]

    train5 = agent5[:int(train_ratio * d5)]
    valid5 = agent5[int(train_ratio * d5):int((train_ratio + valid_ratio) * d5)]
    test5 = agent5[int((train_ratio + valid_ratio) * d5):]

    train6 = agent6[:int(train_ratio * d6)]
    valid6 = agent6[int(train_ratio * d6):int((train_ratio + valid_ratio) * d6)]
    test6 = agent6[int((train_ratio + valid_ratio) * d6):]

    train7 = agent7[:int(train_ratio * d7)]
    valid7 = agent7[int(train_ratio * d7):int((train_ratio + valid_ratio) * d7)]
    test7 = agent7[int((train_ratio + valid_ratio) * d7):]

    train8 = agent8[:int(train_ratio * d8)]
    valid8 = agent8[int(train_ratio * d8):int((train_ratio + valid_ratio) * d8)]
    test8 = agent8[int((train_ratio + valid_ratio) * d8):]

    train9 = agent9[:int(train_ratio * d9)]
    valid9 = agent9[int(train_ratio * d9):int((train_ratio + valid_ratio) * d9)]
    test9 = agent9[int((train_ratio + valid_ratio) * d9):]

    train10 = agent10[:int(train_ratio * d10)]
    valid10 = agent10[int(train_ratio * d10):int((train_ratio + valid_ratio) * d10)]
    test10 = agent10[int((train_ratio + valid_ratio) * d10):]

    train = train1 + train2 + train3 + train4 + train5 + train6 + train7 + train8 + train9 + train10
    valid = valid1 + valid2 + valid3 + valid4 + valid5 + valid6 + valid7 + valid8 + valid9 + valid10
    test = test1 + test2 + test3 + test4 + test5 + test6 + test7 + test8 + test9 + test10
    
    print(len(train))
    print(len(valid))
    print(len(test))

    q1 = len(qagent1)
    q2 = len(qagent2)
    q3 = len(qagent3)
    q4 = len(qagent4)
    q5 = len(qagent5)
    q6 = len(qagent6)
    q7 = len(qagent7)
    q8 = len(qagent8)
    q9 = len(qagent9)
    q10 = len(qagent10)

    qtrain1 = qagent1[:int(train_ratio * q1)]
    qvalid1 = qagent1[int(train_ratio * q1):int((train_ratio + valid_ratio) * q1)]
    qtest1 = qagent1[int((train_ratio + valid_ratio) * q1):]

    qtrain2 = qagent2[:int(train_ratio * q2)]
    qvalid2 = qagent2[int(train_ratio * q2):int((train_ratio + valid_ratio) * q2)]
    qtest2 = qagent2[int((train_ratio + valid_ratio) * q2):]

    qtrain3 = qagent3[:int(train_ratio * q3)]
    qvalid3 = qagent3[int(train_ratio * q3):int((train_ratio + valid_ratio) * q3)]
    qtest3 = qagent3[int((train_ratio + valid_ratio) * q3):]

    qtrain4 = qagent4[:int(train_ratio * q4)]
    qvalid4 = qagent4[int(train_ratio * q4):int((train_ratio + valid_ratio) * q4)]
    qtest4 = qagent4[int((train_ratio + valid_ratio) * q4):]

    qtrain5 = qagent5[:int(train_ratio * q5)]
    qvalid5 = qagent5[int(train_ratio * q5):int((train_ratio + valid_ratio) * q5)]
    qtest5 = qagent5[int((train_ratio + valid_ratio) * q5):]

    qtrain6 = qagent6[:int(train_ratio * q6)]
    qvalid6 = qagent6[int(train_ratio * q6):int((train_ratio + valid_ratio) * q6)]
    qtest6 = qagent6[int((train_ratio + valid_ratio) * q6):]

    qtrain7 = qagent7[:int(train_ratio * q7)]
    qvalid7 = qagent7[int(train_ratio * q7):int((train_ratio + valid_ratio) * q7)]
    qtest7 = qagent7[int((train_ratio + valid_ratio) * q7):]

    qtrain8 = qagent8[:int(train_ratio * q8)]
    qvalid8 = qagent8[int(train_ratio * q8):int((train_ratio + valid_ratio) * q8)]
    qtest8 = qagent8[int((train_ratio + valid_ratio) * q8):]

    qtrain9 = qagent9[:int(train_ratio * q9)]
    qvalid9 = qagent9[int(train_ratio * q9):int((train_ratio + valid_ratio) * q9)]
    qtest9 = qagent9[int((train_ratio + valid_ratio) * q9):]

    qtrain10 = qagent10[:int(train_ratio * q10)]
    qvalid10 = qagent10[int(train_ratio * q10):int((train_ratio + valid_ratio) * q10)]
    qtest10 = qagent10[int((train_ratio + valid_ratio) * q10):]

    qtrain = qtrain1 + qtrain2 + qtrain3 + qtrain4 + qtrain5 + qtrain6 + qtrain7 + qtrain8 + qtrain9 + qtrain10
    qvalid = qvalid1 + qvalid2 + qvalid3 + qvalid4 + qvalid5 + qvalid6 + qvalid7 + qvalid8 + qvalid9 + qvalid10
    qtest = qtest1 + qtest2 + qtest3 + qtest4 + qtest5 + qtest6 + qtest7 + qtest8 + qtest9 + qtest10

    # train_dataset = EpisodeDataset(train, token_to_idx)
    # valid_dataset = EpisodeDataset(valid, token_to_idx)
    # test_dataset = EpisodeDataset(test, token_to_idx)

    config = Config(vocab_size, block_size, n_layer=num_layers, n_head=num_layers, n_embd=embed_size)

    training_pipeline(folder_path = 'Linear_Probe', num_samples = 8, layers = list(range(1, 9)), linear = True, model_load_path = 'Model_12.pth', train_random = True, \
                      train = train, qtrain = qtrain, valid = valid, qvalid = qvalid, test = test, qtest = qtest, config = config, token_to_idx = token_to_idx)

def training_pipeline(folder_path: str, num_samples: int, layers: list, linear, model_load_path, train_random, train, qtrain, valid, qvalid, test, qtest, config, token_to_idx):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in layers:

        curr_path = os.path.join(folder_path, f"Layer_{i}")

        if not os.path.exists(curr_path):
            os.makedirs(curr_path)
        
        print(f"Layer {i}")
        
        # Retreive Embeddings and Normalize Q-Values

        embed_train, qval_train = get_embeddings_qvalues(num_samples, train, qtrain, i, config, token_to_idx, model_load_path = model_load_path, data_path = 'sampled_q_vals_train.pkl')
        embed_valid, qval_valid = get_embeddings_qvalues(num_samples, valid, qvalid, i, config, token_to_idx, model_load_path = model_load_path, data_path = 'sampled_q_vals_valid.pkl')
        embed_test, qval_test = get_embeddings_qvalues(num_samples, test, qtest, i, config, token_to_idx, model_load_path = model_load_path, data_path = 'sampled_q_vals_test.pkl')

        qval_train_norm, min, max = min_max_normalization(qval_train)
        qval_valid_norm = min_max_normalization(qval_valid, min, max)
        qval_test_norm = min_max_normalization(qval_test, min, max)

        d = len(qval_train_norm[0])
        n = embed_train[0].shape[0]

        # Non-Random

        probe_dataset_train = ProbeDataset(embed_train, qval_train_norm)
        probe_dataset_valid = ProbeDataset(embed_valid, qval_valid_norm)
        probe_dataset_test = ProbeDataset(embed_test, qval_test_norm)

        print("\nTraining Normal Probe\n")
        model_path, train_loss, valid_loss = train_probe(probe_dataset_train, probe_dataset_valid, device = device, epochs = 50, params = (d, n), model_dir = os.path.join(curr_path, f"Non_Random_Layer_{i}"), linear=linear)

        with open(os.path.join(curr_path, "normal_train_loss"), 'wb') as f:
            pickle.dump(train_loss, f)
        with open(os.path.join(curr_path, "normal_valid_loss"), 'wb') as f:
            pickle.dump(valid_loss, f)

        test_loss = test_probe(probe_dataset_test, model_path, (d, n), device, linear)
        print(f"MSE Loss: {test_loss:.4f}")

        if train_random:

            # Random
        
            random_embeddings_train = [torch.randn(512, 1) for _ in range(len(embed_train))]
            random_embeddings_valid = [torch.randn(512, 1) for _ in range(len(embed_valid))]
            random_embeddings_test = [torch.randn(512, 1) for _ in range(len(embed_test))]
            
            random_dataset_train = ProbeDataset(random_embeddings_train, qval_train_norm)
            random_dataset_valid = ProbeDataset(random_embeddings_valid, qval_valid_norm)
            random_dataset_test = ProbeDataset(random_embeddings_test, qval_test_norm)
    
            print("\nTraining Random Probe\n")
            random_path, random_train_loss, random_valid_loss = train_probe(random_dataset_train, random_dataset_valid, device = device, epochs = 50, params = (d, n), model_dir = os.path.join(curr_path, f"Random_Layer_{i}"), linear=linear)
    
            with open(os.path.join(curr_path, "random_train_loss"), 'wb') as f:
                pickle.dump(random_train_loss, f)
            with open(os.path.join(curr_path, "random_valid_loss"), 'wb') as f:
                pickle.dump(random_valid_loss, f)
    
            rand_loss = test_probe(random_dataset_test, random_path, (d, n), device, linear)
            print(f"Random MSE Loss: {rand_loss:.4f}\n")  


