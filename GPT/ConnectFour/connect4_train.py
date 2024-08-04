import os
import sys
import pickle
import shutil
import torch
import argparse

sys.path.append('../')

from accelerate import Accelerator, notebook_launcher
from dataset import EpisodeDataset, collate_fn
from model import Config, GPTModel
from trainer import train_model, validate_model
from torch.utils.data import DataLoader

def train_main(train_dataset, valid_dataset, vocab_size, block_size, num_layers, embed_size, mode, save_directory = None, epochs = 15):
    
    accelerator = Accelerator()

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    config = Config(vocab_size, block_size, n_layer=num_layers, n_head=num_layers, n_embd=embed_size)
    model = GPTModel(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
    
    train_loader, valid_loader, model, scheduler, optimizer = accelerator.prepare(train_loader, valid_loader, model, scheduler, optimizer)

    epoch = 0

    model_path = None
    min_loss = 1e10
    
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        accelerator.print(f'Epoch {epoch}')

        train_loss = train_model(model, train_loader, optimizer, accelerator)
        valid_loss = validate_model(model, valid_loader, accelerator)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        scheduler.step()

        if accelerator.is_main_process:
            print(f'Validation Loss: {valid_loss:.8f}')

            model_save_path = f"Model_{epoch+1}_mode_{mode}.pth"
            accelerator.save(accelerator.unwrap_model(model).state_dict(), model_save_path)

            if valid_loss < min_loss:
                min_loss = valid_loss
                model_path = model_save_path

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        shutil.copy(model_path, save_directory)

    with open(f'train_losses_mode_{mode}.pkl', 'wb') as f:
        pickle.dump(train_losses, f)
    with open(f'valid_losses_mode_{mode}.pkl', 'wb') as f:
        pickle.dump(valid_losses, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=0, choices=[0, 1, 2], help='Data Mode (state, action, state-action)')
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

    path = r'C:\Users\wmasi\Documents\TF-Agent-States\RL_Training_ConnectFour'

    with open(os.path.join(path, rf'\training_data\1\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent1 = pickle.load(f)
    with open(os.path.join(path, rf'\training_data\2\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent2 = pickle.load(f)
    with open(os.path.join(path, rf'\training_data\3\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent3 = pickle.load(f)
    with open(os.path.join(path, rf'\training_data\4\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent4 = pickle.load(f)
    with open(os.path.join(path, rf'\training_data\5\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent5 = pickle.load(f)
    with open(os.path.join(path, rf'\training_data\6\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent6 = pickle.load(f)
    with open(os.path.join(path, rf'\training_data\7\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent7 = pickle.load(f)
    with open(os.path.join(path, rf'\training_data\8\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent8 = pickle.load(f)
    with open(os.path.join(path, rf'\training_data\9\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent9 = pickle.load(f)
    with open(os.path.join(path, rf'\training_data\10\training_games_1000000_mode_{args.m}.pkl'), 'rb') as f:
        agent10 = pickle.load(f)

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

    train_dataset = EpisodeDataset(train, token_to_idx)
    valid_dataset = EpisodeDataset(valid, token_to_idx)
    test_dataset = EpisodeDataset(test, token_to_idx)

    train_main(train_dataset, valid_dataset, vocab_size, block_size, num_layers, embed_size, args.m)

