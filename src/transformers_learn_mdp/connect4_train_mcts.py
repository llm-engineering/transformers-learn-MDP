import os
import sys
import pickle
import shutil
import torch
import argparse
import wandb
from tqdm import tqdm

from accelerate import Accelerator
from .dataset import EpisodeDataset, collate_fn
from .model import Config, GPTModel
from .trainer import train_model, validate_model
from torch.utils.data import DataLoader

from .data_utils import information_parser


wandb.init(project="mdp-learning")


"""
Training pipeline for transformer on Connect-4 data generated through MCTS.
"""
def train_main(train_dataset, valid_dataset, vocab_size, block_size, num_layers, embed_size, mode, seed, save_directory = None, epochs = 50):
    
    accelerator = Accelerator()

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

    config = Config(vocab_size, block_size, n_layer=num_layers, n_head=num_layers // 2, n_embd=embed_size)
    model = GPTModel(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=len(train_loader), epochs=epochs)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
    
    train_loader, valid_loader, model, scheduler, optimizer = accelerator.prepare(train_loader, valid_loader, model, scheduler, optimizer)

    epoch = 0

    model_path = None
    min_loss = 1e10
    
    train_losses = []
    valid_losses = []

    for epoch in tqdm(range(epochs)):
        accelerator.print(f'Epoch {epoch}')
        wandb.log({"Epoch": epoch})

        train_loss = train_model(model, train_loader, optimizer, accelerator, scheduler)
        valid_loss = validate_model(model, valid_loader, accelerator)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        scheduler.step()

        #print("Learning Rate: ", scheduler.get_last_lr())

        if accelerator.is_main_process:
            val_loss_str = f'Validation loss {valid_loss:.8f}'
            wandb.log({"Validation Loss": valid_loss, "Training Loss": train_loss})
            accelerator.print(val_loss_str)

            model_save_path = f"model_{epoch+1}_mode_{mode}_seed_{seed}.pth"
            accelerator.save(accelerator.unwrap_model(model).state_dict(), model_save_path)

            if valid_loss < min_loss:
                min_loss = valid_loss
                model_path = model_save_path

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        shutil.copy(model_path, save_directory)

    with open(f'train_losses_mode_{mode}_seed_{seed}.pkl', 'wb') as f:
        pickle.dump(train_losses, f)
    with open(f'valid_losses_mode_{mode}_seed_{seed}.pkl', 'wb') as f:
        pickle.dump(valid_losses, f)
    
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=0, choices=[0, 1, 2], help='Data Mode (state, action, state-action)')
    parser.add_argument('-s', type=int, default=23456, choices=[0, 1, 2], help='Seed')
    parser.add_argument('-i', type=str, help='Input Path')
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
    block_size = 52
    embed_size = 64
    num_layers = 8
    
    path = ''

    with open(args.i, 'r') as f:
        data = f.readlines()
        data = information_parser(data)
        agent1 = [[action for (_,action) in x] for x in data]


    train_ratio = 0.8
    valid_ratio = 0.1

    d1 = len(agent1)

    train = agent1[:int(train_ratio * d1)]
    valid = agent1[int(train_ratio * d1):int((train_ratio + valid_ratio) * d1) ]
    test = agent1[int((train_ratio + valid_ratio) * d1): ]

    print(len(train))
    print(len(valid))
    print(len(test))

    train_dataset = EpisodeDataset(train, token_to_idx)
    valid_dataset = EpisodeDataset(valid, token_to_idx)

    train_main(train_dataset, valid_dataset, vocab_size, block_size, num_layers, embed_size, args.m, args.s, "best_model")

if __name__ == "__main__":
    main()

