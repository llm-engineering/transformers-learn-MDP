import os
import sys
import pickle
import shutil
import torch
import wandb
from tqdm import tqdm
import hydra
import itertools
from omegaconf import DictConfig, OmegaConf, open_dict

from accelerate import Accelerator
from .dataset import EpisodeDataset, collate_fn
from .model import Config, GPTModel
from .trainer import train_model, validate_model, Loss, Mode, SeqSubSet
from torch.utils.data import DataLoader

from .data_utils import information_parser, actions_to_col_row
from enum import Enum


def train(training_config, training_dataset, validation_dataset, token_to_idx, wandb):

    train_dataset = EpisodeDataset(training_dataset, token_to_idx)
    valid_dataset = EpisodeDataset(validation_dataset, token_to_idx)

    accelerator = Accelerator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    config = Config(
        training_config.vocab_size,
        training_config.seq_len,
        n_layer=training_config.num_layers,
        n_head=training_config.num_heads,
        n_embd=training_config.embedding_size,
    )
    model = GPTModel(config)

    #optimizer = torch.optim.AdamW(
    #    model.parameters(),
    #    lr=training_config.lr,
    #    weight_decay=training_config.weight_decay,
    #)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=training_config.lr,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.0005,
        steps_per_epoch=len(train_loader),
        epochs=training_config.epochs,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)

    train_loader, valid_loader, model, scheduler, optimizer = accelerator.prepare(
        train_loader, valid_loader, model, scheduler, optimizer
    )

    epoch = 0

    model_path = None
    min_loss = 1e10

    train_losses = []
    valid_losses = []



    # TODO: this is just pulling things out from a config
    mode = Mode(training_config.mode)
    loss_type = Loss(training_config.loss_type)
    seq_type = SeqSubSet(training_config.seq_type)

    for epoch in tqdm(range(training_config.epochs), desc="Epoch"):
        accelerator.print(f"Epoch {epoch}")
        wandb.log({"Epoch": epoch})

        train_loss = train_model(
            model, train_loader, optimizer, accelerator, scheduler, wandb, mode, loss_type, seq_type
        )
        valid_loss, p1_acc, p2_acc, total_acc = validate_model(model, valid_loader, accelerator, mode, loss_type, seq_type)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        #scheduler.step()

        # print("Learning Rate: ", scheduler.get_last_lr())

        mode = training_config.mode
        seed = training_config.seed

        if accelerator.is_main_process:
            val_loss_str = f"Validation loss {valid_loss:.8f}"
            wandb.log({"Validation Loss": valid_loss, "Training Loss": train_loss, "P1 Acc": p1_acc, "P2 Acc": p2_acc, "Total accuracy": total_acc})
            accelerator.print(val_loss_str)

            model_save_path = f"model_{epoch+1}_mode_{mode}_seed_{seed}.pth"
            accelerator.save(
                accelerator.unwrap_model(model).state_dict(), model_save_path
            )

            if valid_loss < min_loss:
                min_loss = valid_loss
                model_path = model_save_path

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        shutil.copy(model_path, training_config.save_directory)

    with open(f"train_losses_mode_{mode}_seed_{seed}.pkl", "wb") as f:
        pickle.dump(train_losses, f)
    with open(f"valid_losses_mode_{mode}_seed_{seed}.pkl", "wb") as f:
        pickle.dump(valid_losses, f)

    wandb.finish()


def split_dataset(data, train_ratio, valid_ratio):
    train = data[: int(train_ratio * len(data))]
    valid = data[
        int(train_ratio * len(data)) : int((train_ratio + valid_ratio) * len(data))
    ]
    test = data[int((train_ratio + valid_ratio) * len(data)) :]
    return train, valid, test


def mode_to_token_to_idx(mode):
    if mode == 0:
        token_to_idx = {(i, j): i * 7 + j + 1 for i in range(6) for j in range(7)}
        vocab_size = 43
        transformation = actions_to_col_row
    elif mode == 1:
        token_to_idx = {i: i + 1 for i in range(7)}
        vocab_size = 8
        transformation = lambda x: x
    elif mode == 2:
        token_to_idx = {(i, j): i * 7 + j + 1 for i in range(6) for j in range(7)} | {
            i: i + 44 for i in range(7)
        }
        vocab_size = 51
        transformation = lambda x: list(itertools.chain(*zip(x,actions_to_col_row(x))))
    token_to_idx["<pad>"] = 0  # Padding token

    token_to_idx[51] = 51
    vocab_size += 1

    return token_to_idx, vocab_size, transformation


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    training_config = cfg.training

    mode = training_config.mode
    token_to_idx, vocab_size, transformation = mode_to_token_to_idx(mode)

    # Make this a function
    with open(training_config.data_path, "r") as f:
        data = f.readlines()
        data = information_parser(data)
        raw_dataset = [transformation([action for (_, action) in x]) for x in data]


    training_dataset, validation_dataset, test_dataset = split_dataset(
        raw_dataset, training_config.train_ratio, training_config.val_ratio
    )

    with open_dict(training_config):
        training_config["vocab_size"] = vocab_size
        training_config["dataset_length"] = len(raw_dataset)

    wandb.init(project=cfg.wandb.project_name, config=dict(training_config), id=cfg.wandb.id)

    train(training_config, training_dataset, validation_dataset, token_to_idx, wandb)


if __name__ == "__main__":
    main()
