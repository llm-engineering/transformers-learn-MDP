import torch
import torch.nn as nn
import itertools

from tqdm import tqdm
import torch.nn.functional as F
from enum import Enum


class Loss(Enum):
    CrossEntropy = 0
    KLDivergence = 1


class Mode(Enum):
    STATE = 0
    ACTION = 1
    STATE_ACTION = 2


# NOTE: as in whether to include just player 1 or the whole sequence
class SeqSubSet(Enum):
    WHOLE = 0
    PLAYER_1 = 1


def batch_one_hot(batch, number_of_classes):
    """
    One-hot encode a batch of sequences.
    """
    batch_size, seq_length = batch.size()
    one_hot = torch.zeros(batch_size, seq_length, number_of_classes).to(batch.device)
    one_hot.scatter_(2, batch.unsqueeze(-1), 1)
    return one_hot


def loss_calc(loss_type, loss_fn, logits, target, indices, padding_token=0):

    assert loss_fn.reduction == "none"

    vocab_length = logits.shape[-1]

    target = (
        target[:, indices].contiguous().view(-1)
    )  # expect this to be one dimensional
    logits = logits[:, indices, :].contiguous().view(-1, logits.shape[-1])

    if loss_type == Loss.KLDivergence:
        """
        If loss is KL Divergence, the target needs to be a probability distribution over the
        vocabulary
        """
        logits = F.log_softmax(logits, dim=-1)
        target = F.one_hot(
            target, num_classes=vocab_length
        )  # expect batch size is (batch_size x seq_length, vocab_length)

    loss = loss_fn(logits, target)

    if loss_type == Loss.KLDivergence:
        """
        KLDivergence without reduction shoots out a seq of dim (seq_length ,vocab)
        """
        loss = loss.mean(dim=-1)

    assert len(loss.shape) == 1

    # ----- mask making -----------

    mask = (target != padding_token).float()

    return (loss * mask, mask)


def logit_selection(length, mode, seq_type):
    """

    If mode is 0, select all the odd indices from 1 to length-1, because player 2 is randomly selecting the column.

    For mode 2 it's action state

    (a_0,s_0,a_1,s_1,a_0,s_1 ..... -> (a_0,s_0 ....), (s_0, a_1, ....)


    """

    whole_seq = range(length)

    if seq_type == SeqSubSet.WHOLE:
        return list(whole_seq), list(whole_seq)

    if mode == 0 or mode == 1:
        player_1 = list(range(0, length, 2))
        player_2 = [x for x in whole_seq if x not in player_1]
        return (player_1, player_2)

    player_1 = [0, 1]  # Why is 0 here? because action can be used to predict state
    for i in range(3, length, 4):
        player_1.extend(list(range(i, min(i + 3, length))))
    
    if player_1[-1] != length -1:
        player_1.append(length-1)

    return (player_1, [x for x in whole_seq if x not in player_1])


def criterion_f(loss_type):
    if loss_type == Loss.CrossEntropy:
        return nn.CrossEntropyLoss(
            ignore_index=0, reduction="none", label_smoothing=0.0
        )
    else:
        return nn.KLDivLoss(reduction="none")


# TODO: mode
def validate_model(model, valid_loader, accelerator, mode, loss_type, seq_type):

    model.eval()
    criterion = criterion_f(loss_type)

    valid_loss = torch.tensor(0.0).to(accelerator.device)
    valid_data = torch.tensor(0.0).to(accelerator.device)
    player_1_accuracy = torch.tensor(0.0).to(accelerator.device)
    player_2_accuracy = torch.tensor(0.0).to(accelerator.device)
    total_accuracy = torch.tensor(0.0).to(accelerator.device)
    player_1_total = torch.tensor(0.0).to(accelerator.device)
    player_2_total = torch.tensor(0.0).to(accelerator.device)

    with torch.no_grad():

        for X_batch, Y_batch in valid_loader:

            p1_indices, p2_indices = logit_selection(X_batch.size(1), mode, seq_type)

            logits = model(X_batch)  # Shape: [batch_size, seq_length, vocab_size]

            logits = F.log_softmax(logits, dim=-1)

            p1_indices_, p2_indices_ = logit_selection(X_batch.size(1), mode, SeqSubSet.PLAYER_1)

            player_1_accuracy_ = (
                logits[:, p1_indices_].argmax(dim=-1) == Y_batch[:, p1_indices_]
            ).float()
            player_2_accuracy_ = (
                logits[:, p2_indices_].argmax(dim=-1) == Y_batch[:, p2_indices_]
            ).float()
            total_accuracy_ = (logits.argmax(dim=-1) == Y_batch).float()

            masked_loss, mask = loss_calc(
                loss_type, criterion, logits, Y_batch, p1_indices
            )

            valid_loss += masked_loss.sum()  # Sum the losses at valid positions
            valid_data += mask.sum()  # Count valid positions
            player_1_accuracy += player_1_accuracy_.sum()
            player_2_accuracy += player_2_accuracy_.sum()
            total_accuracy += total_accuracy_.sum()
            player_1_total += player_1_accuracy_.numel()
            player_2_total += player_2_accuracy_.numel()

    accelerator.wait_for_everyone()

    valid_loss = accelerator.gather(valid_loss).sum()
    valid_data = accelerator.gather(valid_data).sum()
    player_1_accuracy = accelerator.gather(player_1_accuracy).sum()
    player_2_accuracy = accelerator.gather(player_2_accuracy).sum()
    total_accuracy = accelerator.gather(total_accuracy).sum()
    player_1_total = accelerator.gather(player_1_total).sum()
    player_2_total = accelerator.gather(player_2_total).sum()

    if accelerator.is_main_process:
        return (
            (valid_loss / valid_data).item(),
            (player_1_accuracy / player_1_total).item(),
            (player_2_accuracy / player_2_total).item(),
            (total_accuracy / (player_1_total + player_2_total)).item(),
        )
    else:
        return None


def train_model(
    model,
    train_loader,
    optimizer,
    accelerator,
    scheduler,
    wandb,
    mode,
    loss_type,
    seq_type,
):

    model.train()

    criterion = criterion_f(loss_type)

    train_loss = torch.tensor(0.0).to(accelerator.device)
    train_data = torch.tensor(0.0).to(accelerator.device)

    for X_batch, Y_batch in tqdm(train_loader, desc="Training"):

        optimizer.zero_grad()

        p1_indices, p2_indices = logit_selection(X_batch.size(1), mode, seq_type)

        logits = model(X_batch)  # Shape: [batch_size, seq_length, vocab_size]

        masked_loss, mask = loss_calc(None, criterion, logits, Y_batch, p1_indices)

        loss_sum = masked_loss.sum()
        data_sum = mask.sum()

        loss = loss_sum / data_sum
        accelerator.backward(loss)
        # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            wandb.log({"Learning Rate": scheduler.get_last_lr()[0]})

        train_loss += loss_sum.item()
        train_data += mask.sum().item()

        grad_norms = []

        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        if len(grad_norms) == 0:
            return None  # No gradients yet (e.g., before first backward pass)

        grad_tensor = torch.tensor(grad_norms)

        stats = {
            "grad_mean": grad_tensor.mean().item(),
            "grad_max": grad_tensor.max().item(),
            "grad_std": grad_tensor.std().item(),
            "grad_p95": grad_tensor.quantile(0.95).item(),
        }

        # accelerator.print('Gradient norm:', stats)
        wandb.log(stats)

    accelerator.wait_for_everyone()

    train_loss = accelerator.gather(train_loss).sum()
    train_data = accelerator.gather(train_data).sum()

    accelerator.print("Training Loss:", (train_loss / train_data).item())

    if accelerator.is_main_process:
        return (train_loss / train_data).item()
    else:
        return None
