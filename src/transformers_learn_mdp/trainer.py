import torch
import torch.nn as nn

from tqdm import tqdm

def validate_model(model, valid_loader, accelerator):
    
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction = 'none')

    valid_loss = torch.tensor(0.0).to(accelerator.device)
    valid_data = torch.tensor(0.0).to(accelerator.device)

    with torch.no_grad():

        for X_batch, Y_batch in valid_loader:

            logits = model(X_batch)
            logits = logits.view(-1, logits.size(-1))  # Shape: [batch_size * seq_length, vocab_size]
            Y_batch = Y_batch.view(-1)  # Shape: [batch_size * seq_length]

            # Assuming the padding token index is 0
            padding_token_index = 0
            mask = (Y_batch != padding_token_index).float()  # Create a mask for valid positions

            loss = criterion(logits, Y_batch)  # Calculate loss without reduction
            masked_loss = loss * mask  # Apply mask

            valid_loss += masked_loss.sum().item() # Sum the losses at valid positions
            valid_data += mask.sum().item() # Count valid positions

    accelerator.wait_for_everyone()
    
    valid_loss = accelerator.gather(valid_loss).sum()
    valid_data = accelerator.gather(valid_data).sum()

    if accelerator.is_main_process:
        return (valid_loss / valid_data).item()
    else:
        return None

def train_model(model, train_loader, optimizer, accelerator, scheduler=None):

    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction = 'none', label_smoothing=0.0)

    train_loss = torch.tensor(0.0).to(accelerator.device)
    train_data = torch.tensor(0.0).to(accelerator.device)

    for X_batch, Y_batch in tqdm(train_loader, desc="Training"):
        
        optimizer.zero_grad()
        logits = model(X_batch)

        logits = logits.view(-1, logits.size(-1))  # Shape: [batch_size * seq_length, vocab_size]
        Y_batch = Y_batch.view(-1)  # Shape: [batch_size * seq_length]

        padding_token_index = 0  # Assuming the padding token index is 0
        mask = (Y_batch != padding_token_index).float()
        
        loss = criterion(logits, Y_batch)
        masked_loss = loss * mask
        
        loss_sum = masked_loss.sum()
        data_sum = mask.sum()

        loss = loss_sum / data_sum
        accelerator.backward(loss)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

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
            "mean": grad_tensor.mean().item(),
            "max": grad_tensor.max().item(),
            "std": grad_tensor.std().item(),
            "p95": grad_tensor.quantile(0.95).item()
        }

        #accelerator.print('Gradient norm:', stats)

    accelerator.wait_for_everyone()
    
    train_loss = accelerator.gather(train_loss).sum()
    train_data = accelerator.gather(train_data).sum()
    
    accelerator.print('Training Loss:', (train_loss / train_data).item())

    if accelerator.is_main_process:
        return (train_loss / train_data).item()
    else:
        return None
    

