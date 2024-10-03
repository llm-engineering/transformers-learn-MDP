import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer):
        self.vocab_size = vocab_size  # Vocabulary size
        self.block_size = block_size  # Input sequence length
        self.n_embd = n_embd          # Embedding dimension
        self.n_head = n_head          # Number of attention heads
        self.n_layer = n_layer        # Number of transformer layers
        # Dropout rates
        self.embd_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.attn_pdrop = 0.1

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert self.config.n_embd % self.config.n_head == 0, "embedding dimension must be divisible by the number of heads."

        # Key, query, and value linear transformations
        self.key = nn.Linear(self.config.n_embd, self.config.n_embd)
        self.query = nn.Linear(self.config.n_embd, self.config.n_embd)
        self.value = nn.Linear(self.config.n_embd, self.config.n_embd)

        # Dropout layers
        self.attn_drop = nn.Dropout(self.config.attn_pdrop)
        self.resid_drop = nn.Dropout(self.config.resid_pdrop)

        # Output projection
        self.proj = nn.Linear(self.config.n_embd, self.config.n_embd)

        # Causal mask to prevent attention to future tokens
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).unsqueeze(0).unsqueeze(1))

    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension

        # Calculate projections and reshape for multi-headed attention
        k = self.key(x).view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)

        # Compute the attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att) @ v

        # Re-assemble all head outputs side by side
        y = att.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_drop(self.proj(y))
        return y

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.config.block_size, self.config.n_embd))
        self.drop = nn.Dropout(self.config.embd_pdrop)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(self.config.n_layer)])
        
        # Final layer normalization and linear output layer
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):

        # Initialize weights and biases
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, num_probe=None):
        
        # Processing input
        _, T = idx.size()
        assert T <= self.config.block_size, "Input sequence too long."
        
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :T, :]
        x = self.drop(token_embeddings + position_embeddings)

        # Retrieve Embedding if num_probe is specified
        if num_probe is not None:
            for block in self.blocks[:num_probe]:
                x = block(x)
            return x
                
        # Process through all blocks, then project to vocabulary size
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
