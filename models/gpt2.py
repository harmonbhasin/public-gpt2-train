from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    # key, query, value projection for all heads, but in a batch
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    # output projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1
    # regularization
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.dropout = nn.Dropout(config.dropout)
    # not really a 'bias' more of a mask, but following the OpenAI/HF naming though
    self.register_buffer(
      'bias',
      torch.tril(torch.ones(config.block_size, config.block_size)).view(
        1, 1, config.block_size, config.block_size
      ),
    )

  def forward(self, x):
    B, T, C = x.size()  # batch size, seq length, embed dim
    # nh is "number of heads", hs is "head size", and C (number of channels) = nh*hs
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
    # attention; written by hand to be replaced with option
    # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    # att = F.softmax(att, dim=-1)
    # att = self.dropout(att)  # Apply dropout to attention weights
    # y = att @ v

    # This is faster
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout.p)
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.c_proj(y)
    return y


class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
    self.gelu = nn.GELU(approximate='tanh')
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    x = self.dropout(x)  # Apply dropout to MLP output
    return x


# Attention is reduce, and MLP is map
class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x


@dataclass
class GPTConfig:
  block_size: int = 1024  # max seq length
  vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 byes tokens +
  # 1 <|endoftoken|> token (delimits and starts generation of documents)
  n_layer: int = 12  # num layers
  n_head: int = 12  # num of heads
  n_embd: int = 768  # embedding
  dropout: float = 0.1  # dropout rate


class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.transformer = nn.ModuleDict(
      dict(
        wte=nn.Embedding(config.vocab_size, config.n_embd),
        wpe=nn.Embedding(config.block_size, config.n_embd),
        h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ln_f=nn.LayerNorm(config.n_embd),
      )
    )
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
    self.dropout = nn.Dropout(config.dropout)

    # weight sharing scheme; this makes sense now; improves results
    self.transformer.wte.weight = self.lm_head.weight

    # init params
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      std = 0.02
      if hasattr(module, 'NANOGPT_SCALE_INIT'):
        std *= (2 * self.config.n_layer) ** -0.5
      torch.nn.init.normal_(module.weight, mean=0.0, std=std)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    # idx is of shape (B,T)
    B, T = idx.size()
    assert self.config.block_size >= T

    pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
    pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
    tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
    x = tok_emb + pos_emb
    x = self.dropout(x)  # Apply dropout to embeddings

    for block in self.transformer.h:
      x = block(x)
    x = self.transformer.ln_f(x)
    loss = None
    logits = self.lm_head(x)  # (B,T, vocab_size)
    if targets is not None:
      # 3 to two dimensional tensor: BxT, vocab_size; BxT
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
