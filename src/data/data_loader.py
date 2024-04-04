from typing import Literal

import tiktoken
import torch
from datasets import load_dataset


class DataLoaderLite:
    def __init__(self, B: int, T: int, split: Literal['train', 'validation', 'test'] = 'train'):
        self.B = B
        self.T = T
        self.split = split

        # Load dataset from HuggingFace
        print(f'Loading WikiText-103 {split} split...')
        dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split)

        # Combine all texts and tokenize
        print('Tokenizing text...')
        enc = tiktoken.get_encoding('gpt2')
        all_text = '\n\n'.join(dataset['text'])
        tokens = enc.encode(all_text)
        self.tokens = torch.tensor(tokens)

        print(f'Loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

    def __len__(self):
        return len(self.tokens) // (self.B * self.T)
