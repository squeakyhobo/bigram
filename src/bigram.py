import torch.nn as nn


class bigram(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.vocab_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self,x):
        logits = self.vocab_embedding_table(x) # raw data not normalised
        return logits