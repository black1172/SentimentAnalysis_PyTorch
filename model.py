import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SentimentRNN, self).__init__()
        
        # Layer 1: Convert word IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Layer 2: Process sequence with memory
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        
        # Layer 3: Final classification
        self.fc = nn.Linear(hidden_size, 1)  # 1 output for binary classification
        
    def forward(self, x):
        
        pass