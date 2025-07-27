import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SentimentLSTM, self).__init__()
        
        # Layer 1: Convert word IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Layer 2: Process sequence with memory (Layer size 2)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers= 3, batch_first=True)

        # Layer 3: Add dropout to prevent over fitting
        self.dropout = nn.Dropout(0.5)
        
        # Layer 4: Final classification
        self.fc = nn.Linear(hidden_size, 1)  # 1 output for binary classification
        
    def forward(self, x):
        # Step 1: Convert word IDs to word vectors
        embedded = self.embedding(x)
        
        # Step 2: Process through LSTM
        lstm_output, hidden = self.lstm(embedded)
        
        # Step 3: Get final sentiment representation
        final_output = lstm_output[:, -1, :]

        # Step 4: Use Dropout
        final_output = self.dropout(final_output)
        
        # Step 5: Convert to sentiment probability (0-1 range)
        output = torch.sigmoid(self.fc(final_output))
        
        return output