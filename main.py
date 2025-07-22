from model import SentimentRNN
from tokenizer import SimpleTokenizer
import data as d
import torch.nn as nn
import torch.optim as optim
import torch

# Get data
reviews = d.get_sample_data()

# Extract just text for vocabulary building
texts = [review_text for review_text, label in reviews]

# Build vocabulary
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(texts)

# Create model with specific sizes
vocab_size = len(tokenizer.word_to_id)  # Number of unique words
embed_size = 50    # Size of word vectors
hidden_size = 64   # Size of RNN memory
model = SentimentRNN(vocab_size, embed_size, hidden_size)

# Training setup
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
final_average_loss = 0
for x in range(100):
    for review_text, label in reviews:

        # Step 1: Convert text to tokens
        tokens = tokenizer.tokenize(review_text)
        
        # Step 2: Convert to tensor and add batch dimension (can use brackets for batch dim)
        input_tensor = torch.tensor([tokens])
        target_tensor = torch.tensor([label], dtype=torch.float32)
        
        # Step 3: Forward pass
        prediction = model(input_tensor)   # implicit forward call by pytorch
        
        # Step 4: Calculate loss
        loss = criterion(prediction.squeeze(-1), target_tensor) # -1 removes all but first dim
        
        # Step 5: Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_average_loss += loss.item()
        
# Printing final
for review_text, label in reviews:
    print(f"Text: {review_text}")
    print(f"Prediction: {prediction.item():.3f}, Actual: {label}")
    print(f"Loss: {loss.item():.3f}\n")
        
# Final Loss reporting
final_average_loss /= len(reviews)
print(f"Final Average loss: {final_average_loss: .3f}")