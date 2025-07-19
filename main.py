from model import SentimentRNN
from tokenizer import SimpleTokenizer
import data as d

# Get data
reviews = d.get_sample_data()

# Build maps
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(reviews[0 ,:])

# Create model with specific sizes
vocab_size = len(tokenizer.word_to_id)  # Number of unique words
embed_size = 50    # Size of word vectors
hidden_size = 64   # Size of RNN memory
model = SentimentRNN(vocab_size, embed_size, hidden_size)

# Display Vocab
print("Vocabulary:", tokenizer.word_to_id)

# Training setup
criterion = # TODO: What loss function for binary classification?
optimizer = # TODO: What optimizer? (hint: like your CNN project)

# Training loop
for review_text, label in reviews:
    # Step 1: Convert text to tokens
    tokens = # TODO: Use tokenizer to convert review_text
    
    # Step 2: Convert to tensor and add batch dimension
    input_tensor = # TODO: Convert tokens list to tensor, add batch dim [1, seq_len]
    target_tensor = # TODO: Convert label to tensor
    
    # Step 3: Forward pass (your description is perfect!)
    # - Embedding layer converts tokens to vectors
    # - RNN processes sequence and builds memory  
    # - Linear layer + sigmoid squeezes to [0,1]
    prediction = # TODO: Pass input through model
    
    # Step 4: Calculate loss
    loss = # TODO: Compare prediction to target
    
    # Step 5: Backward pass and optimization
    # TODO: Zero gradients, backward pass, optimizer step
    
    print(f"Text: {review_text}")
    print(f"Prediction: {prediction.item():.3f}, Actual: {label}")
    print(f"Loss: {loss.item():.3f}\n")
    

    
