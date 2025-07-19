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

# Training loop
for review_text, label in reviews:  # Unpack the tuple
    tokenized_review = tokenizer.tokenize(review_text)
    prediction = model(tokenized_review)
    

    
