from model import SentimentRNN
from tokenizer import SimpleTokenizer

# Create tokenizer first
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(your_reviews)  # Build vocabulary

# Now create model with specific sizes
vocab_size = len(tokenizer.word_to_id)  # Number of unique words
embed_size = 50    # Size of word vectors
hidden_size = 64   # Size of RNN memory

model = SentimentRNN(vocab_size, embed_size, hidden_size)
from tokenizer import SimpleTokenizer

# Sample movie reviews
reviews = [
    "This movie was great!",
    "Bad film, not good",
    "Amazing movie, really loved it",
    "Terrible, worst movie ever"
]

# Test it
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(reviews)

print("Vocabulary:", tokenizer.word_to_id)
print("\nTokenizing 'This movie rocks!':")
tokens = tokenizer.tokenize("This movie rocks!")
print("Tokens:", tokens)