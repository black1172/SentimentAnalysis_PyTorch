# Test the tokenizer
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