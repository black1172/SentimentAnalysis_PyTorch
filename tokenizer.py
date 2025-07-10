import string

# Create tokenizer.py
class SimpleTokenizer:
    def __init__(self):
        # We need a map: word → number
        self.word_to_id = {}  # Like {"movie": 45, "good": 67}
        # Might also need the reverse: number → word (for debugging)
        self.id_to_word = {}  # Like {45: "movie", 67: "good"}
    
    def clean_text(text):
        # Handle punctuation
        for punct in string.punctuation:
            sentence = sentence.replace(punct, '')

        # Handle upper lower
        sentence = sentence.lower()
        return sentence

    def build_vocab(self, texts):
        unique_words = set()  # Create empty set
                
        for sentence in texts:
            sentence = SimpleTokenizer.clean_text(unique_words) # get clean text
            words = sentence.split() # Chop up sentence
            unique_words.update(words)  # Add words to set

        return unique_words   
    
    def tokenize(self, text):
        # TODO: How do we convert "This movie rocks!" to numbers?
        