import string

# Create tokenizer.py
class SimpleTokenizer:
    def __init__(self):
        # We need a map: word → number
        self.word_to_id = {}  # Like {"movie": 45, "good": 67}
        # Might also need the reverse: number → word (for debugging)
        self.id_to_word = {}  # Like {45: "movie", 67: "good"}
    
    def clean_text(self, text):
        # Handle punctuation
        for punct in string.punctuation:
            text = text.replace(punct, '')

        # Handle upper lower
        text = text.lower()
        return text

    def build_vocab(self, texts):
        unique_words = set()  # Create empty set
                
        # process all reviews        
        for sentence in texts:
            clean_sentence = self.clean_text(sentence) # get clean text
            words = clean_sentence.split() # Chop up sentence
            unique_words.update(words)  # Add words to set

        # After building unique_words set
        for i, word in enumerate(unique_words):
            self.word_to_id[word] = i
            self.id_to_word[i] = word 
    
    def tokenize(self, text):
        # Step 1: Clean the text (same as build_vocab)
        clean_text = self.clean_text(text)
        
        # Step 2: Split into words
        words = clean_text.split()
        
        # Step 3: Convert each word to its number
        token_ids = []
        for word in words:
            token_ids.append(self.word_to_id.get(word))

        return token_ids