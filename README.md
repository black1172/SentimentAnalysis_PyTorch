# Movie Review Sentiment Analysis with RNN/LSTM

A PyTorch implementation of a recurrent neural network (RNN) for sentiment classification of movie reviews. This project demonstrates sequence modeling, text preprocessing, and approaches to mitigating overfitting in natural language processing (NLP).

## Project Overview

This project applies RNN and LSTM architectures to classify movie reviews as positive or negative. It explores:

- Custom tokenization and vocabulary management
- Word embeddings for dense text representations
- Architectural improvements including dropout regularization and bidirectional LSTMs to improve generalization

The goal is to understand both the capabilities and limitations of sequence models for sentiment analysis.

## Technical Highlights

**Architecture:**

- Bidirectional LSTM with dropout regularization
- Embedding layer for word representations
- Linear classification layer with sigmoid activation

```python
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, 1)
```

- **Sequence Modeling:** Captures sequential dependencies in text, with bidirectional LSTM providing context from both past and future tokens.

- **Regularization & Overfitting:** Dropout and architectural experimentation help mitigate classic overfitting seen in training vs. test performance.

- **Training Details:** Binary cross-entropy loss, Adam optimizer, and systematic hyperparameter tuning.

## Key Learning Outcomes

- **Text Processing:** Built custom tokenizer, managed vocabulary, and handled out-of-vocabulary words.

- **Sequence Modeling:** Gained hands-on experience with RNN memory mechanisms, LSTM improvements, and bidirectional processing.

- **Overfitting & Generalization:** Learned to identify overfitting, apply dropout, and adjust architectures to improve model robustness.

- **NLP Workflow:** Practiced end-to-end pipeline design: text preprocessing → embedding → sequential modeling → binary classification.

## Files & Components

- `tokenizer.py` – Custom tokenizer and vocabulary management
- `model.py` – RNN/LSTM architecture definitions
- `data.py` – Training data preparation and preprocessing
- `main.py` – Training loop, evaluation, and experiment tracking

## Impact & Use Case

This project highlights the challenges of sequence modeling for text classification, illustrating how advanced RNN architectures like bidirectional LSTMs can improve context understanding while still facing generalization challenges. It’s a clear demonstration of practical NLP engineering, applicable to tasks like sentiment analysis, chatbots, or recommendation systems.

## Technologies Used

- Python 3.11+
- PyTorch for model implementation and training
- NumPy & standard Python libraries for preprocessing and data handling