# Sample movie reviews with sentiment labels
def get_sample_data():
    # Format: (review_text, label)
    # 1 = positive, 0 = negative
    reviews = [
        ("This movie was amazing", 1),
        ("Awesome film loved it", 1),
        ("Fantastic movie really enjoyed", 1),
        ("Best movie ever seen", 1),
        ("Wonderful acting and story", 1),
        
        ("Terrible movie hated it", 0),
        ("Worst film ever made", 0),
        ("Bad acting terrible story", 0),
        ("Boring movie fell asleep", 0),
        ("Awful film waste of time", 0),
    ]
    return reviews

# Sample data that the model is not trained on
def get_untrained_reviews():
    # Format: (review_text, label)
    # 1 = positive, 0.5 = neutral, 0 = bad
    reviews = [
        ("This was a good piece of cinema.", 1),
        ("Not good, but not bad.", 0.5),
        ("What a bad no good horrible movie.", 0),
        ("Was a great movie 10/10.", 1),
    ]
    return reviews