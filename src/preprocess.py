"""
Text Preprocessing Module for Customer Support Ticket Classification
Team Batch 10, Project 6: Customer Support Ticket Classification

This module provides a unified text preprocessing pipeline shared across:
1. Model training (src/train.py)
2. Interactive evaluation & notebooks (notebooks/ticket_classification.ipynb)
3. Production Streamlit inference (app.py)
4. Offline prediction CLI (src/predict.py)

Key Preprocessing Decisions & NLP Justifications:
------------------------------------------------
1. Lowercasing:
   Standardizes casing so that 'Billing', 'billing', and 'BILLING' map to the same
   lexical token, reducing vocabulary sparsity.

2. URL & Email Normalization/Removal:
   Support tickets frequently contain customer emails (e.g. user@corp.com) and links.
   These contain unique random domains that inflate feature dimensions without adding
   category signals. Removing them focuses features on intent-bearing vocabulary.

3. Contraction Expansion:
   Expands contractions like "can't" to "cannot", "won't" to "will not", ensuring negation
   tokens are consistently preserved before punctuation stripping.

4. Negation-Preserving Stopword Filtering:
   Default stopword lists blindly drop words like 'not', 'no', 'never', 'cannot'.
   Dropping them would turn "I did NOT receive my parcel" into "receive parcel",
   severely corrupting semantic meaning. We curate an explicit whitelist of negation
   words that are always retained.

5. Lemmatization (WordNet):
   Converts inflected forms (e.g., 'charges' -> 'charge', 'crashed' -> 'crash',
   'deliveries' -> 'delivery') to their canonical base dictionary lemma. Unlike stemming,
   lemmatization preserves genuine dictionary words. A fallback stemmer is included
   for extreme offline resilience.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Contraction mapping for common English expressions
CONTRACTIONS = {
    "can't": "cannot",
    "cant": "cannot",
    "won't": "will not",
    "wont": "will not",
    "don't": "do not",
    "dont": "do not",
    "didn't": "did not",
    "didnt": "did not",
    "doesn't": "does not",
    "doesnt": "does not",
    "isn't": "is not",
    "isnt": "is not",
    "aren't": "are not",
    "arent": "are not",
    "wasn't": "was not",
    "wasnt": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not"
}

# Negation words that MUST NEVER be removed as stopwords
NEGATION_WORDS = {
    "no", "not", "nor", "neither", "never", "cannot", "none", "nobody", "nowhere", "nothing"
}

# Ensure NLTK datasets are downloaded silently on cold start (e.g. Streamlit Cloud)
for res_type, pkg in [("corpora", "stopwords"), ("corpora", "wordnet"), ("tokenizers", "punkt"), ("tokenizers", "punkt_tab")]:
    try:
        nltk.data.find(f"{res_type}/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

# Initialize stopwords with negation words removed
try:
    standard_stopwords = set(stopwords.words("english"))
except Exception:
    # Safe fallback if corpus unavailable
    standard_stopwords = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
        "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
        "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
        "about", "against", "between", "into", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "just", "should", "now"
    }

CUSTOM_STOPWORDS = standard_stopwords - NEGATION_WORDS

# Initialize lemmatizer and fallback stemmer
try:
    LEMMATIZER = WordNetLemmatizer()
    # Test lemmatizer functionality
    _ = LEMMATIZER.lemmatize("tickets")
except Exception:
    LEMMATIZER = None

STEMMER = PorterStemmer()

def expand_contractions(text: str) -> str:
    """Expand common English contractions to preserve negation components."""
    tokens = text.split()
    expanded = []
    for token in tokens:
        clean = token.lower().strip(string.punctuation)
        if clean in CONTRACTIONS:
            expanded.append(CONTRACTIONS[clean])
        else:
            expanded.append(token)
    return " ".join(expanded)

def clean_text(text: str) -> str:
    """
    Apply low-level regex cleaning:
    - Lowercase text
    - Strip URLs and email addresses
    - Strip ticket reference numbers (e.g. #ORD-12345, #TXN9982)
    - Remove extra punctuation and digits while preserving words
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Expand contractions (e.g., can't -> cannot)
    text = expand_contractions(text)
    
    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    
    # 4. Remove email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)
    
    # 5. Remove reference tags and numbers (e.g., #10045, #TXN2819, $150)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"\$\d+", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    
    # 6. Remove remaining punctuation and non-alphanumeric characters (except whitespace)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # 7. Normalize multi-whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def preprocess_text(text: str) -> str:
    """
    Complete NLP preprocessing pipeline:
    1. Cleaning & regex filtering
    2. Tokenization
    3. Negation-aware stopword removal
    4. WordNet Lemmatization (with Porter fallback)
    5. Reassembly into normalized text string for TF-IDF vectorization
    """
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    
    # Tokenization: use NLTK word_tokenize with regex fallback
    try:
        tokens = word_tokenize(cleaned)
    except Exception:
        tokens = re.findall(r"\b[a-zA-Z]+\b", cleaned)
        
    processed_tokens = []
    for token in tokens:
        if token not in CUSTOM_STOPWORDS and len(token) > 1:
            # Lemmatize noun and verb forms
            if LEMMATIZER:
                try:
                    lemma = LEMMATIZER.lemmatize(token, pos="v")
                    lemma = LEMMATIZER.lemmatize(lemma, pos="n")
                except Exception:
                    lemma = STEMMER.stem(token)
            else:
                lemma = STEMMER.stem(token)
            processed_tokens.append(lemma)
            
    return " ".join(processed_tokens)

if __name__ == "__main__":
    sample_text = "I did NOT receive order #ORD-98214! The app crashes and can't send me my refund of $120. Email: test@example.com."
    print("Raw text:")
    print(sample_text)
    print("\nCleaned & Preprocessed text:")
    print(preprocess_text(sample_text))
