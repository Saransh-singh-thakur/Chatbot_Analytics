import re
import spacy
from typing import List

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback if model not found, though we should have downloaded it
    nlp = None

def clean_text(text: str) -> str:
    """
    Perform basic text cleaning: lowercase and remove special characters.
    
    Args:
        text (str): Input text to clean.
        
    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters using Regex
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def lemmatize_text(text: str) -> str:
    """
    Perform lemmatization using spaCy.
    
    Args:
        text (str): Input text to lemmatize.
        
    Returns:
        str: Lemmatized text.
    """
    if not nlp or not isinstance(text, str):
        return text
        
    doc = nlp(text)
    lemmatized = [token.lemma_ for token in doc]
    return " ".join(lemmatized)

def preprocess_pipeline(text: str) -> str:
    """
    Full cleaning and lemmatization pipeline.
    
    Args:
        text (str): Raw input text.
        
    Returns:
        str: Preprocessed text.
    """
    cleaned = clean_text(text)
    return lemmatize_text(cleaned)
