import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from typing import Dict, List, Tuple

def perform_sentiment_analysis(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Perform sentiment analysis using NLTK VADER.
    
    Args:
        df (pd.DataFrame): DataFrame containing text data.
        text_column (str): Name of the column with text.
        
    Returns:
        pd.DataFrame: DataFrame with added sentiment scores.
    """
    sia = SentimentIntensityAnalyzer()
    
    # Calculate scores
    df['sentiment_scores'] = df[text_column].apply(lambda x: sia.polarity_scores(str(x)))
    
    # Extract compound score
    df['sentiment_compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])
    
    # Categorize sentiment
    df['sentiment_label'] = df['sentiment_compound'].apply(
        lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
    )
    
    return df

def perform_topic_modeling(texts: List[str], n_topics: int = 5) -> Tuple[LatentDirichletAllocation, CountVectorizer, List[str]]:
    """
    Perform Topic Modeling (LDA) using scikit-learn.
    
    Args:
        texts (List[str]): List of preprocessed texts.
        n_topics (int): Number of topics to discover.
        
    Returns:
        Tuple: (LDA model, Vectorizer, Feature Names)
    """
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5, learning_method='online', random_state=42)
    lda.fit(tf)
    
    feature_names = vectorizer.get_feature_names_out()
    
    return lda, vectorizer, feature_names

def get_top_words_per_topic(model: LatentDirichletAllocation, feature_names: List[str], n_top_words: int = 10) -> Dict[int, List[str]]:
    """
    Get top words for each topic discovered by LDA.
    """
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics[topic_idx] = top_words
    return topics
