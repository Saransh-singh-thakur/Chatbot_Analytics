import pandas as pd
import sys
import os
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Add src to path if needed
sys.path.append(os.path.join(os.getcwd(), 'src'))

from modules.cleaning import preprocess_pipeline
from modules.analysis import perform_sentiment_analysis, perform_topic_modeling, get_top_words_per_topic
from modules.visualization import create_dashboard

def main():
    """
    Main orchestration pipeline for Chatbot Analytics.
    """
    print("--- Starting Chatbot Analytics Pipeline ---")
    
    # 1. Load Data
    nlp_data_path = "data/raw/nlp_customer_support_classification.csv"
    business_data_path = "data/raw/customer_support_tickets.csv"
    
    if not os.path.exists(nlp_data_path) or not os.path.exists(business_data_path):
        print("Error: Dataset files not found in data/raw/")
        return

    print("Loading datasets...")
    nlp_df = pd.read_csv(nlp_data_path)
    business_df = pd.read_csv(business_data_path)
    
    # 2. Cleaning & Preprocessing (NLP Logic)
    print("Preprocessing NLP text data...")
    # Using 'text' column from nlp_customer_support_classification.csv
    nlp_df['processed_text'] = nlp_df['text'].apply(preprocess_pipeline)
    
    # 3. Sentiment Analysis
    print("Performing Sentiment Analysis...")
    nlp_df = perform_sentiment_analysis(nlp_df, 'text')
    
    # 4. Topic Modeling (LDA)
    print("Running Topic Modeling...")
    lda_model, vectorizer, feature_names = perform_topic_modeling(nlp_df['processed_text'].tolist())
    topics = get_top_words_per_topic(lda_model, feature_names)
    
    print("\nDiscovered Topics:")
    for idx, words in topics.items():
        print(f"Topic {idx}: {', '.join(words)}")
    
    # 5. Visualization & EDA
    print("\nGenerating Visualizations...")
    # Use nlp_df for sentiment distribution
    # Use business_df for business EDA as requested
    
    # We need to perform sentiment analysis on business_df too if we want to show it in the dashboard
    business_df = perform_sentiment_analysis(business_df, 'Ticket Description')
    
    # Create dashboard using business data for "Business EDA" as requested
    # We use 'Ticket Type' as the intent column for business charts
    create_dashboard(business_df, intent_col='Ticket Type', output_dir='models')
    
    # Save processed data
    print("\nSaving processed data...")
    nlp_df.to_csv("data/processed/processed_nlp_logs.csv", index=False)
    business_df.to_csv("data/processed/processed_business_tickets.csv", index=False)
    
    print("--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()
