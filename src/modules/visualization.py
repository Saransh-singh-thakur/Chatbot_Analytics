import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_sentiment_distribution(df: pd.DataFrame, save_path: str = "models/sentiment_dist.png"):
    """
    Plot the distribution of sentiments.
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.countplot(data=df, x='sentiment_label', palette='viridis')
    plt.title('Sentiment Distribution of Chat Logs', fontsize=15)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    plt.savefig(save_path)
    plt.close()

def plot_top_intents(df: pd.DataFrame, intent_column: str, save_path: str = "models/top_intents.png"):
    """
    Plot the most frequent user intents.
    """
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    top_intents = df[intent_column].value_counts().head(10)
    sns.barplot(x=top_intents.values, y=top_intents.index, palette='magma')
    
    plt.title('Top 10 User Intents/Topics', fontsize=15)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Intent', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_dashboard(df: pd.DataFrame, intent_col: str, output_dir: str = "models"):
    """
    Generate all visualizations.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plot_sentiment_distribution(df, os.path.join(output_dir, "sentiment_distribution.png"))
    plot_top_intents(df, intent_col, os.path.join(output_dir, "top_intents.png"))
    
    print(f"Dashboard visualizations saved to {output_dir}")
