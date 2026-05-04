# Chatbot Analytics Framework

## 🚀 Overview
The **Chatbot Analytics Framework** is a specialized NLP and Data Engineering pipeline designed to enhance Conversational AI systems. It provides automated tools for cleaning chat logs, performing sentiment analysis, discovering latent topics, and generating business-centric dashboards.

## 🛠️ Technology Stack
- **Language**: Python 3.x
- **NLP**: `spaCy` (Lemmatization), `NLTK VADER` (Sentiment Analysis)
- **Machine Learning**: `scikit-learn` (LDA Topic Modeling)
- **Data Handling**: `pandas`
- **Visualization**: `seaborn`, `matplotlib`

## 📁 Directory Structure
```text
chatbot/
├── data/
│   ├── raw/                # Original datasets (CSV/JSON)
│   └── processed/          # Data after cleaning and analysis
├── models/                 # Saved visualizations and trained models
├── src/
│   ├── modules/            # Core logic (cleaning, analysis, visualization)
│   └── utils/              # Helper functions
├── main.py                 # Pipeline entry point
├── requirements.txt        # Project dependencies
└── venv/                   # Python virtual environment
```

## ✨ Key Features
- **Advanced Cleaning**: Automated lowercasing, regex-based special character removal, and spaCy-powered lemmatization.
- **Sentiment Intelligence**: Multi-category sentiment labeling (Positive, Neutral, Negative) using the VADER lexicon.
- **Topic Discovery**: Latent Dirichlet Allocation (LDA) to identify top user intents and conversational themes.
- **Business EDA**: Generation of dashboards showing sentiment trends and top support ticket categories.

## ⚙️ Installation & Setup

1. **Initialize Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python3 -m spacy download en_core_web_sm
   ```

## 🚀 Usage
Run the entire pipeline (Load → Clean → Analyze → Visualize) with a single command:
```bash
python3 main.py
```

## 📊 Datasets Used
- **`nlp_customer_support_classification.csv`**: Primary source for NLP logic and sentiment benchmarking.
- **`customer_support_tickets.csv`**: Used for generating Business EDA, intent distribution, and trend charts.

## 👤 Author
**Senior Data Engineer & NLP Specialist**
*Chatbot Analytics Project*
