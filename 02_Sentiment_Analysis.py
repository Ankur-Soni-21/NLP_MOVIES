# Import required libraries
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path
from utils.logging_config import configure_logging
logger = configure_logging()

# Define input and output paths
INPUT_PATH_PKL = Path('output/data/01_Clean_Data.pkl')
INPUT_PATH_CSV = Path('output/data/01_Clean_Data.csv')
OUTPUT_PATH_DATA = Path('output/data')
OUTPUT_PATH_VISUAL = Path('output/02')

# Output files
SENTIMENT_PLOT = OUTPUT_PATH_VISUAL / '02_Sentiment_Analysis.png'
TEMPORAL_PLOT = OUTPUT_PATH_VISUAL / '03_Temporal_Sentiment_Analysis.png'
PROCESSED_DATA_PKL = OUTPUT_PATH_DATA / '02_Sentiment_Data.pkl'
PROCESSED_DATA_CSV = OUTPUT_PATH_DATA / '02_Sentiment_Data.csv'

def load_data(file_path):
    if str(file_path).endswith('.csv'):
        logger.info(f"Data loaded from {file_path}")
        return pd.read_csv(file_path)
    elif str(file_path).endswith('.pkl'):
        logger.info(f"Data loaded from {file_path}")
        return pd.read_pickle(file_path)
    else:
        logger.error("Unsupported file format. Please use CSV or PKL file.")
        raise ValueError("Unsupported file format. Please use CSV or PKL file.")

def calculate_sentiment_metrics(df):
    def get_sentiment(text):
        blob = TextBlob(str(text))
        return pd.Series({
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        })
    
    logger.info("Calculating sentiment metrics...")
    # Calculate sentiment metrics
    sentiment_df = df['transcript'].apply(get_sentiment)
    
    # Add new columns to original dataframe
    df['polarity'] = sentiment_df['polarity']
    df['subjectivity'] = sentiment_df['subjectivity']
    
    logger.info("Sentiment metrics calculated successfully!")
    return df

def plot_sentiment_analysis(df, output_path):
    logger.info("Plotting sentiment analysis...")
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot
    plt.scatter(df['polarity'], df['subjectivity'], color='red', alpha=0.5)
    
    # Customize plot
    plt.title('Sentiment Analysis', fontsize=20)
    plt.xlabel('-ve                           +ve', fontsize=12)
    plt.ylabel('Facts                        Opinions', fontsize=12)
    
    # Calculate x-axis limits to center around 0
    max_abs_polarity = max(abs(df['polarity'].max()), abs(df['polarity'].min()))
    plt.xlim(-max_abs_polarity, max_abs_polarity)
    
    # Add vertical line at x=0
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Sentiment analysis plot saved to: {output_path}")
    plt.close()

def main():
    """
    Main function to orchestrate the entire sentiment analysis process
    """
    try:
        # Create output directory if it doesn't exist
        OUTPUT_PATH_VISUAL.mkdir(parents=True, exist_ok=True)
        
        # Try to load PKL file first, if not available, load CSV
        try:
            df = load_data(INPUT_PATH_PKL)
        except FileNotFoundError:
            df = load_data(INPUT_PATH_CSV)
        
        # Calculate overall sentiment metrics
        df = calculate_sentiment_metrics(df)
        
        # Plot and save overall sentiment analysis
        plot_sentiment_analysis(df, SENTIMENT_PLOT)
        
        df.to_pickle(PROCESSED_DATA_PKL)
        df.to_csv(PROCESSED_DATA_CSV, index=False)
        
        
        
        print("Complete sentiment analysis completed successfully!")
        print(f"Processed data saved to: {PROCESSED_DATA_PKL}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()