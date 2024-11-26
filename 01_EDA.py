# os for file operations
import os
# logging for logging
import logging
# warnings to suppress warnings
import warnings

# pandas for datframes
import pandas as pd
# seaborn for visualizations
import seaborn as sns

# tqdm for progress bars
from tqdm import tqdm

# typing for data types
from typing import List
from typing import Tuple
from typing import Optional

# imdb for fetching movie information
from imdb import Cinemagoer
import concurrent.futures


# langdetect for language detection
from langdetect import detect
from langdetect import DetectorFactory

# nltk for text processing
from nltk import word_tokenize
from nltk.corpus import stopwords

# matplotlib for visualizations
import matplotlib.pyplot as plt
    
# wordcloud for word clouds
from wordcloud import WordCloud

# gensim for text processing
from gensim.utils import simple_preprocess

# configure logging settings
from utils.logging_config import configure_logging
logger = configure_logging()

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set seed for consistent language detection
DetectorFactory.seed = 0

INPUT_FILE_PATHS  = [
    'csv/0-100.csv',
    'csv/101-200.csv',
    'csv/201-300.csv',
    'csv/301-400.csv',
    'csv/401-500.csv',
    'csv/501-600.csv',
    'csv/601-700.csv',
    'csv/701-800.csv',
]
OUTPUT_FILE_PATH_CSV = 'output/data/01_Clean_Data.csv'
OUTPUT_FILE_PATH_PKL = 'output/data/01_Clean_Data.pkl'


def combine_csv_files(file_paths: List[str], num_of_records : int) -> pd.DataFrame:
    try:
        # Create an empty list to store individual DataFrames
        dfs = []
        
        # Read each CSV file and append to the list
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            dfs.append(df)
            logging.info(f"Successfully read file: {file_path}")
        
        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        logging.info(f"Successfully combined {len(file_paths)} files with total {len(combined_df)} records")
        
        if num_of_records > len(combined_df):
            num_of_records = len(combined_df)
            
        return combined_df.head(num_of_records)
        
    except Exception as e:
        logging.error(f"Error combining files: {e}")
        raise

def read_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Rename columns to match the expected format
        df = df.rename(columns={
            'web-scraper-start-url': 'url',
            'title': 'title',
            'date_posted': 'date_posted',
            'transcript': 'transcript'
        })
        
        # Extract comedian name from the first two words of the title
        
        
        # Drop the web-scraper-order column
        df = df.drop('web-scraper-order', axis=1, errors='ignore')
        
        # Clean data
        df = df.dropna(subset=['transcript'])  # Remove rows with no transcript
        df = df.drop_duplicates(subset=['title', 'transcript'])  # Remove duplicates
        
        
        logging.info(f"Successfully cleaned data, remaining records: {len(df)}")
        return df
        
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise

def detect_language(df: pd.DataFrame) -> pd.DataFrame:
    try:
        def safe_detect(text: str) -> str:
            try:
                # Use only first 500 characters for faster processing
                return detect(str(text)[:500])
            except:
                return 'unknown'
        
        df['language'] = df['transcript'].apply(safe_detect)
        
        # Log language distribution
        lang_counts = df['language'].value_counts()
        logging.info(f"Language distribution:\n{lang_counts}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error in language detection: {e}")
        raise
    
def get_imdb_info(df: pd.DataFrame, use_cache: str) -> pd.DataFrame:
    if use_cache == 'true':
        logging.info(f"Loading cached data from: {OUTPUT_FILE_PATH_CSV}")
        return pd.read_csv(OUTPUT_FILE_PATH_CSV)
    
    ia = Cinemagoer()  # Updated to use Cinemagoer instead of IMDb
    errors = 0

    def fetch_movie_info(title: str) -> Optional[Tuple[str, str]]:
        try:
            # Search with a cleaned and truncated title
            clean_title = str(title).split(' - ')[0].split(' | ')[0]
            results = ia.search_movie(clean_title)
            if results:
                movie = ia.get_movie(results[0].movieID)
                runtime = movie.get('runtimes', [''])[0]
                rating = movie.get('rating', '')
                if runtime and rating:
                    return runtime, rating
            
            return None
        except Exception as e:
            nonlocal errors
            errors += 1
            logging.warning(f"Error fetching info for '{title}': {e}")
            return None

    # Process only the first 10 titles
    logging.info("Fetching IMDB information...")
    
    titles = df['title']
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:  # Specify the number of threads
        results = list(tqdm(executor.map(fetch_movie_info, titles), total=len(titles), desc='Fetching IMDB info'))
        
    # Unpack results
    results = [result for result in results if result is not None]
    runtimes, ratings = zip(*results)

    # Add new columns with default values for the rest
    df['runtime'] = pd.Series(runtimes).reindex(df.index, fill_value=pd.NA)
    df['rating'] = pd.Series(ratings).reindex(df.index, fill_value=pd.NA)

    # Convert to numeric where possible
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    

    logging.info(f"IMDB info fetching complete. Failed to fetch {errors} titles")
    return df

def clean_data_with_nan_ratings(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with NaN rating
    df = df.dropna(subset=['rating'])
    logging.info(f"Removed rows with NaN ratings. Remaining records: {len(df)}")
    df.to_pickle('output/data/01_Clean_Data_Rating.pkl')
    print(df.head(20))
    return df
    

def create_rating_features(df: pd.DataFrame) -> pd.DataFrame:
    # Plot the distribution of ratings using a histogram
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(df['rating'], bins=20, kde=False)
    ax.set(title='Distribution of IMDb Ratings', xlabel='Rating', ylabel='Frequency')
    plt.savefig('output/01/01_Rating_Histogram.png')
    plt.close()
    
    # Plot the distribution of ratings using a KDE plot
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(df['rating'], fill=True, color="g")
    ax.set(title='KDE of IMDb Ratings', xlabel='Rating')
    plt.savefig('output/01/02_Rating_KDE.png')
    plt.close()
    
    return df

def plot_runtime_and_ratings(df: pd.DataFrame):
    
    # Runtime analysis
    valid_runtime = df[df.runtime > 0].runtime.astype(int)
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(valid_runtime, fill=True, color="r")
    ax.set_title('Runtime KDE')
    ax.set(xlabel='minutes')
    plt.savefig('output/01/02_Runtime_Distribution.png')
    plt.close()
    
    logging.info(f'Runtime Mean: {valid_runtime.mean():.2f}')
    logging.info(f'Runtime SD: {valid_runtime.std():.2f}')
    
    # Rating analysis
    valid_rating = df[df.rating > 0].rating
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(valid_rating, fill=True, color="g")
    ax.set_title('IMDb Rating KDE')
    plt.savefig('output/01/03_Rating_Distribution.png')
    plt.close()
    
    logging.info(f'Rating Mean: {valid_rating.mean():.2f}')
    logging.info(f'Rating SD: {valid_rating.std():.2f}')

def process_text(df: pd.DataFrame) -> pd.DataFrame:
    
    stop_words = stopwords.words('english')
    stop_words.extend(['audience', 'laughter', 'laughing', 'announcer', 'narrator', 'cos'])
    
    # Tokenize and clean words
    df['words'] = df.transcript.apply(
        lambda x: [word for word in simple_preprocess(x, deacc=True) 
                  if word not in stop_words]
    )
    
    # Word count
    df['word_count'] = df.words.apply(len)
    
    # Process swear words
    swear_words = ['fuck', 'fucking', 'fuckin', 'fucker', 'muthafucka', 
               'motherfuckers', 'motherfucke', 'motha', 'motherfucker' , 
               'shit', 'shitter', 'shitting', 'shite', 'bullshit', 'shitty']
    
    # Combine all swear words into one column
    df['swear_words'] = df.words.apply(lambda x: sum(word.lower() in swear_words for word in x))
    
    # Plot the distribution of swear words using a KDE plot
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(df['swear_words'], fill=True, color="r")
    ax.set_title('Swear Words Count KDE')
    plt.savefig('output/01/Swear_Words_Distribution.png')
    plt.close()
    
    # Remove swear words from words list
    df['words'] = df.words.apply(lambda x: [word for word in x if word not in swear_words])
    
    # Create diversity features
    df['diversity'] = df.words.apply(lambda x: len(set(x)))
    df['diversity_ratio'] = df.diversity / df.word_count
    
    return df

def plot_word_visualizations(df: pd.DataFrame):
    # F-words distribution
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(df.f_words, fill=True, color="r")
    ax.set_title('F-Words Count KDE')
    plt.savefig('output/01/04_F_Words_Distribution.png')
    plt.close()
    
    # S-words distribution
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(df.s_words, fill=True, color="r")
    ax.set_title('S-Words Count KDE')
    plt.savefig('output/01/05_S_Words_Distribution.png')
    plt.close()
    
    # Word diversity
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(df.diversity, fill=True, color="purple")
    ax.set_title('Word Diversity KDE')
    plt.savefig('output/01/06_Word_Diversity.png')
    plt.close()
    
    # Diversity ratio
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(df.diversity_ratio, fill=True, color="g")
    ax.set_title('Diversity / Total words KDE')
    plt.savefig('output/01/07_Diversity_Ratio.png')
    plt.close()
    
def plot_wordclouds(df: pd.DataFrame):
    logging.info("Creating word clouds...")
    
    wordcloud = WordCloud(
        background_color="white", 
        max_words=5000, 
        contour_width=3, 
        contour_color='midnightblue'
    )
    
     # Ensure the output directory exists
    os.makedirs('output', exist_ok=True)
    
    df_words = 0
    for _, row in df.iterrows():
        df_words += len(row['words'])
    logging.info(f"Total words in DataFrame: {df_words}")
    
    # Concatenate all words in the df.words column
    all_words = ' '.join([' '.join(words) for words in df.words])
    
    # Generate word cloud from the combined text
    wordcloud.generate(all_words)
    wordcloud.to_file('output/01/08_Wordcloud_All.png')
    logging.info("Word cloud saved for all words")
    
def process_transcript_data(file_paths: List[str]) -> pd.DataFrame:
    try:
        # Combine and process data
        combined_df = combine_csv_files(file_paths,800)
        df = read_and_clean_data(combined_df)
        df = detect_language(df)
        df = get_imdb_info(df, 'true')
        df = clean_data_with_nan_ratings(df)
        
        # Create features
        df = create_rating_features(df)
        df = process_text(df)
        
        # plot visualizations
        plot_runtime_and_ratings(df)
        plot_word_visualizations(df)
        plot_wordclouds(df)
        
        # Save processed data
        os.makedirs('output/data', exist_ok=True)
        df.to_pickle(OUTPUT_FILE_PATH_PKL)
        logging.info("Analysis completed successfully")
        return df
        
    except Exception as e:
        logging.error(f"Error in main process: {e}")
        raise
    
if __name__ == "__main__":
    
    processed_df = process_transcript_data(INPUT_FILE_PATHS)
    logging.info("\nFirst few rows of processed data:")
    logging.info(processed_df.head())
    
    # Save processed data
    processed_df.to_csv(OUTPUT_FILE_PATH_CSV, index=False)