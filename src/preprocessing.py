import pandas as pd
import numpy as np
import os
from datetime import datetime

def main():
    print("Loading data from data/movies.csv")
    df = pd.read_csv("data/movies.csv")
    
    print("Processing data...")
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    
    df['title_length'] = df['title'].str.len()
    df['overview_length'] = df['overview'].fillna('').str.len()
    
    df['log_popularity'] = np.log1p(df['popularity'])
    df['log_vote_count'] = np.log1p(df['vote_count'])
    df['high_rating'] = (df['vote_average'] > 7.0).astype(int)
    
    df['is_recent'] = (df['release_year'] >= 2000).astype(int)
    os.makedirs("data", exist_ok=True)
    
    output_path = "data/processed_movies.csv"
    print(f"Saving processed data to {output_path}")
    df.to_csv(output_path, index=False)
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()