"""
Generate a test dataset from the wine quality training data.

This script loads the wine quality dataset and creates a smaller test dataset
by sampling randomly from the original dataset.
"""
import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate test data from wine quality dataset')
    parser.add_argument('--input', type=str, default='../data/winequality-red.csv',
                        help='Path to the input wine quality dataset')
    parser.add_argument('--output', type=str, default='../data/wine_test_samples.csv',
                        help='Path to save the test samples')
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of samples to include in the test dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Create the data directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load the wine quality dataset
    try:
        data = pd.read_csv(args.input, sep=';')
        print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns from {args.input}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Sample rows randomly
    test_data = data.sample(n=min(args.samples, len(data)), random_state=args.seed)
    print(f"Created test dataset with {test_data.shape[0]} samples")
    
    # Save the test dataset
    try:
        test_data.to_csv(args.output, sep=';', index=False)
        print(f"Test dataset saved to {args.output}")
    except Exception as e:
        print(f"Error saving test dataset: {e}")

if __name__ == "__main__":
    main()