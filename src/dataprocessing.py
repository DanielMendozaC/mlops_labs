"""
Data processing utilities for wine quality prediction.

This module contains functions for loading, preprocessing, and transforming
wine quality data for ML model training and inference.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Union

def load_wine_data(filepath: str, separator: str = ';') -> pd.DataFrame:
    """
    Load wine quality data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        separator: CSV separator character (default: ';')
        
    Returns:
        DataFrame containing the wine quality data
    """
    try:
        df = pd.read_csv(filepath, sep=separator)
        print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        raise

def split_features_target(data: pd.DataFrame, 
                         target_col: str = 'class') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the data into features and target.
    
    Args:
        data: DataFrame containing the wine data
        target_col: Name of the target column (default: 'class')
        
    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series
    """
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data. Available columns: {data.columns.tolist()}")
    
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    return X, y

def clean_wine_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the wine data by handling missing values and outliers.
    
    Args:
        data: DataFrame containing the wine data
        
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Found {missing_values.sum()} missing values")
        # For simplicity, we'll fill missing values with column means
        df = df.fillna(df.mean())
    
    # Check for outliers using IQR method
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"Found {outliers} outliers in column '{col}'")
            
            # Cap outliers rather than removing them
            df[col] = df[col].clip(lower_bound, upper_bound)
    
    return df

def create_feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from the existing data.
    
    Args:
        data: DataFrame containing the wine data
        
    Returns:
        DataFrame with additional engineered features
    """
    df = data.copy()
    
    # Print columns to debug
    print("Available columns:", df.columns.tolist())
    
    # Check if columns exist before using them
    if 'fixed_acidity' in df.columns and 'volatile_acidity' in df.columns:
        # Total acidity
        df['total_acidity'] = df['fixed_acidity'] + df['volatile_acidity']
        
        # Alcohol to acidity ratio (if alcohol column exists)
        if 'alcohol' in df.columns:
            df['alcohol_to_acidity_ratio'] = df['alcohol'] / (df['volatile_acidity'] + 0.01)
    
    # Check if sulfur dioxide columns exist
    if 'free_sulfur_dioxide' in df.columns and 'total_sulfur_dioxide' in df.columns:
        # Free to total sulfur dioxide ratio
        df['free_to_total_sulfur_ratio'] = df['free_sulfur_dioxide'] / (df['total_sulfur_dioxide'] + 0.01)
    
    return df

def preprocess_wine_data(
    data: pd.DataFrame, 
    target_col: str = 'class',
    feature_engineering: bool = True,
    clean_data: bool = True
) -> Tuple[pd.DataFrame, Optional[Union[pd.Series, None]]]:
    """
    Complete preprocessing pipeline for wine data.
    
    Args:
        data: DataFrame containing the wine data
        target_col: Name of the target column (default: 'class')
        feature_engineering: Whether to create new features (default: True)
        clean_data: Whether to clean the data (default: True)
    
    Returns:
        Tuple of (X, y) where X is processed features and y is target (if available)
    """
    df = data.copy()
    
    # Clean data
    if clean_data:
        df = clean_wine_data(df)
    
    # Feature engineering
    if feature_engineering:
        df = create_feature_engineering(df)
    
    # Split features and target if target column exists
    y = None
    if target_col in df.columns:
        X, y = split_features_target(df, target_col)
        print(f"Target column '{target_col}' found. Feature shape: {X.shape}, Target shape: {y.shape}")
    else:
        X = df
        print(f"Warning: Target column '{target_col}' not found in data. Available columns: {df.columns.tolist()}")
    
    return X, y