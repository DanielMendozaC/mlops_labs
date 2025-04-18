import pandas as pd

# Try reading both wine files to see their structure
try:
    wine_samples = pd.read_csv('data/wine_samples.csv')
    print("Wine samples columns:", wine_samples.columns.tolist())
    print("Wine samples first row:", wine_samples.iloc[0])
except Exception as e:
    print(f"Error reading wine_samples.csv: {e}")

try:
    wine_test = pd.read_csv('data/wine_test_samples.csv')
    print("\nWine test samples columns:", wine_test.columns.tolist())
    print("Wine test samples first row:", wine_test.iloc[0])
except Exception as e:
    print(f"Error reading wine_test_samples.csv: {e}")

try:
    winequality = pd.read_csv('data/winequality-red.csv', sep=';')
    print("\nWine quality columns:", winequality.columns.tolist())
    print("Wine quality first row:", winequality.iloc[0])
except Exception as e:
    print(f"Error reading winequality-red.csv: {e}")