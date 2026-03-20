import numpy as np

def build_features(df):
    df = df.copy()
    
    # Log transform the 'Amount' feature to reduce skewness
    df["amount_log"] = np.log1p(df["Amount"])
    
    # Transform time to hour of the day
    df["hour"] = (df["Time"] // 3600) % 24
    
    # Suspicious activity hours
    df["is_night"] = df["hour"].apply(lambda x: 1 if (x < 6 or x > 22) else 0)
    
    # Simple normalization
    df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
    
    return df