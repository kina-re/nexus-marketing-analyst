import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- CONFIGURATION ---
MMM_DATA_PATH = r"C:\Users\Krina\atlas\data\mmm_data_2016_2017.csv" 

def normalize_mmm_channels(df):
    """Same logic as main.py to ensure we check the same data"""
    mapping = {
        'Social_Media': 'social',
        'Facebook_Ads': 'social',
        'Paid_Search': 'paid_search',
        'Display_Ads': 'display',
        'Affiliates': 'affiliates',
        'TV_Campaign': 'tv',
        'Billboard_Ads': 'billboard'
    }
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns=mapping)

def calculate_vif(df):
    """
    Calculates Variance Inflation Factor (VIF).
    VIF > 5-10 indicates 'Multicollinearity' (The model is confused).
    """
    from sklearn.linear_model import LinearRegression
    
    # --- FIX: Calculate list first, then build DataFrame ---
    vif_scores = []
    
    for i in range(len(df.columns)):
        # Regress each feature against all others
        cols = [c for c in df.columns if c != df.columns[i]]
        y = df[df.columns[i]]
        X = df[cols]
        
        model = LinearRegression().fit(X, y)
        r_squared = model.score(X, y)
        
        # VIF formula: 1 / (1 - R^2)
        if r_squared < 1.0:
            vif = 1 / (1 - r_squared)
        else:
            vif = float('inf')
            
        vif_scores.append(round(vif, 2))
        
    vif_data = pd.DataFrame({
        "feature": df.columns,
        "VIF": vif_scores
    })
        
    return vif_data.sort_values("VIF", ascending=False)

def run_sanity_checks():
    print(">>> RUNNING DATA SANITY CHECKS...")
    
    # 1. Load & Clean
    try:
        raw_df = pd.read_csv(MMM_DATA_PATH)
    except FileNotFoundError:
        print("Error: MMM data not found.")
        return

    df = normalize_mmm_channels(raw_df)
    
    # Handle duplicates (e.g. FB + Social)
    # Fix the deprecation warning by explicitly selecting numeric columns if needed
    # For now, we assume all columns except Date are numeric
    if 'Date' in df.columns:
        date_col = df['Date']
        df = df.drop(columns=['Date'])
    
    # Sum duplicates
    df = df.groupby(lambda x: x, axis=1).sum()
    
    # Separate Spend (Features) vs Revenue (Target)
    target = df['Revenue']
    features = df.drop(columns=['Revenue', 'Conversions'])
    
    # --- CHECK 1: CORRELATION MATRIX ---
    print("\n[1] CORRELATION MATRIX (High correlation = Risk of Negative ROI)")
    corr = features.corr()
    print(corr.round(2))
    
    # Find highly correlated pairs (> 0.8)
    print("\n--- WARNINGS: HIGH CORRELATION DETECTED ---")
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.8:
                pair = f"{corr.columns[i]} <--> {corr.columns[j]}"
                score = corr.iloc[i, j]
                print(f"  ! {pair}: {score:.2f}")
                high_corr_pairs.append(pair)
                
    if not high_corr_pairs:
        print("  (None found - Data looks independent)")

    # --- CHECK 2: VARIANCE INFLATION FACTOR (VIF) ---
    print("\n[2] VIF SCORES (VIF > 10 means the model is breaking)")
    vif_df = calculate_vif(features)
    print(vif_df)
    
    # --- CHECK 3: SPEND SHARE vs REVENUE CORRELATION ---
    print("\n[3] RAW SIGNAL STRENGTH (R-Squared against Revenue)")
    print("    (Does this channel actually move with Revenue?)")
    
    signal_strength = []
    for col in features.columns:
        # Simple linear regression of Channel vs Revenue
        X = features[[col]]
        y = target
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)
        signal_strength.append({'channel': col, 'R2_with_Revenue': r2})
        
    signal_df = pd.DataFrame(signal_strength).sort_values("R2_with_Revenue", ascending=False)
    print(signal_df.round(3))
    
    print("\n>>> SANITY CHECK COMPLETE")
    print("INTERPRETATION GUIDE:")
    print("1. If Display has High VIF (>10) or High Correlation with Paid Search, that explains the negative ROI.")
    print("2. If Paid Search has high R2 with Revenue, that explains why the model gave it 50%.")

if __name__ == "__main__":
    run_sanity_checks()