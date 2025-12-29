import pandas as pd
import os

def upload_data(file_path):
    """
    Smart Ingestion: Detects if the file is for Attribution (User-Level) 
    or MMM (Aggregate) and cleans it based on your specific code needs.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read the CSV (low_memory=False prevents mixed-type warnings)
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns] # Remove accidental spaces

    filename = os.path.basename(file_path).lower()

    # --- CASE 1: MMM DATA (Time-Series) ---
    # Based on your mmm.py: Needs 'Date' and 'Revenue'
    if "mmm" in filename or "revenue" in df.columns.str.lower():
        print(f"📊 Processing MMM Time-Series: {filename}")
        
        # Standardize Date and Revenue names
        for col in df.columns:
            if col.lower() == 'date': df = df.rename(columns={col: 'Date'})
            if col.lower() == 'revenue': df = df.rename(columns={col: 'Revenue'})
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce').fillna(0)
        
        return df

    # --- CASE 2: ATTRIBUTION DATA (User-Level Paths) ---
    # Based on your markov_shapley.py: Needs 'user_id', 'session_time', 'channel', 'converted'
    else:
        print(f"👣 Processing Attribution Paths: {filename}")
        
        # Map your ga_2.csv columns to exactly what markov_shapley.py expects
        mapping = {
            'user_id': ['user_id', 'cookie', 'client_id'],
            'session_time': ['session_time', 'timestamp', 'time'],
            'converted': ['converted', 'conversion', 'is_conversion']
        }

        for target, aliases in mapping.items():
            for col in df.columns:
                if col.lower() in aliases:
                    df = df.rename(columns={col: target})
                    break

        # Convert 'converted' column (True/False or "true"/"false") to 1/0
        if 'converted' in df.columns:
            df['converted'] = df['converted'].astype(str).str.lower().map({
                'true': 1, '1': 1, '1.0': 1, 'false': 0, '0': 0, '0.0': 0
            }).fillna(0).astype(int)

        return df