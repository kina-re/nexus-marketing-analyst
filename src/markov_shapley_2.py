import pandas as pd
import numpy as np
from itertools import chain, combinations

def create_user_journeys(df):
    """
    Sorts the dataframe by cookie and timestamp to create ordered user paths.
    """
    df = df.sort_values(['cookie', 'time'])
    df['path'] = df.groupby('cookie')['channel'].transform(lambda x: ' > '.join(x))
    
    if 'conversion' not in df.columns:
        df['conversion'] = 0 
    
    paths = df.drop_duplicates(subset=['cookie'], keep='last')[['cookie', 'path', 'conversion', 'conversion_value']]
    paths['path'] = paths.apply(lambda x: x['path'] + ' > conversion' if x['conversion'] == 1 else x['path'] + ' > drop', axis=1)
    
    return paths

def create_transition_matrix(df, paths_df):
    path_lists = paths_df['path'].str.split(' > ')
    unique_channels = set(x for l in path_lists for x in l)
    unique_channels.add('start')
    
    transitions = []
    for path in path_lists:
        full_path = ['start'] + path
        for i in range(len(full_path) - 1):
            transitions.append({
                'source': full_path[i],
                'destination': full_path[i+1]
            })
            
    trans_df = pd.DataFrame(transitions)
    prob_df = pd.crosstab(trans_df['source'], trans_df['destination'], normalize='index')
    
    return trans_df, prob_df

def extract_q_matrix(prob_matrix):
    cols = prob_matrix.columns
    absorbing = [c for c in cols if 'conversion' in c or 'drop' in c]
    transient = [c for c in cols if c not in absorbing]
    
    valid_rows = [r for r in transient if r in prob_matrix.index]
    Q = prob_matrix.loc[valid_rows, transient]
    
    for t in transient:
        if t not in Q.index:
            Q.loc[t] = 0
            
    Q = Q.fillna(0)
    return Q, list(transient)

def extract_r_matrix(prob_matrix, absorbing_states):
    cols = prob_matrix.columns
    transient = [c for c in cols if c not in absorbing_states]
    valid_absorbing = [c for c in absorbing_states if c in prob_matrix.columns]
    
    valid_rows = [r for r in transient if r in prob_matrix.index]
    R = prob_matrix.loc[valid_rows, valid_absorbing]
    
    for t in transient:
        if t not in R.index:
            R.loc[t] = 0
            
    return R.fillna(0), valid_absorbing

def compute_fundamental_matrix(Q):
    I = np.eye(len(Q))
    try:
        N = np.linalg.inv(I - Q.values)
        N_df = pd.DataFrame(N, index=Q.index, columns=Q.columns)
        return N_df
    except np.linalg.LinAlgError:
        # Singularity fix (Jitter)
        jitter = 1e-9
        N = np.linalg.inv(I - Q.values + np.eye(len(Q)) * jitter)
        N_df = pd.DataFrame(N, index=Q.index, columns=Q.columns)
        return N_df

def removal_effects(prob_matrix, absorbing_states, baseline_N, baseline_R):
    """
    Calculates removal effect. Includes FIX for matrix alignment.
    """
    # --- THE FIX IS HERE ---
    # Force R to have the exact same rows as N, filling missing rows with 0
    baseline_R = baseline_R.reindex(baseline_N.index).fillna(0)
    
    # Calculate Baseline Conversion
    if 'start' not in baseline_N.index:
        baseline_conv = 1.0
    else:
        conv_col = [c for c in absorbing_states if 'conversion' in c]
        if not conv_col:
            return pd.DataFrame()
        
        # Now this multiplication is safe because indexes match
        B = baseline_N @ baseline_R
        baseline_conv = B.loc['start', conv_col[0]]

    channels = [c for c in baseline_N.columns if c != 'start']
    removal_results = []
    
    for channel in channels:
        temp_Q, _ = extract_q_matrix(prob_matrix)
        if channel in temp_Q.index:
            temp_Q = temp_Q.drop(index=channel, columns=channel)
            
        temp_N = compute_fundamental_matrix(temp_Q)
        temp_R, _ = extract_r_matrix(prob_matrix, absorbing_states)
        
        if channel in temp_R.index:
            temp_R = temp_R.drop(index=channel)
            
        # Re-align R for the loop iteration too
        temp_R = temp_R.reindex(temp_N.index).fillna(0)
        
        if 'start' in temp_N.index:
            temp_B = temp_N @ temp_R
            new_conv = temp_B.loc['start', conv_col[0]]
        else:
            new_conv = 0
            
        if baseline_conv > 0:
            eff = 1 - (new_conv / baseline_conv)
        else:
            eff = 0
            
        removal_results.append({'channel': channel, 'removal_effect': eff})
        
    return pd.DataFrame(removal_results)

def preprocess_for_shapley(df_paths):
    shap_data = []
    for _, row in df_paths.iterrows():
        clean_path = row['path'].replace(' > conversion', '').replace(' > drop', '')
        channels = clean_path.split(' > ')
        shap_data.append({
            'channels': channels,
            'conversion': row['conversion']
        })
    return pd.DataFrame(shap_data)

def compute_shapley(shap_df):
    all_channels = set(x for l in shap_df['channels'] for x in l)
    contributions = {c: 0 for c in all_channels}
    total_score = 0
    
    for _, row in shap_df.iterrows():
        if row['conversion'] == 1:
            share = 1.0 / len(row['channels'])
            for ch in row['channels']:
                contributions[ch] += share
                
    total_score = sum(contributions.values())
    shapley_res = []
    for k, v in contributions.items():
        pct = v / total_score if total_score > 0 else 0
        shapley_res.append({'channel': k, 'shapley_val': pct})
        
    return pd.DataFrame(shapley_res)