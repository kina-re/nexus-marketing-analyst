from itertools import combinations
import pandas as pd
import numpy as np
import math


def preprocess_for_shapley(df_paths):
    """
    Prepare user-channel sets and conversion labels for Shapley.
    Removes 'start', 'conversion', 'dropped' states.
    """
    absorbing_states = ['conversion', 'dropped', 'null', 'unknown']  # flexible safety
    unwanted_states = ['start'] + absorbing_states
    
    df_shap = df_paths.copy()
    
    # Extract actual channels only
    df_shap['channels'] = df_shap['user_journey'].apply(
        lambda journey: list({ch for ch in journey if ch not in unwanted_states})
    )
    
    # Ensure conversion boolean exists
    df_shap['converted_flag'] = df_shap['converted'].apply(lambda x: 1 if x else 0)
    
    # Filter out empty-channel users
    df_shap = df_shap[df_shap['channels'].map(len) > 0]
    
    return df_shap[['user_id', 'channels', 'converted_flag']]



def compute_shapley(df_shap):
    """
    Compute Shapley attribution values for marketing channels.
    Uses coalition logic and weighted marginal contribution.
    """
    all_channels = sorted({ch for channels in df_shap['channels'] for ch in channels})
    shapley_values = {ch: 0.0 for ch in all_channels}
    
    total_conversions = df_shap['converted_flag'].sum()
    
    if total_conversions == 0:
        raise ValueError("No conversions in data — Shapley undefined")

    # Precompute contributions of all coalitions
    coalition_values = {}

    def coalition_key(channels):
        return tuple(sorted(channels))

    def value_of_subset(ch_subset):
        key = coalition_key(ch_subset)
        if key in coalition_values:
            return coalition_values[key]

        # Users whose channels are subset of coalition
        mask = df_shap['channels'].apply(lambda ch: set(ch).issubset(ch_subset))
        value = df_shap.loc[mask, 'converted_flag'].sum()
        coalition_values[key] = value
        return value

    # Shapley value calculation
    for ch in all_channels:
        other_channels = [o for o in all_channels if o != ch]
        
        for k in range(len(other_channels) + 1):
            for subset in combinations(other_channels, k):
                subset = list(subset)
                v_without = value_of_subset(subset)
                v_with = value_of_subset(subset + [ch])
                
                marginal_contribution = v_with - v_without
                
                weight = (math.factorial(len(subset)) *
                          math.factorial(len(all_channels) - len(subset) - 1) /
                          math.factorial(len(all_channels)))
                
                shapley_values[ch] += weight * marginal_contribution

    # Normalize into percentage attribution shares
    shapley_series = pd.Series(shapley_values)
    shapley_percent = 100 * shapley_series / shapley_series.sum()

    return shapley_percent.round(2).sort_values(ascending=False)