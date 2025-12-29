import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def _add_sigma(prior_df: pd.DataFrame, min_rel_sigma: float = 0.10, max_rel_sigma: float = 1.0) -> pd.DataFrame:
    """
    Adds sigma to prior_df heuristically.
    Used as a fallback if Markov/Shapley divergence cannot be calculated.
    """
    df = prior_df.copy()

    # We need 'attr_weight' to calculate sigma. 
    # If it's missing, try to create it from any available 'share' columns.
    if "attr_weight" not in df.columns:
        share_cols = [c for c in df.columns if 'share' in c]
        if share_cols:
            df['attr_weight'] = df[share_cols].mean(axis=1)
        else:
            # If we can't find weights, we can't calculate sigma scale. Return as is.
            return df

    if "confidence" not in df.columns:
        df["confidence"] = 0.5

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.5).clip(0.0, 0.95)

    df["relative_sigma"] = max_rel_sigma - (df["confidence"] * (max_rel_sigma - min_rel_sigma))

    global_scale = float(df["attr_weight"].mean()) if len(df) else 0.01
    if global_scale <= 0:
        global_scale = 0.01

    df["sigma"] = global_scale * df["relative_sigma"]

    # Avoid ultra narrow / huge priors
    df["sigma"] = np.clip(df["sigma"], a_min=0.01 * global_scale, a_max=2.0 * global_scale)
    return df

def plot_prior_distributions(prior_df, show_plot=False):
    """
    Generates a Forest Plot comparing Attribution Models.
    
    RETURNS:
    - fig: The matplotlib figure object.
    - df: The dataframe with sigma/confidence columns.
    """
    
    # 1. Initialize Figure (Prevent NoneType error)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 2. Handle Empty Data
    if prior_df is None or prior_df.empty:
        ax.text(0.5, 0.5, "No Attribution Data Available", ha='center')
        return fig, prior_df

    df = prior_df.copy()
    
    # 3. Try Standard Calculation (Markov vs Shapley)
    if 'markov_share' in df.columns and 'shapley_share' in df.columns:
        # Mean Attribution
        df['attr_weight'] = (df['markov_share'] + df['shapley_share']) / 2
        
        # Sigma (Divergence between models)
        df['sigma'] = df[['markov_share', 'shapley_share']].std(axis=1)
        
        # If models agree perfectly, sigma is 0. Add tiny noise to avoid div/0 later
        df['sigma'] = df['sigma'].replace(0, 0.001)

    # 4. FALLBACK: If Sigma is still missing (e.g. columns missing), use your helper function
    if 'sigma' not in df.columns or df['sigma'].isna().all():
        print("⚠️ [Forest Plot] Standard Sigma calc failed. Using heuristic fallback (_add_sigma).")
        df = _add_sigma(df)

    # 5. Final Safety Check (Ensure columns exist for plotting)
    if 'attr_weight' not in df.columns or 'sigma' not in df.columns:
        # If fallback also failed, create dummy cols to prevent crash
        if 'attr_weight' not in df.columns and 'markov_share' in df.columns:
            df['attr_weight'] = df['markov_share']
        df['sigma'] = df.get('sigma', 0.01)
        df['attr_weight'] = df.get('attr_weight', 0.1)

    # 6. Calculate Confidence & Intervals for Plotting
    # (Recalculate confidence derived from sigma if not already set)
    if 'confidence' not in df.columns:
        df['confidence'] = 1 - (df['sigma'] / (df['attr_weight'] + 0.0001))
        df['confidence'] = df['confidence'].clip(0, 1)

    df['lower_bound'] = df['attr_weight'] - (1.96 * df['sigma'])
    df['upper_bound'] = df['attr_weight'] + (1.96 * df['sigma'])
    
    # Sort
    df = df.sort_values('attr_weight', ascending=True)

    # 7. Plotting
    y_pos = np.arange(len(df))
    
    # Forest Plot Error Bars
    ax.errorbar(
        x=df['attr_weight'], 
        y=y_pos, 
        xerr=1.96 * df['sigma'], 
        fmt='o', 
        color='#1E40AF',
        ecolor='#3B82F6',
        elinewidth=3,
        capsize=5,
        markersize=8,
        label='Mean Attribution (±95% CI)'
    )
    
    # Add Markers if columns exist
    if 'markov_share' in df.columns:
        ax.scatter(df['markov_share'], y_pos, color='#EF4444', marker='|', s=50, label='Markov', alpha=0.6)
    if 'shapley_share' in df.columns:
        ax.scatter(df['shapley_share'], y_pos, color='#10B981', marker='|', s=50, label='Shapley', alpha=0.6)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['channel'], fontsize=10, fontweight='bold')
    ax.set_xlabel('Attribution Share', fontsize=10)
    ax.set_title('Attribution Model Consensus (Forest Plot)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    
    if show_plot:
        plt.show()
        
    return fig, df