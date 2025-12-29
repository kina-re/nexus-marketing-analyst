import numpy as np
import pandas as pd
from collections import Counter

def create_user_journeys(df):
    """
    Builds journey for each user

    Parameters:
    - df: Input DataFrame with columns 'channel', 'converted', 'session_time', 'user_id'

    Returns:
    - df_paths: Output DataFrame with user_id, channel_list, result, and user_journey
    """


    df['channel'] = df['channel'].str.lower()


    df['Timestamp'] = pd.to_datetime(df['session_time'])


    df_sorted = df.sort_values(by=['user_id', 'session_time'])


    df_paths = df_sorted.groupby("user_id").agg(list)


    df_paths.rename(columns={"channel": "channel_list"}, inplace=True)


    df_paths = df_paths.reset_index()


    df_paths["result"] = df_paths["converted"].apply(lambda lst: "conversion" if any(lst) else "dropped")

    # Build user journey add the start state as it will be needed for sankey diagram
    df_paths["user_journey"] = df_paths.apply(
        lambda p: ["start"] + [channel for channel in p["channel_list"]] + [p["result"]],
        axis=1
    )

    return df_paths




def create_transition_matrix(df_original, df_paths, journey_col='user_journey', channel_col='channel'):
    """
    Creates the transition matrix showing both counts and probabilities.
    
    Parameters:
    - df_original : DataFrame
        The original dataset containing the channel column.
    - df_paths : DataFrame
        The dataset with user journeys created from create_user_journeys().
    - journey_col : str
        Column name in df_paths containing user journey lists.
    - channel_col : str
        Column name in df_original containing channel names.
    
    Returns:
    - count_df : pd.DataFrame
        Transition count matrix between states.
    - prob_df : pd.DataFrame
        Transition probability matrix between states.
    """

    # Get unique channel names from the original dataset
    unique_channels = df_original[channel_col].str.lower().unique().tolist()

    # Full ordered state list: start + channels + terminal states
    states = ['start'] + sorted(unique_channels) + ['conversion', 'dropped']

    # Map each state to index
    state_idx = {state: idx for idx, state in enumerate(states)}
    num_states = len(states)

    # Initialize transition count matrix
    count_matrix = np.zeros((num_states, num_states), dtype=int)

    # Populate transition counts
    for journey in df_paths[journey_col]:
        for i in range(len(journey) - 1):
            from_state = journey[i]
            to_state = journey[i + 1]
            if from_state in state_idx and to_state in state_idx:
                count_matrix[state_idx[from_state], state_idx[to_state]] += 1

    # Convert to DataFrame for readability
    count_df = pd.DataFrame(count_matrix, index=states, columns=states)

    # Create probability matrix
    row_sum = count_matrix.sum(axis=1, keepdims=True)
    prob_matrix = np.divide(count_matrix, row_sum, 
                            out=np.zeros_like(count_matrix, dtype=float),
                            where=row_sum != 0)

    prob_df = pd.DataFrame(prob_matrix.round(4), index=states, columns=states)

    return count_df, prob_df



# Define absorbing states globally (common convention)
absorbing_states = ['conversion', 'dropped']


def extract_q_matrix(prob_df, absorbing_states=absorbing_states):
    """
    Extracts the Q matrix: transient → transient
    """
    transient_states = [s for s in prob_df.index if s not in absorbing_states]
    Q = prob_df.loc[transient_states, transient_states]
    return Q, transient_states


def extract_r_matrix(prob_df, absorbing_states):
    # Same transient states logic
    transient_states = [state for state in prob_df.index if state not in absorbing_states]
    r_df = prob_df.loc[transient_states, absorbing_states]
    return r_df, transient_states


def compute_fundamental_matrix(q_matrix):
    """
    Computes the Fundamental Matrix N = (I - Q)^(-1)
    for transient-state Markov chains.

    Parameters:
        q_matrix (pd.DataFrame): Q matrix with transient-to-transient probabilities

    Returns:
        pd.DataFrame: Fundamental matrix with same index and columns as Q
    """

    transient_states = q_matrix.index.tolist()
    Q = q_matrix.values

    I = np.eye(len(Q))
    I_minus_Q = I - Q

    # Safety check: ensure invertible matrix
    if np.linalg.det(I_minus_Q) == 0:
        raise ValueError("Matrix (I - Q) is singular (non-invertible). Check Q structure.")

    N = np.linalg.inv(I_minus_Q)

    N_df = pd.DataFrame(N, index=transient_states, columns=transient_states)
    return N_df



# Rank Channel 

def rank_channels_by_expected_visits(q_matrix, compute_fundamental_matrix_func):
    """
    Computes the fundamental matrix and ranks channels by expected visits before absorption.

    Parameters:
    - q_matrix (pd.DataFrame): Q matrix representing transient-to-transient transition probabilities
    - compute_fundamental_matrix_func (function): Function to compute the fundamental matrix

    Returns:
    - ranked_channels (pd.Series): Channels ranked by expected visits
    - n_df (pd.DataFrame): Fundamental matrix
    """
    # Compute fundamental matrix
    n_df = compute_fundamental_matrix_func(q_matrix)

    # Rank channels by expected visits (column-wise sum)
    ranked_channels = n_df.sum(axis=0).sort_values(ascending=False)

    print("Ranked Channels by Expected Visits Before Absorption:")
    print(ranked_channels)

    return ranked_channels, n_df



# Absorption probability function

def compute_absorption_probabilities(n_df, r_matrix):
    """
    Computes the absorption probability matrix B = N × R.

    Parameters:
    - n_df (pd.DataFrame): Fundamental matrix (N)
    - r_matrix (pd.DataFrame): R matrix representing transient → absorbing transitions

    Returns:
    - b (pd.DataFrame): Absorption probability matrix
    """
    b = n_df @ r_matrix
    b = b.round(4)

    print("Absorption probability matrix (B = N × R):")
    print(b)

    return b


def build_transition_counts(df_paths, journey_col="user_journey"):
    transitions = []
    for path in df_paths[journey_col]:
        transitions += list(zip(path[:-1], path[1:]))
    
    trans_counts = Counter(transitions)

    df_trans = pd.DataFrame([
        {"source": src, "target": tgt, "count": c}
        for (src, tgt), c in trans_counts.items()
    ])

    df_trans = df_trans.sort_values("count", ascending=False).reset_index(drop=True)
    return df_trans


def compute_absorption_probabilities(n_df, r_matrix):
    """
    Computes the absorption probability matrix B = N × R.

    Parameters:
    - n_df (pd.DataFrame): Fundamental matrix (N)
    - r_matrix (pd.DataFrame): R matrix representing transient → absorbing transitions

    Returns:
    - b (pd.DataFrame): Absorption probability matrix
    """
    b = n_df @ r_matrix
    b = b.round(4)

    print("Absorption probability matrix (B = N × R):")
    print(b)

    return b


def get_conversion_probabilities(b_matrix, absorbing_state='conversion'):
    """
    Extracts and ranks channels by conversion probability (% format).
    """
    conversion_probs = (
        b_matrix[absorbing_state]
        .sort_values(ascending=False)
        .apply(lambda x: f"{x * 100:.2f}%")  # format as percent string
    )

    print(f"Channel Conversion Probabilities (→ {absorbing_state}):")
    print(conversion_probs)

    return conversion_probs



# Removal Effect


def removal_effects(prob_df, absorbing_states, baseline_N, R_matrix):
    """
    Calculate removal effects of each transient channel on conversion probability.
    """

    # Transient states only (exclude start + absorbing)
    transient_states = [s for s in prob_df.index if s not in absorbing_states and s != 'start']

    # Baseline conversion probability (from start)
    baseline_conversion = (baseline_N @ R_matrix).loc['start', 'conversion']

    results = {}

    for channel in transient_states:

        # Remaining transient states after removal
        trans_reduced = [s for s in transient_states if s != channel]

        # Reduced Q = transient -> transient only
        Q_reduced = prob_df.loc[trans_reduced, trans_reduced].astype(float).to_numpy()

        # Reduced R = transient -> absorbing only
        R_reduced = prob_df.loc[trans_reduced, absorbing_states].astype(float).to_numpy()

        # New Fundamental Matrix
        N_reduced = np.linalg.inv(np.eye(len(trans_reduced)) - Q_reduced)

        # Absorption probability from each transient state
        B_reduced = pd.DataFrame(N_reduced @ R_reduced,
                                 index=trans_reduced,
                                 columns=absorbing_states)

        # Get new conversion probability from start path:
        # start -> first-touch must be included:
        start_to_trans = prob_df.loc['start', trans_reduced].astype(float).to_numpy()
        reduced_conversion = (start_to_trans @ B_reduced['conversion'].to_numpy())

        # Removal effect in percentage points
        drop_pp = 100 * (baseline_conversion - reduced_conversion)
        results[channel] = round(drop_pp, 2)

    return pd.Series(results).sort_values(ascending=False)


# =========================
# SHAPLEY ATTRIBUTION
# =========================


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



def top_converting_paths(
    df_paths,
    journey_col="user_journey",
    result_col="result",
    top_n=10,
    min_conversions=1,
):
    """
    Returns the top converting channel paths with collective contribution.
    Robust to all pandas versions.
    """

    converted = df_paths[df_paths[result_col] == "conversion"]

    cleaned_paths = converted[journey_col].apply(
        lambda p: tuple(
            ch for ch in p
            if ch not in ("start", "conversion", "dropped")
        )
    )

    path_counts = (
        cleaned_paths
        .groupby(cleaned_paths)
        .size()
        .reset_index(name="conversions")
        .rename(columns={journey_col: "path"})
    )

    path_counts = path_counts[path_counts["conversions"] >= min_conversions]

    if path_counts.empty:
        return pd.DataFrame(columns=["path", "conversions", "share_of_conversions_pct"])

    total_conversions = path_counts["conversions"].sum()

    path_counts["share_of_conversions_pct"] = (
        100 * path_counts["conversions"] / total_conversions
    ).round(2)

    path_counts["path"] = path_counts["path"].apply(lambda p: " → ".join(p))

    return (
        path_counts
        .sort_values("conversions", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )




def build_priors_from_attribution(
    prior_df: pd.DataFrame,
    revenue_anchor: float,
    min_sigma: float = 0.1,
    max_sigma: float = 2.5,
):
    """
    Converts attribution priors into Bayesian priors (mu, sigma)

    Parameters
    ----------
    prior_df : DataFrame
        Output of synthesize_attribution_prior
    revenue_anchor : float
        Estimated total revenue driven by marketing (e.g. 30% of total revenue)
    min_sigma : float
        Lower bound for confidence (strong prior)
    max_sigma : float
        Upper bound for uncertainty (weak prior)

    Returns
    -------
    DataFrame with columns: channel, mu, sigma
    """

    df = prior_df.copy()

    # --- Mean (μ) ---
    # Allocate total revenue proportionally
    df["mu"] = df["attr_weight"] * revenue_anchor

    # --- Sigma (σ) ---
    # High confidence → small sigma
    # Low confidence → large sigma
    df["sigma"] = max_sigma - df["confidence"] * (max_sigma - min_sigma)

    return df[["channel", "mu", "sigma"]]
