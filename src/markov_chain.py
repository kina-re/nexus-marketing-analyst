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
