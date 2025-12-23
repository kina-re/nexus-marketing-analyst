import pandas as pd
import numpy as np


def geometric_adstock(x: np.ndarray, alpha: float = 0.5):
    """
    Apply geometric adstock to a spend series.

    x[t] = spend at time t
    out[t] = x[t] + alpha * out[t-1]
    """

    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)

    out[0] =x [0]

    for t in range(1, len(x)):
        out[t] = x[t] + alpha * out[t - 1]
    
    return out


    return numerator / (denominator + 1e-9)

def hill_saturation(x: np.ndarray, half_saturation: float = None, slope: float = 1.0) :
    """
    Apply Hill (diminishing returns) saturation to a positive signal.

   
    """

    x = np.asarray(x, dtype=float)

    # Guard against negatives
    x = np.clip(x, 0.0, None)

    # Choose half-saturation automatically if not provided
    if half_saturation is None:
        positive = x[x > 0]
        half_saturation = np.median(positive) if positive.size > 0 else 1.0

    # Hill function
    numerator = np.power(x, slope)
    denominator = numerator + np.power(half_saturation, slope)

    return numerator / (denominator + 1e-9)



def run_mmm(df: pd.DataFrame, date_col: str = "Date", revenue_col: str= "revenue"):

    
    data = df.copy()

    # check if revenue_col exists
    if revenue_col not in data.columns:
        raise ValueError(f"Missing revenue column: {revenue_col}")
    
    # Get spend columns
    spend_cols = [c for c in data.columns if c not in {date_col, revenue_col}]

    if not spend_cols:
        raise ValueError("No spend columns found for MMM")
    
    # Output placeholder
    results_df = pd.DataFrame({
        "channel": spend_cols,
        "spend": [data[c].sum() for c in spend_cols]
    })

    return results_df