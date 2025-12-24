import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


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





def run_mmm(
    df: pd.DataFrame,
    date_col: str = "Date",
    revenue_col: str = "Revenue",
    adstock_alpha: float = 0.5,
    hill_slope: float = 1.0,
    ridge_alpha: float = 1.0,
):
    """
    Run a revenue-based MMM with adstock + Hill saturation.

    
    """

    data = df.copy()

    if revenue_col not in data.columns:
        raise ValueError(f"Missing revenue column: {revenue_col}")

    exclude_cols = {date_col, revenue_col, "Conversions"}

    spend_cols = [c for c in data.columns if c not in exclude_cols]

    if not spend_cols:
        raise ValueError("No spend columns found")

    y = data[revenue_col].values.astype(float)

    # --- Transform pipeline ---
    transformed = {}

    for col in spend_cols:
        ad = geometric_adstock(data[col].values, alpha=adstock_alpha)
        sat = hill_saturation(ad, slope=hill_slope)
        transformed[col] = sat

    X = pd.DataFrame(transformed)

    # --- Scale for numerical stability ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Regression ---
    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(X_scaled, y)

    y_hat = model.predict(X_scaled)

    r2 = model.score(X_scaled, y)

    return {
        "coefficients": dict(zip(spend_cols, model.coef_)),
        "intercept": model.intercept_,
        "r2": r2,
        "design_matrix": X,
        "fitted_values": y_hat,
    }
