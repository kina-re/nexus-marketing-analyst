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


def add_seasonality_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Add smooth annual seasonality using Fourier terms.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    day_of_year = out[date_col].dt.dayofyear.values

    out["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    return out



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

    data = add_seasonality_features(data, date_col)


    if revenue_col not in data.columns:
        raise ValueError(f"Missing revenue column: {revenue_col}")
    
    control_cols = ["season_sin", "season_cos"]

    exclude_cols = {date_col, revenue_col, "Conversions", *control_cols}

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

    X_spend = pd.DataFrame(transformed)
    X_controls = data[control_cols].reset_index(drop=True)

    X = pd.concat([X_spend, X_controls], axis=1)

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
    "design_matrix_scaled": X_scaled, 
    "scaler": scaler,                
    "fitted_values": y_hat,
}


def compute_channel_contribution_and_roi(
    df: pd.DataFrame,
    date_col: str = "Date",
    revenue_col: str = "Revenue",
    adstock_alpha: float = 0.5,
    hill_slope: float = 1.0,
    ridge_alpha: float = 1.0,
):
    """
    Compute counterfactual channel contribution and ROI using MMM.

    """

    # fir mmm once, coefficeint shpould not be reoptimized.
    mmm = run_mmm(
        df,
        date_col=date_col,
        revenue_col=revenue_col,
        adstock_alpha=adstock_alpha,
        hill_slope=hill_slope,
        ridge_alpha=ridge_alpha,
    )

    X_unscaled = mmm["design_matrix"]
    scaler = mmm["scaler"]
    
    # Scaled baseline
    X_scaled = pd.DataFrame(
    scaler.transform(X_unscaled),
    columns=X_unscaled.columns
    )

    coefs = mmm["coefficients"]
    intercept = mmm["intercept"]
    fitted = mmm["fitted_values"]

    baseline_revenue = fitted.sum()

    # Identify spend columns (same exclusion rule)
    exclude_cols = {date_col, revenue_col, "Conversions", "season_sin", "season_cos"}
    spend_cols = [c for c in df.columns if c not in exclude_cols]

    results = []

    for channel in spend_cols:
        # Counterfactual in UN-SCALED space
        X_cf_unscaled = X_unscaled.copy()
        X_cf_unscaled[channel] = 0.0

        # Re-scale using SAME scaler
        X_cf_scaled = pd.DataFrame(
        scaler.transform(X_cf_unscaled),
        columns=X_unscaled.columns
        )

        # Predict
        y_cf = intercept
        for col, coef in coefs.items():
            y_cf += coef * X_cf_scaled[col].values

        cf_revenue = y_cf.sum()

        incremental_revenue = baseline_revenue - cf_revenue
        total_spend = df[channel].sum()

        roi = (
            incremental_revenue / total_spend
            if total_spend > 0
            else float("nan")
        )

        results.append({
            "channel": channel,
            "incremental_revenue": incremental_revenue,
            "spend": total_spend,
            "roi": roi,
        })

    return (
        pd.DataFrame(results)
        .sort_values("incremental_revenue", ascending=False)
        .reset_index(drop=True)
    )
       