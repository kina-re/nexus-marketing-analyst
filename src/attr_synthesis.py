import pandas as pd
import numpy as np

def canonical_channel_name(x: str) -> str:
    x = str(x).strip().lower()
    for ch in ["&", "-", " ", "/", "\\", ".", ",", ":"]:
        x = x.replace(ch, "_")
    while "__" in x:
        x = x.replace("__", "_")
    return x.strip("_")


def synthesize_attribution_prior(
    markov_removal_pp: pd.Series,
    shapley_pct: pd.Series,
    mmm_channels: list | None = None,
    mix_weight: float = 0.60,       # Weight given to Markov
    conf_markov: float = 0.90,      # Trust in Markov method
    conf_shapley: float = 0.60,     # Trust in Shapley method
    conf_none: float = 0.10,        # Minimum confidence floor
    epsilon_weight: float = 1e-6,
    disagreement_damping: float = 0.05 # <--- NEW: 5% buffer 
):
    """
    Builds a confidence-weighted attribution prior with 'Smoothed Disagreement'.
    
    The 'disagreement_damping' (default 0.05) prevents 100% penalties 
    when one model is 0.0 and the other is small (e.g., 0.02).
    """

    # ---------- 1. Normalize inputs ----------
    m = markov_removal_pp.rename(index=canonical_channel_name).clip(lower=0)
    s = shapley_pct.rename(index=canonical_channel_name).clip(lower=0)

    # Normalize to sum to 1.0 (Safe division)
    m_w = m / m.sum() if m.sum() > 0 else m
    s_w = s / s.sum() if s.sum() > 0 else s

    # ---------- 2. Channel Universe ----------
    if mmm_channels:
        channels = [canonical_channel_name(c) for c in mmm_channels]
        channels = list(dict.fromkeys(channels)) # Dedupe preserving order
    else:
        channels = sorted(set(m_w.index).union(set(s_w.index)))

    rows = []

    # Calculate Base Confidence (Weighted Average of method trust)
    base_conf = (mix_weight * conf_markov) + ((1 - mix_weight) * conf_shapley)

    for ch in channels:
        mw = float(m_w.get(ch, 0.0))
        sw = float(s_w.get(ch, 0.0))

        # ---------- 3. Blended Weight ----------
        weight = (mix_weight * mw) + ((1 - mix_weight) * sw)

        # ---------- 4. Confidence with Damping ----------
        # Logic: abs(diff) / (max_val + damping)
        # If mw=0, sw=0.02, damping=0.05:
        # Penalty = 0.02 / (0.02 + 0.05) = 0.02 / 0.07 ≈ 0.28 (28% penalty)
        # Instead of 100% penalty, we keep ~70% of our confidence.
        
        diff = abs(mw - sw)
        max_val = max(mw, sw)
        
        # Calculate penalty
        penalty = diff / (max_val + disagreement_damping)
        
        # Apply penalty to base confidence
        confidence = base_conf * (1 - penalty)

        # Apply Floor
        final_conf = max(confidence, conf_none)
        final_conf = np.clip(final_conf, conf_none, 0.95)


        # Label Source
        if mw > 0 and sw > 0:
            source = "markov+shapley"
        elif mw > 0:
            source = "markov_only"
        elif sw > 0:
            source = "shapley_only"
        else:
            source = "none"
            weight = epsilon_weight 
            final_conf = conf_none

        rows.append({
            "channel": ch,
            "attr_weight": weight,
            "source": source,
            "confidence": final_conf,
            "markov_weight": mw,
            "shapley_weight": sw,
        })

    df = pd.DataFrame(rows)

    # ---------- 5. Final Normalization ----------
    total = df["attr_weight"].sum()
    if total > 0:
        df["attr_weight"] = df["attr_weight"] / total
    
    return df.sort_values("attr_weight", ascending=False).reset_index(drop=True)