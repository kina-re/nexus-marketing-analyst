import numpy as np
import pandas as pd


def canonical_channel_name(x: str) -> str:
    """Make channel names consistent across GA (Markov/Shapley) and MMM columns."""
    x = str(x).strip().lower()
    x = x.replace("&", "and")
    for ch in ["-", " ", "/", "\\", ".", ",", ":", ";", "|"]:
        x = x.replace(ch, "_")
    while "__" in x:
        x = x.replace("__", "_")
    return x.strip("_")


def _norm_positive(s: pd.Series) -> pd.Series:
    s = s.copy().astype(float)
    s[s < 0] = 0.0
    total = s.sum()
    if total <= 0:
        return pd.Series(1.0 / len(s), index=s.index)
    return s / total


def build_priors_from_markov_shapley(
    markov_removal: pd.Series,
    shapley_pct: pd.Series,
    spend_by_channel: pd.Series,
    total_incremental_revenue_anchor: float,
    mix_weight: float = 0.5,
    min_sigma_roi: float = 0.5,
    max_sigma_roi: float = 6.0,
) -> pd.DataFrame:
    """
    Build Normal priors for ROI:
      ROI_c ~ Normal(mu_roi_c, sigma_roi_c)

    markov_removal: removal effects (higher => more necessary)
    shapley_pct: attribution shares (percent or any positive weights)
    spend_by_channel: MMM spend totals
    anchor: dollar amount to convert prior share -> dollars -> ROI
    """

    # Canonicalize indices
    m = markov_removal.copy()
    m.index = [canonical_channel_name(i) for i in m.index]

    s = shapley_pct.copy()
    s.index = [canonical_channel_name(i) for i in s.index]

    spend = spend_by_channel.copy()
    spend.index = [canonical_channel_name(i) for i in spend.index]

    # Normalize to weights
    m_w = _norm_positive(m)
    s_w = s.astype(float)
    s_w = s_w / s_w.sum() if s_w.sum() != 0 else pd.Series(1.0 / len(s_w), index=s_w.index)

    # Keep only overlapping channels
    channels = sorted(set(m_w.index) & set(s_w.index) & set(spend.index))
    if not channels:
        raise ValueError("No overlapping channels between Markov, Shapley, and MMM spend. Check naming.")

    m_w = m_w.reindex(channels).fillna(0.0)
    s_w = s_w.reindex(channels).fillna(0.0)
    spend = spend.reindex(channels).fillna(0.0)

    prior_share = mix_weight * m_w + (1 - mix_weight) * s_w

    prior_incremental_rev = prior_share * float(total_incremental_revenue_anchor)
    mu_roi = prior_incremental_rev / spend.replace(0.0, np.nan)

    # Sigma based on disagreement between Markov and Shapley (bigger disagreement => wider prior)
    disagreement = (m_w - s_w).abs()
    if disagreement.max() > 0:
        d = disagreement / disagreement.max()
    else:
        d = disagreement * 0.0

    sigma_roi = (min_sigma_roi + d * (max_sigma_roi - min_sigma_roi)).clip(lower=1e-6)

    out = pd.DataFrame(
        {
            "markov_w": m_w,
            "shapley_w": s_w,
            "prior_share": prior_share,
            "mu_roi": mu_roi,
            "sigma_roi": sigma_roi,
            "spend": spend,
        },
        index=channels,
    )
    return out


def bayesian_blend_roi(
    prior_df: pd.DataFrame,
    mmm_roi_df: pd.DataFrame,
    evidence_sigma_roi: float = 2.5,
) -> pd.DataFrame:
    """
    Normal-Normal conjugate blend (MAP).
    evidence_sigma_roi controls trust in MMM:
      smaller => trust MMM more
      larger  => trust priors more
    """

    mmm = mmm_roi_df.copy()
    mmm["channel_key"] = mmm["channel"].map(canonical_channel_name)
    mmm = mmm.set_index("channel_key")[["roi", "incremental_revenue", "spend"]]
    mmm = mmm.rename(columns={"roi": "mmm_roi"})

    merged = prior_df.join(mmm, how="inner")

    mu_p = merged["mu_roi"].astype(float)
    sig_p = merged["sigma_roi"].astype(float).clip(lower=1e-6)

    mu_e = merged["mmm_roi"].astype(float)
    sig_e = float(evidence_sigma_roi)

    prec_p = 1.0 / (sig_p ** 2)
    prec_e = 1.0 / (sig_e ** 2)

    merged["posterior_roi"] = (mu_p * prec_p + mu_e * prec_e) / (prec_p + prec_e)
    merged["posterior_sigma_roi"] = np.sqrt(1.0 / (prec_p + prec_e))

    return merged.reset_index(names=["channel_key"])
