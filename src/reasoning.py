
import pandas as pd
import numpy as np

class ReasoningEngine:
    """
    Translates raw mathematical results into marketing narratives.
    Used to ground the LLM so it doesn't hallucinate.
    """
    
    def __init__(self, mmm_df, attribution_prior_df):
        self.mmm_df = mmm_df
        self.prior_df = attribution_prior_df
        
    def analyze_channel(self, channel_name):
        """
        Generates a structured dictionary of facts for a specific channel.
        """
        # normalize name for matching
        c_clean = channel_name.lower().strip().replace(' ', '_')
        
        # 1. GET MMM FACTS
        # Find row in MMM results
        mmm_row = self.mmm_df[self.mmm_df['match_key'] == c_clean]
        if not mmm_row.empty:
            roi = mmm_row.iloc[0]['roi']
            share = mmm_row.iloc[0]['incremental_revenue'] # This might need normalization in main
        else:
            roi = 0
            share = 0

        # 2. GET ATTRIBUTION FACTS
        # Find row in Prior results
        attr_row = self.prior_df[self.prior_df['match_key'] == c_clean]
        if not attr_row.empty:
            attr_share = attr_row.iloc[0]['attr_weight']
            sigma = attr_row.iloc[0]['sigma']
        else:
            attr_share = 0
            sigma = 0

        # 3. DERIVE NARRATIVE ARCHETYPES
        # Logic: Compare ROI vs Volume to determine role
        role = "Unknown"
        if roi > 15:
            role = "Efficiency Driver (High ROI)"
        elif roi < 5:
            role = "Volume Driver (Low ROI)"
        else:
            role = "Balanced Performer"

        # Logic: Divergence check
        if attr_share == 0:
            divergence = "Offline/Upper Funnel (Invisible to Tracker)"
        elif abs(share - attr_share) > (2 * sigma):
            divergence = "Major Discrepancy (Model Disagrees with Tracker)"
        else:
            divergence = "Validated (Model Agrees with Tracker)"

        return {
            "role": role,
            "divergence_type": divergence,
            "stat_summary": f"MMM ROI is {roi:.2f}. MMM Share is {share:.1%}. Tracker Share is {attr_share:.1%}."
        }