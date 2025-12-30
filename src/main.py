import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
from PIL import Image 
from datetime import datetime

# --- FIX: FORCE NON-INTERACTIVE BACKEND ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- PIPELINE IMPORTS ---
from markov_shapley import (
    create_user_journeys, create_transition_matrix, extract_q_matrix,
    extract_r_matrix, compute_fundamental_matrix, removal_effects,
    preprocess_for_shapley, compute_shapley, top_converting_paths
)
from attr_synthesis import synthesize_attribution_prior
from viz_forest import plot_prior_distributions
from viz import generate_sankey_pro 
from mmm import compute_channel_contribution_and_roi
from reasoning import ReasoningEngine 
import llm_analyst as nexus 
import viz_report
import pdf_generator

def normalize_mmm_channels(df):
    mapping = {
        'Social_Media': 'social', 'Facebook_Ads': 'social',
        'Paid_Search': 'paid_search', 'Display_Ads': 'display',
        'Affiliates': 'affiliates', 'TV_Campaign': 'tv',
        'Billboard_Ads': 'billboard'
    }
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns=mapping)

def parse_nexus_report(text):
    """
    Robust Parser: Uses strict delimiter splitting to guarantee sections are found.
    """
    # Default fallback content
    sections = {
        'executive_summary': "Summary unavailable.",
        'q_matrix_insight': "Insight unavailable.",
        'r_matrix_insight': "Insight unavailable.",
        'removal_insight': "Insight unavailable."
    }
    
    if not text or "Error" in text[:50]: 
        return sections

    # 1. Define the exact delimiters we used in the prompt
    delimiters = {
        'executive_summary': '===EXECUTIVE_SUMMARY===',
        'q_matrix_insight': '===Q_MATRIX_INSIGHT===',
        'r_matrix_insight': '===R_MATRIX_INSIGHT===',
        'removal_insight': '===REMOVAL_INSIGHT==='
    }

    # 2. Iterate and extract text between delimiters
    # We sort keys by position in text to extract sequentially if needed, 
    # but simple string splitting is safest.
    
    for key, tag in delimiters.items():
        if tag in text:
            # Split the text by the tag, take the SECOND part (index 1)
            # Then split by '===' to stop at the next tag
            try:
                # Example: "text... ===TAG=== The Content Next ===NEXT_TAG=== ..."
                content = text.split(tag)[1].split('===')[0].strip()
                
                # Clean up any leftover Markdown artifacts
                content = content.replace('**', '').replace('##', '').strip()
                
                if len(content) > 10:
                    sections[key] = content
            except IndexError:
                continue

    return sections

# --- UPDATED: Added status_callback parameter ---
def run_analysis_pipeline(ga_path, mmm_path, output_dir, status_callback=None):
    
    # Helper to print AND update UI
    def log(message):
        print(message)
        if status_callback:
            status_callback(message)

    log("⏳ Running Attribution Models (Markov & Shapley)...")
    
    ga_path = Path(ga_path)
    mmm_path = Path(mmm_path)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ga_df = pd.read_csv(ga_path, low_memory=False)
    
    # 1. MARKOV
    df_paths = create_user_journeys(ga_df)
    _, prob_df = create_transition_matrix(ga_df, df_paths)
    Q, _ = extract_q_matrix(prob_df)
    R, _ = extract_r_matrix(prob_df, ['conversion', 'dropped'])
    N = compute_fundamental_matrix(Q)
    markov_removal = removal_effects(prob_df, ['conversion', 'dropped'], N, R)

    # 2. GENERATE CHARTS
    log("🎨 Generating Report Charts...")
    img_paths = {
        'q_matrix': str(output_dir / "q_matrix.png"),
        'r_matrix': str(output_dir / "r_matrix.png"),
        'removal': str(output_dir / "removal_effects.png")
    }
    viz_report.plot_q_matrix(Q, img_paths['q_matrix'])
    viz_report.plot_r_matrix(R, img_paths['r_matrix'])
    viz_report.plot_removal_effects(markov_removal, img_paths['removal'])
    
    if 'conversion' in R.columns:
        img_paths['conv_probs'] = str(output_dir / "conv_probs.png")
        viz_report.plot_conversion_probs(R['conversion'], img_paths['conv_probs'])

    # 3. SHAPLEY
    shap_df = preprocess_for_shapley(df_paths)
    shapley_pct = compute_shapley(shap_df)
    top_paths = top_converting_paths(df_paths, top_n=10)
    
    # 4. FOREST PLOT
    prior_df = synthesize_attribution_prior(markov_removal, shapley_pct)
    
    # UNPACK THE TUPLE (Fig, DF)
    fig_forest, prior_df_with_sigma = plot_prior_distributions(prior_df, show_plot=False)
    
    forest_path = output_dir / "attribution_forest_plot.png"
    
    # SAVE AND CLOSE
    if fig_forest is not None:
        fig_forest.savefig(str(forest_path), bbox_inches='tight')
        plt.close(fig_forest)
    else:
        print("⚠️ Warning: Forest Plot figure was None. Skipping save.")

    # 5. SANKEY
    log("🌊 generating Sankey Diagram...")
    sankey_fig = generate_sankey_pro(df_paths)
    sankey_path = output_dir / "customer_journey_sankey.html"
    sankey_fig.write_html(str(sankey_path))

    # 6. MMM MERGE
    prior_df_with_sigma['match_key'] = prior_df_with_sigma['channel'].apply(lambda x: x.lower().strip().replace(' ', '_'))
    mmm_raw_df = pd.read_csv(mmm_path)
    mmm_df = normalize_mmm_channels(mmm_raw_df)
    if 'Date' in mmm_df.columns: mmm_df['Date'] = pd.to_datetime(mmm_df['Date'])
    mmm_df = mmm_df.T.groupby(level=0).sum().T
    
    roi_df = compute_channel_contribution_and_roi(df=mmm_df, date_col='Date', revenue_col='Revenue')
    roi_df['match_key'] = roi_df['channel'].apply(lambda x: x.lower().strip().replace(' ', '_'))
    total_rev = roi_df['incremental_revenue'].sum()
    roi_df['mmm_share'] = roi_df['incremental_revenue'] / total_rev if total_rev > 0 else 0

    comparison_df = pd.merge(
        roi_df[['match_key', 'channel', 'roi', 'mmm_share', 'incremental_revenue']],
        prior_df_with_sigma[['match_key', 'attr_weight', 'confidence', 'sigma']],
        on='match_key', how='left'
    ).fillna(0)

    # ---------------------------------------------------------
    # 6.5 PREPARE REMOVAL DATA (The Missing Link)
    # ---------------------------------------------------------
    # 'markov_removal' was calculated in Step 1. 
    # We convert it to a DataFrame so Step 7 can read it.
    try:
        removal_df = pd.DataFrame(list(markov_removal.items()), columns=['channel', 'removal_effect'])
    except Exception as e:
        log(f"⚠️ Error formatting removal effects: {e}")
        removal_df = pd.DataFrame(columns=['channel', 'removal_effect'])

    # ... inside run_analysis_pipeline ...

    # 7. LLM REPORT
    log("🧠 Nexus is analyzing strategy & writing report...")
    
    # --- PREPARE DATA FOR AI (Text Serialization) ---
    
    # A. Format R-Matrix (Conversion/Drops)
    r_lines = []
    if 'conversion' in R.columns:
        R_sorted = R.sort_values('conversion', ascending=False)
        for channel, row in R_sorted.iterrows():
            conv = row.get('conversion', 0)
            drop = row.get('dropped', 0)
            r_lines.append(f"- {channel}: {conv:.1%} chance to Convert, {drop:.1%} chance to Drop off.")
    r_text = "\n".join(r_lines)

    # B. Format Q-Matrix (Transitions)
    q_str = Q.round(3).to_string()

    # C. Format Removal Effects (Strategic Value)
    removal_lines = []
    if 'removal_effect' in removal_df.columns:
        # Sort by impact to highlight most important channels
        rem_sorted = removal_df.sort_values('removal_effect', ascending=False)
        for _, row in rem_sorted.iterrows():
            # removal_effect is usually 0.0 to 1.0
            val = row['removal_effect']
            removal_lines.append(f"- {row['channel']}: {val:.1%} (This % of total conversions would be LOST if channel is removed)")
    removal_text = "\n".join(removal_lines)
    
    # --- CONSTRUCT PROMPT ---
    full_prompt = f"""
    [ROLE] Senior Marketing Strategist. 
    
    [DATA 1: ROI & ATTRIBUTION]
    {comparison_df.to_string()}
    
    [DATA 2: CONVERSION PROBABILITIES (R-Matrix)]
    {r_text}
    
    [DATA 3: JOURNEY TRANSITIONS (Q-Matrix)]
    {q_str}

    [DATA 4: REMOVAL EFFECTS (Strategic Value)]
    (This measures the "Essentialness" of a channel. High % = Critical Channel.)
    {removal_text}
    
    [TASK] Write a strategic marketing report.
    
    [IMPORTANT] You MUST use the exact delimiters below. Do not change them.
    
    ===EXECUTIVE_SUMMARY===
    (Write a high-level summary of ROI, best channels, and overall performance.)
    
    ===Q_MATRIX_INSIGHT===
    (Analyze the Q-Matrix. Which channels act as 'feeders' sending traffic to others?)
    
    ===R_MATRIX_INSIGHT===
    (Analyze the R-Matrix. Which channels have the highest direct Conversion probability? Which have high Drop-off rates?)
    
    ===REMOVAL_INSIGHT===
    (Analyze [DATA 4]. Which channels have the highest Removal Effect? These are your 'Load Bearing' channels. Compare this to their ROI.)
    """
    
    # Call AI
    raw_report = nexus.chat_with_data(full_prompt, "")
    report_data = parse_nexus_report(raw_report)

   
    # 8. COMPILE PDF
    log("📄 Compiling PDF...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_name = f"Nexus_Marketing_Report_{timestamp}.pdf"
    pdf_path = pdf_generator.generate_full_report(report_data, img_paths, str(output_dir / pdf_name))

    log("✅ Analysis Pipeline Complete.")

    return {
        "pdf_path": pdf_path,
        "prior_df": comparison_df,
        "top_paths": top_paths,
        "img_paths": img_paths,
        "sankey_path": sankey_path,
        "forest_path": forest_path,
        "report_data": report_data
    }
