import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
from PIL import Image 
from datetime import datetime  # <--- NEW IMPORT

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

# --- REPORTING IMPORTS ---
import viz_report
import pdf_generator

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GA_DATA_PATH = PROJECT_ROOT / "data" / "ga_2.csv"
MMM_DATA_PATH = PROJECT_ROOT / "data" / "mmm_data_2016_2017.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    Robust Parser: Splits by [HEADER] tags, ignoring case and formatting errors.
    """
    sections = {
        'executive_summary': "Summary unavailable.",
        'q_matrix_insight': "Insight unavailable.",
        'r_matrix_insight': "Insight unavailable.",
        'removal_insight': "Insight unavailable."
    }
    
    # 1. Check for AI Failure
    if "⚠️" in text or "Error" in text[:50]:
        print(f"⚠️ LLM Generation Failed: {text}")
        return sections

    # 2. Normalize Text (Remove **bolding** around headers)
    clean_text = re.sub(r'\*\*\[(.*?)\]\*\*', r'[\1]', text) 
    
    # 3. Fuzzy Split
    tokens = re.split(r"\[([A-Z ]+)\]", clean_text, flags=re.IGNORECASE)

    # 4. Map Sections
    header_map = {
        'EXECUTIVE SUMMARY': 'executive_summary',
        'Q MATRIX': 'q_matrix_insight',
        'R MATRIX': 'r_matrix_insight',
        'REMOVAL EFFECTS': 'removal_insight'
    }

    for i in range(1, len(tokens)-1, 2):
        header = tokens[i].strip().upper()
        content = tokens[i+1].strip()
        
        if header in header_map:
            sections[header_map[header]] = content
            
    return sections

def run_analysis_pipeline():
    print("⏳ Running Attribution Models...")
    if not GA_DATA_PATH.exists() or not MMM_DATA_PATH.exists():
        raise FileNotFoundError("Data files missing in 'data/' folder.")

    ga_df = pd.read_csv(GA_DATA_PATH, low_memory=False)
    
    # 1. MARKOV
    df_paths = create_user_journeys(ga_df)
    _, prob_df = create_transition_matrix(ga_df, df_paths)
    Q, _ = extract_q_matrix(prob_df)
    R, _ = extract_r_matrix(prob_df, ['conversion', 'dropped'])
    N = compute_fundamental_matrix(Q)
    markov_removal = removal_effects(prob_df, ['conversion', 'dropped'], N, R)

    # 2. GENERATE PDF CHARTS
    print("🎨 Generating Report Charts...")
    img_paths = {
        'q_matrix': str(OUTPUT_DIR / "q_matrix.png"),
        'r_matrix': str(OUTPUT_DIR / "r_matrix.png"),
        'removal': str(OUTPUT_DIR / "removal_effects.png")
    }
    viz_report.plot_q_matrix(Q, img_paths['q_matrix'])
    viz_report.plot_r_matrix(R, img_paths['r_matrix'])
    viz_report.plot_removal_effects(markov_removal, img_paths['removal'])
    
    if 'conversion' in R.columns:
        img_paths['conv_probs'] = str(OUTPUT_DIR / "conv_probs.png")
        viz_report.plot_conversion_probs(R['conversion'], img_paths['conv_probs'])

    # 3. SHAPLEY & TOP PATHS
    shap_df = preprocess_for_shapley(df_paths)
    shapley_pct = compute_shapley(shap_df)
    top_paths = top_converting_paths(df_paths, top_n=10)
    
    # 4. FOREST PLOT & SIGMA
    prior_df = synthesize_attribution_prior(markov_removal, shapley_pct)
    res = plot_prior_distributions(prior_df, show_plot=False)
    prior_df_with_sigma = res[1] if isinstance(res, tuple) else res
    
    forest_path = OUTPUT_DIR / "attribution_forest_plot.png"
    Image.open(img_paths['removal']).save(forest_path) 

    # 5. MMM MERGE
    prior_df_with_sigma['match_key'] = prior_df_with_sigma['channel'].apply(lambda x: x.lower().strip().replace(' ', '_'))
    mmm_raw_df = pd.read_csv(MMM_DATA_PATH)
    mmm_df = normalize_mmm_channels(mmm_raw_df)
    if 'Date' in mmm_df.columns: mmm_df['Date'] = pd.to_datetime(mmm_df['Date'])
    mmm_df = mmm_df.T.groupby(level=0).sum().T
    if 'paid_search' in mmm_df.columns and 'display' in mmm_df.columns:
        mmm_df['performance_ads'] = mmm_df['paid_search'] + mmm_df['display']
        mmm_df = mmm_df.drop(columns=['paid_search', 'display'])
    roi_df = compute_channel_contribution_and_roi(df=mmm_df, date_col='Date', revenue_col='Revenue')
    roi_df['match_key'] = roi_df['channel'].apply(lambda x: x.lower().strip().replace(' ', '_'))
    
    total_rev = roi_df['incremental_revenue'].sum()
    roi_df['mmm_share'] = roi_df['incremental_revenue'] / total_rev if total_rev > 0 else 0

    comparison_df = pd.merge(
        roi_df[['match_key', 'channel', 'roi', 'mmm_share', 'incremental_revenue']],
        prior_df_with_sigma[['match_key', 'attr_weight', 'confidence', 'sigma']],
        on='match_key', how='left'
    ).fillna(0)

    # 6. LLM REPORT WRITING
    print("🧠 Nexus is writing the Executive Report...")
    
    full_prompt = f"""
    [ROLE]
    You are 'Nexus', a Senior Marketing Strategist. 

    [DATA 1: ROI & ATTRIBUTION]
    {comparison_df.to_string()}
    
    [DATA 2: TRANSITION MATRIX (Q)]
    {Q.to_string()}
    
    [DATA 3: ABSORPTION MATRIX (R)]
    {R.to_string()}
    
    [DATA 4: REMOVAL EFFECTS]
    {markov_removal.to_string()}
    
    [INSTRUCTIONS]
    Analyze the strategy. Identify "Optimization Opportunities" and "Critical Drivers".
    
    [IMPORTANT FORMATTING RULE]
    You MUST separate your sections using EXACTLY these tags (brackets included):
    [EXECUTIVE SUMMARY]
    [Q MATRIX]
    [R MATRIX]
    [REMOVAL EFFECTS]
    
    [EXECUTIVE SUMMARY]
    (3 bullet points comparing ROI vs Attribution)
    
    [Q MATRIX]
    (Analyze user journey loops)
    
    [R MATRIX]
    (Analyze conversion vs drop-off)
    
    [REMOVAL EFFECTS]
    (Identify critical channels)
    """

    raw_report = nexus.chat_with_data(full_prompt, "")
    
    print(f"\n--- DEBUG: RAW AI OUTPUT ---\n{raw_report}\n----------------------------\n")
    report_data = parse_nexus_report(raw_report)

    # 7. COMPILE PDF (With Timestamp Fix)
    print("📄 Compiling PDF...")
    
    # --- FIX STARTS HERE ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_name = f"Nexus_Marketing_Report_{timestamp}.pdf"
    # --- FIX ENDS HERE ---
    
    pdf_path = pdf_generator.generate_full_report(report_data, img_paths, str(OUTPUT_DIR / pdf_name))

    return {
        "pdf_path": pdf_path,
        "prior_df": comparison_df,
        "top_paths": top_paths,
        "img_paths": img_paths
    }

def main():
    try:
        res = run_analysis_pipeline()
        
        print("\n" + "="*50)
        print(f"✅ REPORT GENERATED: {res['pdf_path']}")
        print("="*50)

        print("\n🧠 NEXUS ONLINE. (I have analyzed the report data)")
        print("Try asking: 'Explain the Q-Matrix findings' or 'Why is referral critical?'")
        
        context_str = res['prior_df'].to_string()
        
        while True:
            user_input = input("\n👤 YOU: ").strip()
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            if not user_input: continue
            
            image_to_send = None
            if 'removal' in user_input.lower(): image_to_send = res['img_paths']['removal']
            elif 'matrix' in user_input.lower(): image_to_send = res['img_paths']['q_matrix']

            print("🤖 NEXUS: Thinking...", end="\r")
            response = nexus.chat_with_data(user_input, context_str, image_to_send)
            print(f"\r🤖 NEXUS: {response}")

    except Exception as e:
        print(f"Pipeline Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()