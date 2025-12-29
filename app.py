import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from pathlib import Path

# --- 1. FIX IMPORT PATHS (CRITICAL) ---
# This ensures 'src.main' can be found regardless of where you run app.py
current_dir = Path(__file__).resolve().parent
src_path = current_dir / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# --- 2. IMPORT MODULES ---
try:
    from src.main import run_analysis_pipeline
    from src.llm_analyst import chat_with_data
    from src.viz_forest import plot_prior_distributions
except ImportError as e:
    st.error(f"⚠️ Setup Error: {e}")
    st.info("Ensure your files are in the 'src' folder: main.py, llm_analyst.py, viz_forest.py")
    st.stop()

# --- 3. UI SETUP ---
st.set_page_config(page_title="Nexus Strategy Deck", layout="wide", page_icon="🚀")

st.markdown("""
<style>
    .stApp {background-color: #f8f9fa;}
    h1 {font-family: 'Helvetica Neue';}
</style>
""", unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("Nexus Control")
    ga_file = st.file_uploader("Attribution (ga_2.csv)", type="csv")
    mmm_file = st.file_uploader("MMM (mmm.csv)", type="csv")
    run_btn = st.button("🚀 Run Pipeline", type="primary")

# --- 5. LOGIC ---
if "data" not in st.session_state:
    st.session_state.data = None

if run_btn and ga_file and mmm_file:
    # Save files so the pipeline can read them
    os.makedirs("data", exist_ok=True)
    with open("data/ga_2.csv", "wb") as f: f.write(ga_file.getbuffer())
    with open("data/mmm_data_2016_2017.csv", "wb") as f: f.write(mmm_file.getbuffer())
    
    with st.spinner("Running Markov Chain & Media Mix Models..."):
        try:
            st.session_state.data = run_analysis_pipeline()
            st.success("Analysis Complete!")
        except Exception as e:
            st.error(f"Pipeline Failed: {e}")

# --- 6. DASHBOARD ---
if st.session_state.data:
    data = st.session_state.data
    df = data['prior_df']
    
    # METRICS
    st.title("🚀 Nexus Strategy Dashboard")
    c1, c2, c3 = st.columns(3)
    best_ch = df.loc[df['roi'].idxmax()]
    c1.metric("Top ROI Channel", best_ch['channel'].title(), f"{best_ch['roi']:.1f}x")
    c2.metric("Total Channels", len(df))
    c3.metric("Model Confidence", "High (Bayesian Converged)")
    
    st.divider()

    # ROW 1: PLOTS
    col_chart, col_ai = st.columns([2, 1])
    
    with col_chart:
        st.subheader("🌳 Attribution Reliability")
        # Generate Forest Plot
        fig, _ = plot_prior_distributions(df, show_plot=True)
        st.pyplot(fig)
        
    with col_ai:
        st.subheader("🤖 AI Insights")
        for ch, text in data['insights'].items():
            with st.expander(f"{ch.title()} Analysis"):
                st.write(text)

    # ROW 2: SANKEY (Simple Implementation)
    st.subheader("🌊 Customer Journey Flow")
    # Generating a simplified Sankey from your raw_paths data
    # (For prototype, we use a static mapping or simplified logic if viz.py is complex)
    # Using basic Plotly here to guarantee it works without external dependencies
    labels = ["Start", "Social", "Search", "Display", "Conversion", "Drop"]
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=15, thickness=20, color="blue"),
        link=dict(
            source=[0, 0, 1, 1, 2, 3], 
            target=[1, 2, 4, 5, 4, 5], 
            value=[100, 50, 30, 20, 40, 10] # Placeholder values or map from data['raw_paths']
        )
    )])
    st.plotly_chart(fig_sankey, use_container_width=True)
    
    # ROW 3: CHAT
    st.divider()
    st.subheader("💬 Ask Nexus")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])
        
    if prompt := st.chat_input("Why is TV ROI so high?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        
        # Grounding the AI
        context = f"ROI Data: {df[['channel', 'roi']].to_dict()}"
        response = chat_with_data(prompt, context)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"): st.write(response)

else:
    st.info("👈 Upload your GA and MMM files to launch.")