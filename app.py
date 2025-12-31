import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

# Add 'src' to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import src.main as pipeline_main
import src.llm_analyst as nexus

# --- 1. PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(
    page_title="Nexus | Strategic Marketing AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- 2. INITIALIZE STATE ---
if "results" not in st.session_state:
    st.session_state.results = None

if "messages" not in st.session_state:
    st.session_state.messages = []

load_dotenv()

# --- 3. CUSTOM CSS (FIXED FOR HIGHLIGHTS) ---
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #0F172A; font-weight: 700; margin-bottom: 0px; }
    .sub-text { font-size: 1rem; color: #64748B; margin-bottom: 20px; }
    
    /* Container styling */
    [data-testid="stDataFrame"] { background-color: transparent !important; border: none !important; padding: 0px !important; width: 100% !important; }
    [data-testid="stDataFrame"] > div { background-color: #FFFFFF !important; border-radius: 8px !important; width: 100% !important; }
    
    /* Header Styling - Keep White */
    th { background-color: #FFFFFF !important; color: #475569 !important; font-weight: 600 !important; border-bottom: 2px solid #E2E8F0 !important; }
    
    /* Cell Styling - MODIFIED: Removed '!important' background to allow highlights */
    td { color: #1E293B !important; font-size: 0.95rem !important; border-bottom: 1px solid #F1F5F9 !important; }
    
    .report-box { background-color: #F8FAFC; padding: 20px; border-radius: 8px; border-left: 4px solid #3B82F6; margin-bottom: 20px; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8637/8637103.png", width=70)
    st.markdown("### Nexus Control")
    if not os.getenv("GEMINI_API_KEY"):
        st.error("❌ API Key Missing (.env)")
        btn_disabled = True
    else:
        btn_disabled = False
    
    st.divider()
    ga_file = st.file_uploader("Google Analytics (CSV)", type=["csv"])
    mmm_file = st.file_uploader("MMM Data (CSV)", type=["csv"])
    process_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=btn_disabled)
    show_chat = st.toggle("💬 Show Assistant", value=True)

# --- HEADER ---
st.markdown('<div class="main-header">Nexus Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI-Powered Marketing Attribution & Strategy</div>', unsafe_allow_html=True)

# --- MAIN PROCESS LOGIC ---
if process_btn:
    if not ga_file or not mmm_file:
        st.warning("⚠️ Please upload both GA and MMM files.")
    else:
        with st.status("🚀 Initializing Nexus Engine...", expanded=True) as status:
            try:
                temp_dir = Path("temp_uploads")
                output_dir = Path("output")
                os.makedirs(temp_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)
                
                ga_path = temp_dir / "ga_data.csv"
                mmm_path = temp_dir / "mmm_data.csv"
                with open(ga_path, "wb") as f: f.write(ga_file.getbuffer())
                with open(mmm_path, "wb") as f: f.write(mmm_file.getbuffer())
                
                st.session_state.results = pipeline_main.run_analysis_pipeline(
                    ga_path, mmm_path, output_dir, status_callback=lambda m: status.write(m)
                )
                
                if not st.session_state.messages:
                    st.session_state.messages = [{"role": "assistant", "content": "Analysis complete. I have processed your data. What would you like to know?"}]
                
                status.update(label="✅ Analysis Ready!", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline Error: {e}")

# --- DASHBOARD LAYOUT ---
if st.session_state.results:
    res = st.session_state.results
    col_main, col_chat = st.columns([2.2, 1], gap="medium") if show_chat else (st.container(), None)

    with col_main:
        tab_data, tab_viz, tab_report = st.tabs(["📊 Data Tables", "🎨 Visual Analysis", "📝 Strategic Report"])

        # --- TAB 1: DATA TABLES ---
        with tab_data:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("ROI vs Attribution")
                df_roi = res['prior_df'][['channel', 'roi', 'mmm_share', 'attr_weight']].copy()
                df_roi.columns = ['Channel', 'ROI', 'MMM Share', 'Attribution']
                
                # --- APPLY BUPU STYLING (FORCED) ---
                styled_roi = df_roi.style.background_gradient(cmap='BuPu', subset=['ROI', 'Attribution'])\
                    .format({'ROI': "{:.2f}", 'MMM Share': "{:.1%}", 'Attribution': "{:.1%}"})
                
                st.dataframe(styled_roi, hide_index=True, use_container_width=True)

            with col2:
                st.subheader("Attribution Confidence")
                df_sigma = res['prior_df'][['channel', 'attr_weight', 'sigma', 'confidence']].copy()
                df_sigma.columns = ['Channel', 'Attribution', 'Sigma', 'Confidence']
                st.dataframe(
                    df_sigma.style.format({'Attribution': '{:.2%}', 'Sigma': '{:.4f}'}),
                    column_config={"Confidence": st.column_config.ProgressColumn("Confidence Score", format="%.2f", min_value=0, max_value=1)},
                    hide_index=True, use_container_width=True
                )
            
            st.divider()
            st.subheader("Top Conversion Paths")
            st.dataframe(res['top_paths'].head(10), hide_index=True, use_container_width=True)

            st.divider()
            st.subheader("Removal Effects")
            removal_path = Path("output/removal_effects.png").absolute()
            if removal_path.exists():
                st.image(str(removal_path), use_container_width=True)

        # --- TAB 2: VISUALS ---
        with tab_viz:
            st.markdown("### Attribution Uncertainty Plot")
            forest_path = Path("output/consensus_attribution.png").absolute()
            if forest_path.exists():
                st.image(str(forest_path), caption="Consensus Attribution", use_container_width=True)
            else:
                st.warning("Forest plot not found.")
                
            st.divider()
            st.markdown("### Customer Journey Flow")
            sankey_path = Path("output/journey_sankey.html").absolute()
            if sankey_path.exists():
                with open(sankey_path, 'r', encoding='utf-8') as f:
                    st.components.v1.html(f.read(), height=700, scrolling=True)

        # --- TAB 3: REPORT ---
        with tab_report:
            report = res.get('report_data', {})
            st.markdown(f"#### 1. Executive Summary\n<div class='report-box'>{report.get('executive_summary', 'N/A')}</div>", unsafe_allow_html=True)
            c_r1, c_r2 = st.columns(2)
            c_r1.info(f"**Optimization:** {report.get('removal_insight', 'N/A')}")
            c_r2.warning(f"**Friction:** {report.get('q_matrix_insight', 'N/A')}")

    # === RIGHT COLUMN: CHAT ===
    if col_chat:
        with col_chat:
            st.markdown("#### 💬 Nexus Assistant")
            with st.container(height=500, border=True):
                for m in st.session_state.messages:
                    with st.chat_message(m["role"]): st.markdown(m["content"])
            
            if prompt := st.chat_input("Ask Nexus..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()