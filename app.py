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

# --- 3. CUSTOM CSS (FIXED TABLES + WIDTHS) ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #0F172A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-text {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 20px;
    }
    
    /* --- STRICT WHITE TABLE STYLING --- */
    /* Remove outer container styling to prevent white borders/gaps */
    [data-testid="stDataFrame"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0px !important;
        width: 100% !important; /* Forces full width without the warning */
    }
    
    /* Style the internal wrapper */
    [data-testid="stDataFrame"] > div {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
        width: 100% !important;
    }
    
    /* Headers */
    th {
        background-color: #FFFFFF !important;
        color: #475569 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #E2E8F0 !important;
    }
    
    /* Cells */
    td {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        font-size: 0.95rem !important;
        border-bottom: 1px solid #F1F5F9 !important;
    }

    /* Chat Container */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px;
        border: 1px solid #E2E8F0;
    }
    
    /* Report Box */
    .report-box {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 20px;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8637/8637103.png", width=70)
    st.markdown("### Nexus Control")
    st.caption("Upload your data here")
    
    if os.getenv("GEMINI_API_KEY"):
        btn_disabled = False
    else:
        st.error("❌ API Key Missing (.env)")
        btn_disabled = True
    
    st.divider()
    st.write("**Data Upload**")
    ga_file = st.file_uploader("Google Analytics (CSV)", type=["csv"])
    mmm_file = st.file_uploader("MMM Data (CSV)", type=["csv"])
    
    st.markdown("---")
    # Warning fixed: Removed use_container_width if it causes issues, but usually fine for buttons.
    # If this warns, we can remove it, but buttons usually need it to look good.
    process_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=btn_disabled)
    st.markdown("---")
    show_chat = st.toggle("💬 Show Assistant", value=True)

# --- HEADER ---
st.markdown('<div class="main-header">Nexus Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI-Powered Marketing Attribution & Strategy</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Please upload ga_2.csv and mmm_data_2016_2017.csv. If sidebar is not visible please click on arrows.</div>', unsafe_allow_html=True)

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
                
                def ui_logger(message):
                    status.write(message)
                    status.update(label=message)
                
                results = pipeline_main.run_analysis_pipeline(
                    ga_path, mmm_path, output_dir, status_callback=ui_logger
                )
                
                st.session_state.results = results
                
                if not st.session_state.messages:
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Analysis complete. I have processed your data using Markov chain, MMM, and Shapley models. What would you like to know?"}
                    ]
                
                status.update(label="✅ Analysis Ready!", state="complete", expanded=False)
                st.rerun()
                
            except Exception as e:
                status.update(label="❌ Pipeline Failed", state="error")
                st.error(f"Pipeline Error: {e}")

# --- DASHBOARD LAYOUT ---
if st.session_state.results:
    res = st.session_state.results

    if show_chat:
        col_main, col_chat = st.columns([2.2, 1], gap="medium")
    else:
        col_main = st.container()
        col_chat = None

    # === LEFT COLUMN: MAIN DASHBOARD ===
    with col_main:
        tab_data, tab_viz, tab_report = st.tabs(["📊 Data Tables", "🎨 Visual Analysis", "📝 Strategic Report"])

        # --- TAB 1: DATA TABLES ---
        with tab_data:
            st.caption("Raw strategic data processed by the pipeline.")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("ROI vs Attribution")
                df_roi = res['prior_df'][['channel', 'roi', 'mmm_share', 'attr_weight']].copy()
                df_roi.columns = ['Channel', 'ROI', 'MMM Share', 'Attribution']
                # FIX: Removed use_container_width=True to stop warning. CSS handles width.
                st.dataframe(
                    df_roi.style
                    .format({'ROI': "{:.2f}", 'MMM Share': "{:.1%}", 'Attribution': "{:.1%}"})
                    .set_properties(**{'background-color': '#FFFFFF', 'color': '#1E293B', 'border-color': '#E2E8F0'})
                    .hide(axis="index")
                )

            with col2:
                st.subheader("Attribution Confidence")
                df_sigma = res['prior_df'][['channel', 'attr_weight', 'sigma', 'confidence']].copy()
                df_sigma.columns = ['Channel', 'Attribution', 'Sigma', 'Confidence']
                # FIX: Removed use_container_width=True to stop warning. CSS handles width.
                st.dataframe(
                    df_sigma.style
                    .format({'Attribution': '{:.2%}', 'Sigma': '{:.4f}'}) 
                    .set_properties(**{'background-color': '#FFFFFF', 'color': '#1E293B'}),
                    column_config={
                        "Confidence": st.column_config.ProgressColumn(
                            "Confidence Score", format="%.2f", min_value=0, max_value=1
                        )
                    },
                    hide_index=True
                )
            
            st.divider()
            
            # Top Paths
            st.subheader("Top Conversion Paths")
            df_paths = res['top_paths'].head(10).copy()
            # FIX: Removed use_container_width=True to stop warning. CSS handles width.
            st.dataframe(
                df_paths[['path', 'conversions', 'share_of_conversions_pct']].style
                .format({'share_of_conversions_pct': "{:.2f}%"})
                .set_properties(**{'background-color': '#FFFFFF', 'color': '#1E293B'})
                .hide(axis="index")
            )

            st.divider()

            # Removal Effects
            st.subheader("❌ Removal Effects")
            if os.path.exists(res['img_paths']['removal']):
                st.image(Image.open(res['img_paths']['removal']), use_container_width=True)

        # --- TAB 2: VISUALS ---
        with tab_viz:
            st.markdown("###  Attribution Uncertainty Plot")
            if res.get('forest_path') and os.path.exists(res['forest_path']):
                st.image(Image.open(res['forest_path']), caption="Consensus Attribution", use_container_width=True)
            else:
                st.warning("Forest plot not found.")
                
            st.divider()
            
            st.markdown("###  Customer Journey Flow")
            if os.path.exists(res['sankey_path']):
                with open(res['sankey_path'], 'r', encoding='utf-8') as f:
                    st.components.v1.html(f.read(), height=700, scrolling=True)
            
            st.divider()

            st.markdown("### Transition Probabilities")
            if os.path.exists(res['img_paths']['q_matrix']):
                c1, c2, c3 = st.columns([1, 4, 1])
                with c2:
                    st.image(Image.open(res['img_paths']['q_matrix']), caption="User Flow Heatmap", use_container_width=True)

        # --- TAB 3: REPORT ---
        with tab_report:
            st.subheader("📝 AI Strategic Analysis")
            if res.get('pdf_path') and os.path.exists(res['pdf_path']):
                with open(res['pdf_path'], "rb") as pdf_file:
                    st.download_button(
                        label="📄 Download Full PDF Report",
                        data=pdf_file,
                        file_name="Nexus_Strategy_Report.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
            
            st.divider()
            report = res.get('report_data', {})
            st.markdown("#### 1. Executive Summary")
            st.markdown(f"<div class='report-box'>{report.get('executive_summary', 'No summary available.')}</div>", unsafe_allow_html=True)
            
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown("#### 2. Optimization Opportunities")
                st.info(report.get('removal_insight', 'No insight available.'))
            with col_r2:
                st.markdown("#### 3. Journey Friction")
                st.warning(report.get('q_matrix_insight', 'No insight available.'))

    # === RIGHT COLUMN: NEXUS CHAT (CRASH-PROOF SMART MODE) ===
    if col_chat:
        with col_chat:
            st.markdown("#### 💬 Nexus Assistant")
            chat_box = st.container(height=500, border=True)
            
            with chat_box:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Ask Nexus..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        placeholder.markdown("*Thinking...*")
                        
                        # --- SMART CONTEXT ROUTING (TEXT ONLY) ---
                        p_lower = prompt.lower()
                        context_str = f"=== KEY ATTRIBUTION DATA ===\n{res['prior_df'].to_string()}\n"

                        # Instead of sending the IMAGE FILE (which crashes),
                        # we send a DESCRIPTION of the image so the AI knows what you are looking at.
                        
                        if any(x in p_lower for x in ['journey', 'flow', 'path', 'sankey', 'steps']):
                            top_paths_txt = res['top_paths'].head(15).to_string()
                            context_str += f"\n=== TOP CUSTOMER PATHS (Sankey Data) ===\n{top_paths_txt}\n"
                            context_str += "\n[SYSTEM NOTE: The user is asking about the Sankey Diagram visual. Use the path data above to explain the flow.]"

                        elif any(x in p_lower for x in ['matrix', 'probability', 'transition', 'heatmap', 'grid']):
                            context_str += "\n[SYSTEM NOTE: The user is viewing the Q-Matrix (Transition Probabilities) heatmap. Explain that this shows the likelihood of moving from one channel to another.]"

                        elif any(x in p_lower for x in ['removal', 'value', 'impact', 'lose']):
                            context_str += "\n[SYSTEM NOTE: The user is asking about the Removal Effects bar chart. Explain which channels cause the biggest drop in conversions if removed.]"

                        # Forest Plot / Uncertainty Condition
                        elif any(x in p_lower for x in ['forest', 'plot', 'uncertainty', 'confidence', 'attribution', 'model', 'shapley']):
                            context_str += "\n[SYSTEM NOTE: The user is viewing the Attribution Forest Plot. Explain that dots represent the calculated weight and lines represent the error margin/uncertainty.]"

                        # --- SAFE EXECUTION ---
                        try:
                            # WE PASS 'None' FOR IMAGE TO PREVENT CRASH
                            response = nexus.chat_with_data(prompt, context_str, None)
                            placeholder.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            placeholder.error(f"AI Error: {e}")