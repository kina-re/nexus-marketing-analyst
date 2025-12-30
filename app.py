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

# --- 1. INITIALIZE STATE (Must be before set_page_config) ---
if "results" not in st.session_state:
    st.session_state.results = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. DYNAMIC SIDEBAR LOGIC ---
# If results exist, we force the sidebar to collapse. Otherwise, it stays open.
sidebar_state = "collapsed" if st.session_state.results else "expanded"

st.set_page_config(
    page_title="Nexus | Strategic Marketing AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state=sidebar_state # <--- THIS IS THE FIX
)

load_dotenv()

# --- CUSTOM CSS ---
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
    th {
        background-color: #F8FAFC !important;
        color: #334155 !important;
        font-weight: 600 !important;
    }
    /* Style for the Chat Container */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px;
        border: 1px solid #E2E8F0;
    }
    /* Report Box Styling */
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
    process_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=btn_disabled)

    # Chat Toggle
    st.markdown("---")
    show_chat = st.toggle("💬 Show Assistant", value=True)

# --- HEADER ---
st.markdown('<div class="main-header">🧠 Nexus Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI-Powered Marketing Attribution & Strategy</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Please upload ga_2.csv and mmm_data_2016_2017.csv. If sidebar is not visible please click on arrows.</div>', unsafe_allow_html=True)

# --- MAIN PROCESS LOGIC ---
if process_btn:
    if not ga_file or not mmm_file:
        st.warning("⚠️ Please upload both GA and MMM files.")
    else:
        # Start Status Box
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
                
                # Run Pipeline
                results = pipeline_main.run_analysis_pipeline(
                    ga_path, 
                    mmm_path, 
                    output_dir, 
                    status_callback=ui_logger
                )
                
                # Save to Session State
                st.session_state.results = results
                
                # Initialize Chat History
                if not st.session_state.messages:
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Analysis complete. I have processed your data using Markov chain, MMM, and Shapley models. What would you like to know?"}
                    ]
                
                status.update(label="✅ Analysis Ready!", state="complete", expanded=False)
                
                # 🟢 FORCE RERUN TO COLLAPSE SIDEBAR
                st.rerun()
                
            except Exception as e:
                status.update(label="❌ Pipeline Failed", state="error")
                st.error(f"Pipeline Error: {e}")

# --- DASHBOARD LAYOUT ---
if st.session_state.results:
    res = st.session_state.results

    # 1. LAYOUT DEFINITION (Left=Chat, Right=Data)
    if show_chat:
        # Chat gets 33%, Data gets 67% (Broader chat as requested)
        col_chat, col_main = st.columns([1, 2], gap="small")
    else:
        # Chat hidden, Data takes full width
        col_chat = None 
        col_main = st.container()

    # === LEFT COLUMN: NEXUS CHAT ===
    if col_chat:
        with col_chat:
            st.markdown("#### 💬 Nexus Assistant")
            chat_box = st.container(height=800, border=True)
            
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
                        placeholder.markdown("🧠 *Thinking...*")
                        
                        # SMART CONTEXT ROUTING
                        p_lower = prompt.lower()
                        context_str = f"=== KEY ATTRIBUTION DATA ===\n{res['prior_df'].to_string()}\n"
                        image_path = None

                        if any(x in p_lower for x in ['journey', 'flow', 'path', 'sankey', 'steps']):
                            top_paths_txt = res['top_paths'].head(15).to_string()
                            context_str += f"\n=== TOP CUSTOMER PATHS (Sankey Data) ===\n{top_paths_txt}\n"
                            context_str += "\n(NOTE: These paths represent the User Journey Flow. Explain them sequentially.)"

                        elif any(x in p_lower for x in ['matrix', 'probability', 'transition', 'heatmap', 'grid']):
                            if os.path.exists(res['img_paths']['q_matrix']):
                                image_path = str(res['img_paths']['q_matrix'])
                                context_str += "\n(NOTE: The attached image is the Q-Matrix (Transition Probabilities).)\n"

                        elif any(x in p_lower for x in ['forest', 'uncertainty', 'confidence', 'attribution', 'model', 'shapley']):
                            if os.path.exists(res['forest_path']):
                                image_path = str(res['forest_path'])

                        elif any(x in p_lower for x in ['removal', 'value', 'impact', 'lose']):
                            if os.path.exists(res['img_paths']['removal']):
                                image_path = str(res['img_paths']['removal'])

                        # Added 'plot' and 'chart' to catch "forest plot" requests
                        elif any(x in p_lower for x in ['forest', 'plot', 'uncertainty', 'confidence', 'attribution', 'model', 'shapley']):
                            if os.path.exists(res['forest_path']):
                                image_path = str(res['forest_path'])
                                # Explicitly tell AI what this image is
                                context_str += "\n(NOTE: The attached image is the Attribution Forest Plot. The dots are weights, lines are error bars.)\n"
                        
                        response = nexus.chat_with_data(prompt, context_str, image_path)
                        placeholder.markdown(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()

    # === RIGHT COLUMN: MAIN CONTENT ===
    with col_main:
        # We put the "Data" tabs here. This effectively replaces the sidebar "Data" area.
        tab_data, tab_viz, tab_report = st.tabs(["📊 Data Tables", "🎨 Visual Analysis", "📝 Strategic Report"])

        # --- TAB 1: DATA TABLES ---
        with tab_data:
            st.caption("Raw strategic data processed by the pipeline.")
            col1, col2 = st.columns(2)
            
            # ROI
            with col1:
                st.subheader("💰 ROI vs Attribution")
                df_roi = res['prior_df'][['channel', 'roi', 'mmm_share', 'attr_weight']].copy()
                df_roi.index = range(1, len(df_roi) + 1)
                st.dataframe(
                    df_roi.style.format({
                        'roi': "{:.2f}",
                        'mmm_share': "{:.2%}",
                        'attr_weight': "{:.2%}"
                    }).background_gradient(subset=['roi'], cmap="Greens"),
                    use_container_width=True
                )

            # SIGMA
            with col2:
                st.subheader("📉 Attribution Confidence")
                df_sigma = res['prior_df'][['channel', 'attr_weight', 'sigma', 'confidence']].copy()
                df_sigma.index = range(1, len(df_sigma) + 1)
                st.dataframe(
                    df_sigma.style.format({
                        'attr_weight': "{:.2%}",
                        'sigma': "{:.4f}",
                        'confidence': "{:.2f}"
                    }).bar(subset=['confidence'], color='#3B82F6'),
                    use_container_width=True
                )
                
            st.divider()
            col3, col4 = st.columns(2)

            with col3:
                st.subheader("❌ Removal Effects")
                if os.path.exists(res['img_paths']['removal']):
                    rem_img = Image.open(res['img_paths']['removal'])
                    st.image(rem_img, use_container_width=True)

            with col4:
                st.subheader("🛣️ Top Conversion Paths")
                df_paths = res['top_paths'].head(10).copy()
                df_paths.index = range(1, len(df_paths) + 1)
                st.dataframe(
                    df_paths[['path', 'conversions', 'share_of_conversions_pct']].style.format({
                        'share_of_conversions_pct': "{:.2f}%"
                    }),
                    use_container_width=True
                )

        # --- TAB 2: VISUALS ---
        with tab_viz:
            st.markdown("###  Attribution Uncertainty Plot")
            if res.get('forest_path') and os.path.exists(res['forest_path']):
                img = Image.open(res['forest_path'])
                st.image(img, caption="Consensus Attribution (Markov + Shapley)", use_container_width=True)
            else:
                st.warning("Forest plot not found.")
                
            st.divider()
            
            st.markdown("###  Customer Journey Flow")
            if os.path.exists(res['sankey_path']):
                with open(res['sankey_path'], 'r', encoding='utf-8') as f:
                    html_data = f.read()
                st.components.v1.html(html_data, height=700, scrolling=True)
            
            st.divider()

            st.markdown("### Transition Probabilities (Q-Matrix)")
            if os.path.exists(res['img_paths']['q_matrix']):
                q_img = Image.open(res['img_paths']['q_matrix'])
                c1, c2, c3 = st.columns([1, 4, 1])
                with c2:
                    st.image(q_img, caption="User Flow Heatmap", use_container_width=True)

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
            else:
                st.warning("PDF Report not available yet.")

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