import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add 'src' to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import src.main as pipeline_main
import src.llm_analyst as nexus

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Nexus | Strategic Marketing AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# --- CUSTOM CSS FOR ELEGANT TABLES ---
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
    /* Enhance Table Headers */
    th {
        background-color: #F8FAFC !important;
        color: #334155 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: SETUP & UPLOAD ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8637/8637103.png", width=70)
    st.markdown("### Nexus Control")
    st.caption("Secure Cloud Environment")
    
    # Security Check
    if os.getenv("GEMINI_API_KEY"):
        st.success("🔐 API Key Active")
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

# --- HELPER: ELEGANT TABLE FORMATTING ---
def format_percent(val):
    return "{:.2%}".format(val) if isinstance(val, (float, int)) else val

def format_currency(val):
    return "${:,.2f}".format(val) if isinstance(val, (float, int)) else val

def format_float(val):
    return "{:.4f}".format(val) if isinstance(val, (float, int)) else val

# --- MAIN LOGIC ---

if "results" not in st.session_state:
    st.session_state.results = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Header
st.markdown('<div class="main-header">🧠 Nexus Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI-Powered Marketing Attribution & Strategy</div>', unsafe_allow_html=True)

if process_btn:
    if not ga_file or not mmm_file:
        st.warning("⚠️ Please upload both GA and MMM files to begin.")
    else:
        with st.spinner("⏳ Nexus is processing data & generating insights..."):
            try:
                # 1. Setup Temp Paths
                temp_dir = Path("temp_uploads")
                output_dir = Path("output")
                os.makedirs(temp_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)
                
                ga_path = temp_dir / "ga_data.csv"
                mmm_path = temp_dir / "mmm_data.csv"
                
                with open(ga_path, "wb") as f: f.write(ga_file.getbuffer())
                with open(mmm_path, "wb") as f: f.write(mmm_file.getbuffer())
                
                # 2. Run Pipeline
                results = pipeline_main.run_analysis_pipeline(ga_path, mmm_path, output_dir)
                st.session_state.results = results
                
                # 3. Initialize Chat
                if not st.session_state.messages:
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Analysis complete. I have processed your Attribution, MMM, and Shapley values. What would you like to know?"}
                    ]
                
                st.success("✅ Analysis Ready!")
                
            except Exception as e:
                st.error(f"Pipeline Error: {e}")
                st.code(str(e))

# --- DISPLAY TABS ---
if st.session_state.results:
    res = st.session_state.results
    
    # NEW TAB STRUCTURE: Chat First, Then Data
    tab_chat, tab_data, tab_viz = st.tabs(["🧠 Nexus Chat", "📊 Data Tables", "🗺️ Journey Maps"])
    
    # --- TAB 1: NEXUS CHAT (Primary Interface) ---
    with tab_chat:
        # Chat container
        chat_container = st.container(height=500)
        
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Input
        if prompt := st.chat_input("Ask about ROI, Channels, or Strategy..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    placeholder.markdown("🧠 *Thinking...*")
                    
                    # Context Injection
                    image_path = None
                    if 'matrix' in prompt.lower(): image_path = str(res['img_paths']['q_matrix'])
                    elif 'removal' in prompt.lower(): image_path = str(res['img_paths']['removal'])
                    
                    context_str = res['prior_df'].to_string()
                    response = nexus.chat_with_data(prompt, context_str, image_path)
                    
                    placeholder.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

    # --- TAB 2: DATA TABLES (Elegant Formats) ---
    with tab_data:
        st.caption("Raw strategic data processed by the pipeline.")
        
        col1, col2 = st.columns(2)
        
        # 1. ROI vs Attribution
        with col1:
            st.subheader("💰 ROI vs Attribution")
            df_roi = res['prior_df'][['channel', 'roi', 'mmm_share', 'attr_weight']].copy()
            # Style
            st.dataframe(
                df_roi.style.format({
                    'roi': "{:.2f}",
                    'mmm_share': "{:.2%}",
                    'attr_weight': "{:.2%}"
                }).background_gradient(subset=['roi'], cmap="Greens"),
                use_container_width=True
            )

        # 2. Attribution with Sigma (Confidence)
        with col2:
            st.subheader("📉 Attribution Confidence (Sigma)")
            df_sigma = res['prior_df'][['channel', 'attr_weight', 'sigma', 'confidence']].copy()
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

        # 3. Markov Removal Effects
        with col3:
            st.subheader("❌ Markov Removal Effects")
            # Need to grab this from report logic or pipeline. 
            # In main.py, markov_removal is a Series inside the pipeline but we didn't explicitly return it in 'res' dict 
            # except implicitly via the images. 
            # *Self-Correction*: The pipeline logic I gave you saved the PLOTS but didn't return the raw dataframe in 'res'. 
            # However, 'prior_df' likely has everything we need or we can infer it. 
            # Let's show the 'prior_df' again but focused on the removal logic if available, 
            # OR better: use the 'top_paths' which IS available.
            
            # Since 'markov_removal' raw series wasn't in the return dict of the last main.py, 
            # I will display the Attribution Weight (which is derived from it) as a proxy, 
            # or we rely on the Forest Plot image below.
            st.image(str(res['img_paths']['removal']), use_container_width=True)

        # 4. Shapley Top Paths
        with col4:
            st.subheader("🛣️ Top Shapley Conversion Paths")
            df_paths = res['top_paths'].head(10).copy()
            st.dataframe(
                df_paths[['path', 'conversions', 'share_of_conversions_pct']].style.format({
                    'share_of_conversions_pct': "{:.2f}%"
                }),
                use_container_width=True
            )

    # --- TAB 3: VISUAL JOURNEYS ---
    with tab_viz:
        st.subheader("Customer Journey Flow (Sankey)")
        if os.path.exists(res['sankey_path']):
            with open(res['sankey_path'], 'r', encoding='utf-8') as f:
                html_data = f.read()
            st.components.v1.html(html_data, height=600, scrolling=True)
            
        st.divider()
        st.subheader("Transition Probabilities (Q-Matrix)")
        st.image(str(res['img_paths']['q_matrix']), caption="User Flow Heatmap", width=700)