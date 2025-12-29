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

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Nexus | Strategic Marketing AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
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
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8637/8637103.png", width=70)
    st.markdown("### Nexus Control")
    st.caption("Secure Cloud Environment")
    
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

# --- MAIN LOGIC ---
if "results" not in st.session_state:
    st.session_state.results = None

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown('<div class="main-header">🧠 Nexus Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI-Powered Marketing Attribution & Strategy</div>', unsafe_allow_html=True)

if process_btn:
    if not ga_file or not mmm_file:
        st.warning("⚠️ Please upload both GA and MMM files.")
    else:
        # --- REPLACED st.spinner WITH st.status ---
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
                
                # Define the callback function to update the UI
                def ui_logger(message):
                    st.write(message) # Adds a line to the status box
                    status.update(label=message) # Updates the box title
                
                # Pass the logger to the pipeline
                results = pipeline_main.run_analysis_pipeline(
                    ga_path, 
                    mmm_path, 
                    output_dir, 
                    status_callback=ui_logger # <--- PASSING CALLBACK HERE
                )
                
                st.session_state.results = results
                
                if not st.session_state.messages:
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Analysis complete. I have processed your Attribution, MMM, and Shapley values. What would you like to know?"}
                    ]
                
                # Mark as complete and collapse
                status.update(label="✅ Analysis Ready!", state="complete", expanded=False)
                st.success("Analysis Pipeline Finished Successfully.")
                
            except Exception as e:
                status.update(label="❌ Pipeline Failed", state="error")
                st.error(f"Pipeline Error: {e}")

# --- TABS ---
if st.session_state.results:
    res = st.session_state.results
    
    tab_chat, tab_data, tab_viz = st.tabs(["🧠 Nexus Chat", "📊 Data Tables", "🗺️ Journey Maps"])
    
    # --- TAB 1: CHAT ---
    with tab_chat:
        chat_container = st.container(height=500)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Ask about ROI, Channels, or Strategy..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    placeholder.markdown("🧠 *Thinking...*")
                    
                    image_path = None
                    if 'matrix' in prompt.lower(): image_path = str(res['img_paths']['q_matrix'])
                    elif 'removal' in prompt.lower(): image_path = str(res['img_paths']['removal'])
                    
                    context_str = res['prior_df'].to_string()
                    response = nexus.chat_with_data(prompt, context_str, image_path)
                    placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # --- TAB 2: DATA TABLES (Fixed Index) ---
    with tab_data:
        st.caption("Raw strategic data processed by the pipeline.")
        col1, col2 = st.columns(2)
        
        # 1. ROI TABLE
        with col1:
            st.subheader("💰 ROI vs Attribution")
            df_roi = res['prior_df'][['channel', 'roi', 'mmm_share', 'attr_weight']].copy()
            
            # --- FIX: Start Index at 1 ---
            df_roi.index = range(1, len(df_roi) + 1)
            
            st.dataframe(
                df_roi.style.format({
                    'roi': "{:.2f}",
                    'mmm_share': "{:.2%}",
                    'attr_weight': "{:.2%}"
                }).background_gradient(subset=['roi'], cmap="Greens"),
                use_container_width=True
            )

        # 2. CONFIDENCE TABLE
        with col2:
            st.subheader("📉 Attribution Confidence (Sigma)")
            df_sigma = res['prior_df'][['channel', 'attr_weight', 'sigma', 'confidence']].copy()
            
            # --- FIX: Start Index at 1 ---
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

        # 3. REMOVAL EFFECTS
        with col3:
            st.subheader("❌ Markov Removal Effects")
            st.image(str(res['img_paths']['removal']), use_container_width=True)

        # 4. TOP PATHS TABLE
        with col4:
            st.subheader("🛣️ Top Shapley Conversion Paths")
            df_paths = res['top_paths'].head(10).copy()
            
            # --- FIX: Start Index at 1 ---
            df_paths.index = range(1, len(df_paths) + 1)
            
            st.dataframe(
                df_paths[['path', 'conversions', 'share_of_conversions_pct']].style.format({
                    'share_of_conversions_pct': "{:.2f}%"
                }),
                use_container_width=True
            )

    # --- TAB 3: VISUALS ---
    with tab_viz:
        # 1. Forest Plot (Top - The Strategic Summary)
        st.markdown("### 🌲 Attribution Uncertainty (Forest Plot)")
        if res.get('forest_path') and os.path.exists(res['forest_path']):
            img = Image.open(res['forest_path'])
            st.image(img, caption="Consensus Attribution (Markov + Shapley)", use_container_width=True)
        else:
            st.warning("Forest plot not found. Check Step 4 in main.py.")
            
        st.divider()
        
        # 2. Sankey Diagram (Middle - The Journey Story)
        st.markdown("### 🌊 Customer Journey Flow")
        st.caption("Visualizes how users move between channels before converting.")
        
        if os.path.exists(res['sankey_path']):
            with open(res['sankey_path'], 'r', encoding='utf-8') as f:
                html_data = f.read()
            # Give it full width and plenty of height
            st.components.v1.html(html_data, height=700, scrolling=True)
        
        st.divider()

        # 3. Transition Matrix (Bottom - The Mathematical Details)
        st.markdown("### 🔄 Transition Probabilities (Q-Matrix)")
        st.caption("Heatmap showing the probability of moving from one channel to another.")
        
        if os.path.exists(res['img_paths']['q_matrix']):
            q_img = Image.open(res['img_paths']['q_matrix'])
            
            # Optional: Center the heatmap so it doesn't stretch weirdly on huge screens
            c1, c2, c3 = st.columns([1, 3, 1]) 
            with c2:
                st.image(q_img, caption="User Flow Heatmap", use_container_width=True)