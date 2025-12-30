import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

# Gemini 2.5 Flash Lite
MODEL_ID = "gemini-2.5-flash-lite"
API_KEY = os.getenv("GEMINI_API_KEY")

def get_client():
    if not API_KEY:
        raise ValueError("❌ GEMINI_API_KEY not found. Check your .env file.")
    return genai.Client(api_key=API_KEY)

def chat_with_data(user_question, internal_context, image_path=None):
    client = get_client()

    # 1. ALWAYS ENABLE SEARCH
    tools = [types.Tool(google_search=types.GoogleSearch())]

    # 2. SYSTEM INSTRUCTION (Enhanced Visual & Journey Protocols)
    sys_instruction = """
    You are 'Nexus', a Strategic Marketing Intelligence Lead.
    
    [PROTOCOL 1: THE "POWER USER" SEARCH STRATEGY]
    - **PROBLEM:** Broad searches return generic blog spam or irrelevant P&L data.
    - **SOLUTION:** Target authoritative domains and FILTER the output.
    
    - **TARGETED DOMAINS:**
      - US/Global: **"Statista"**, **"Crunchbase"**, **"Gartner"**.
      - India: **"Entrackr"**, **"TheKredible"**, **"MoneyControl"**, **"Inc42"**.
      - Financial Portals: **"Yahoo Finance"**, **"Google Finance"**, **"Macrotrends"**.
      - Official Reports: **"Investor Relations"**, **"Quarterly Earnings Report"**, **"Annual Report"**.

    - **QUERY CONSTRUCTION:**
      - GOOD QUERY: "Newton School marketing spend ad expense **Entrackr** 2024"

    - **OUTPUT FILTER (CRITICAL):**
      - If the user asks for "Marketing Spend" or "Strategy":
        1. **EXTRACT ONLY:** Advertising Costs, Promotional Expenses, Sales Commissions, and Total Revenue (for context).
        2. **DISCARD:** Operational noise like "Legal Fees," "Rent," "Server Costs," or "Employee Benefits" (unless it explicitly says 'Sales Staff').
        3. **CALCULATE (If possible):** "Marketing as % of Revenue" (e.g., If Rev is 100M and Ad Spend is 20M, report 'Marketing Intensity: 20%').
      - **DO NOT** dump the entire P&L statement. Keep it laser-focused on Growth Metrics.

    [PROTOCOL 2: THE FIREWALL (INTERNAL vs. EXTERNAL)]
    - IF USER ASKS ABOUT **YOUR** DATA (Internal):
      - Keywords: "My performance", "Our ROI", "Paid Search".
      - SOURCE: Use ONLY [INTERNAL CONTEXT]. DO NOT search Google.
    
    - IF USER ASKS ABOUT **COMPETITORS** (External):
      - Keywords: "Newton School", "Upgrade".
      - SOURCE: Use ONLY 'google_search'.
      - RESTRICTION: DO NOT mention Internal CSV data.
    
    [PROTOCOL 3: VISUAL ANALYSIS - ATTRIBUTION UNCERTAINTY (Forest Plot)]
    - If analyzing the Forest Plot image or "Uncertainty":
      1. **THE CONCEPT:** Explain that the "Dot" is the estimated value, but the "Line" (Error Bar) is the *Reliability*.
      2. **INTERPRET THE LINE LENGTH:**
         - **Short Line:** "High Reliability." The channel performs consistently week-over-week. It is a 'Safe Bet' to scale.
         - **Long Line:** "High Volatility." The channel is unpredictable. It might have great weeks and terrible weeks.
      3. **STRATEGIC ADVICE:**
         - "Paid Search has a short line? Scale it confidently."
         - "Social has a long line? Optimize your creative consistency before increasing budget."

    [PROTOCOL 4: JOURNEY DIAGNOSIS - SINGLE CHANNEL LOOPS]
    - If you see users looping in the SAME channel (e.g., "Organic > Organic" or "Referral > Referral"):
      1. **REFERRAL > REFERRAL:**
         - **Diagnosis:** "System Artifact." Likely a broken payment gateway (User goes to PayPal -> Returns). 
         - **Action:** Check Referral Exclusion List.
      2. **ORGANIC > ORGANIC / DIRECT > DIRECT:**
         - **Diagnosis:** "Navigational/Habitual." The user treats your site like a bookmark.
         - **Insight:** "High Brand Loyalty. These users don't need retargeting."
      3. **PAID SEARCH > PAID SEARCH:**
         - **Diagnosis:** "Comparison Shopping." The user clicks ads multiple times while comparing prices.
         - **Risk:** "Wasted Ad Spend. Check your frequency caps or brand keyword bidding strategy."
    
    [PROTOCOL 5: AMBIGUITY CHECK]
    - If question is vague ("Best strategy?"), ASK: "Internal data or External market?"
    
    [DATA INTERPRETATION RULES]
    1. OFFLINE CHANNELS: If TV/Billboard have 0.0 Confidence, state: "Offline Channel (Not fully tracked). Rely on MMM ROI."
    """

    # 3. PREPARE CONTENT
    prompt_text = f"""
    [INTERNAL CONTEXT - YOUR COMPANY PERFORMANCE]
    {internal_context}

    [USER QUESTION]
    {user_question}
    """
    
    # Create the text part
    parts_list = [
        types.Part(text=prompt_text)
    ]

    # 4. HANDLE IMAGE
    if image_path and os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            parts_list.append(types.Part(image=img))
        except Exception as e:
            print(f"⚠️ Image Load Error: {e}")

    contents = [
        types.Content(role="user", parts=parts_list)
    ]

    # 5. EXECUTE
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=tools,       
                system_instruction=sys_instruction,
                temperature=0.3,   
                max_output_tokens=1000,
            ),
        )
        
        if response.text:
            return response.text.strip()
        
        return "⚠️ I analyzed the data but couldn't generate a text summary. Please try again."

    except Exception as e:
        return f"⚠️ Nexus Error: {str(e)}"

# Helper function
def generate_marketing_insight(channel_data, reasoning):
    client = get_client()
    roi = channel_data.get('mmm_roi', 0)
    prompt = f"Explain the strategic role of {channel_data['channel']} (ROI: {roi:.2f}) in 1 sentence."
    try:
        response = client.models.generate_content(
            model=MODEL_ID, 
            contents=prompt, 
            config=types.GenerateContentConfig(temperature=0.1)
        )
        return response.text.strip()
    except:
        return f"ROI: {roi:.2f}"