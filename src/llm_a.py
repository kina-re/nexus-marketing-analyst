import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_ID = "gemini-flash-lite-latest"
API_KEY = os.getenv("GEMINI_API_KEY")

# GLOBAL CHAT SESSION (For memory)
chat_session = None

def get_client():
    if not API_KEY:
        return None
    return genai.Client(api_key=API_KEY)

def get_safety_config():
    return [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
    ]

def chat_with_data(user_question, internal_context_str):
    """
    Conversational Chat: Uses a persistent session to remember context.
    """
    global chat_session
    client = get_client()
    if not client: return "⚠️ API Key Missing."

    # 1. INITIALIZE CHAT (Only happens once)
    if chat_session is None:
        system_instruction = f"""
        [ROLE]
        You are 'Nexus', a Strategic Marketing Partner for a Professional Upskilling company.
        
        [CONTEXT - INTERNAL DATA]
        {internal_context_str}
        
        [BEHAVIOR]
        1. **Direct Answers:** Do not repeat the question. Get straight to the answer.
        2. **Competitor Search:** If asked about competitors (UpGrad, Newton School), use Google Search.
        3. **Tone:** Professional, concise, and advisory. Use "You" and "Your".
        """
        
        chat_session = client.chats.create(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.7,
                max_output_tokens=800,
                safety_settings=get_safety_config()
            )
        )

    # 2. SEND MESSAGE
    try:
        # Be nice to the Free Tier (Rate Limit protection)
        time.sleep(2) 
        
        response = chat_session.send_message(user_question)
        
        if response.text:
            return response.text.strip()
        return "⚠️ (No text response)"

    except Exception as e:
        if "429" in str(e):
             return "⚠️ System is cooling down (Rate Limit). Please wait 10s."
        # If session breaks, reset it for next time
        chat_session = None 
        return f"⚠️ Connection Reset: {str(e)}"

def generate_marketing_insight(channel_data, reasoning_context=None):
    """
    Single Insight Generation (No Chat Memory)
    """
    client = get_client()
    if not client: return "Insight unavailable."

    # FIX: Format the ROI to 2 decimal places to avoid messy numbers
    roi_val = float(channel_data['mmm_roi'])
    formatted_roi = f"{roi_val:.2f}"

    prompt = f"""
    Internal Stat: {channel_data['channel']} ROI is {formatted_roi}.
    Task: Search site:statista.com or site:crunchbase.com for {channel_data['channel']} benchmarks in EdTech/Upskilling.
    Compare our ROI to the industry average. Keep it to 2 sentences.
    """
    
    # Retry Logic (Loop)
    for attempt in range(3):
        try:
            time.sleep(5) # Rate limit safety
            
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.3,
                    safety_settings=get_safety_config()
                )
            )
            
            if response.text:
                return response.text.strip() # <--- Returns immediately, preventing loop repetition
                
        except Exception as e:
            if "429" in str(e):
                time.sleep(10) # Wait longer if rate limited
                continue
            return f"⚠️ Insight Error: {str(e)}"

    return f"Your {channel_data['channel']} ROI is {formatted_roi}."