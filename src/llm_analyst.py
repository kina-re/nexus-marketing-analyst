import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image  # 📸 NEW: Needed to load images

load_dotenv()

# Gemini 2.5 Flash is Multimodal (Text + Images)
MODEL_ID = "gemini-2.5-flash-lite"
API_KEY = os.getenv("GEMINI_API_KEY")

def get_client():
    if not API_KEY:
        raise ValueError("❌ GEMINI_API_KEY not found. Check your .env file.")
    return genai.Client(api_key=API_KEY)

def chat_with_data(user_question, internal_context, image_path=None):
    """
    Analyzes text context AND optional graph images.
    """
    client = get_client()

    # 1. Base Text Prompt
    prompt_text = f"""
    [ROLE]
    You are 'Nexus', a Strategic Marketing Lead.You don't just report data; you interpret the data.
    
    [CONTEXT - PERFORMANCE DATA]
    {internal_context}
    
    [USER QUESTION]
    {user_question}
    
    [INSTRUCTION]
    - If an image is provided, analyze the visual trends (error bars, confidence intervals, channel ranking).
    - If explaining the Forest Plot: The dots are Attribution Weights. The lines are 'Sigma' (uncertainty).
    - If explaining the Sankey: Use the context data to explain the top customer paths.
    - Be professional and concise.
    """
    
    contents = [prompt_text]

    # 2. Handle Image Attachment (Multimodal)
    if image_path:
        if os.path.exists(image_path):
            try:
                # Load image and append to request
                img = Image.open(image_path)
                contents.append(img)
                print("   (📎 Attached Graph Image to prompt)")
            except Exception as e:
                print(f"⚠️ Image Load Error: {e}")
        else:
            print(f"⚠️ Image path not found: {image_path}")

    # 3. Call Gemini
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents, # Now supports [Text, Image]
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.3,
                max_output_tokens=1000,
            ),
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Nexus Error: {str(e)}"

def generate_marketing_insight(channel_data, reasoning):
    # (Kept unchanged for brevity)
    client = get_client()
    roi = channel_data.get('mmm_roi', 0)
 
    prompt = f"Explain the strategic role of {channel_data['channel']} (ROI: {roi:.2f}) in 1 sentence."
    try:
        response = client.models.generate_content(
            model=MODEL_ID, contents=prompt, config=types.GenerateContentConfig(temperature=0.1)
        )
        return response.text.strip()
    except:
        return f"ROI: {roi:.2f}"