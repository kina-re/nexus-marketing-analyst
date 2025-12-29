import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

def test_key():
    if not API_KEY:
        print("❌ Error: GEMINI_API_KEY not found.")
        return

    # UPDATED: Using the generic alias found in your list
    model_name = "gemini-flash-latest"
    
    print(f"🔑 Testing Key with model: {model_name}...")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
    
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{"text": "Reply with 'API Connection Successful!'"}]
        }]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            print("\n✅ SUCCESS! Your API Key is working.")
            print(f"🤖 AI Reply: {response.json()['candidates'][0]['content']['parts'][0]['text']}")
        else:
            print(f"\n❌ Failed. Status: {response.status_code}")
            print(f"Error Message: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_key()