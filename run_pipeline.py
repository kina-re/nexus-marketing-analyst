import os
import sys
from dotenv import load_dotenv

# 1. Load environment variables before anything else
load_dotenv()

# 2. Add 'src' to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.main import main

if __name__ == "__main__":
    print("\n🚀 INITIALIZING NEXUS PIPELINE...")
    
    # Safety Check for API Key
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("❌ Error: GEMINI_API_KEY is missing from your .env file.")
        sys.exit(1)
    else:
        print(f"✅ Key Loaded: {key[:5]}***")
    
    # Start the Main Application
    main()
