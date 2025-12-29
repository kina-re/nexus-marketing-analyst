import os
import sys
import shutil
import pandas as pd
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# --- RESOLVE SEARCH PATHS (Fixes the Import Errors) ---
# 1. Get the absolute path to the 'src' directory
BASE_DIR = Path(__file__).resolve().parent
SRC_PATH = str(BASE_DIR / "src")

# 2. Add 'src' to sys.path so we can import modules inside it directly
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# --- NOW IMPORTS WILL WORK ---
try:
    # Importing the core pipeline function used in run_pipeline.py
    from src.main import run_analysis_pipeline 
    import src.llm_analyst as nexus             
    from src.strategy import generate_action_plan 
except ImportError as e:
    print(f"❌ Startup Error: {e}")
    raise e

app = FastAPI(title="Nexus Strategic Engine")

# --- SETUP ---
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = {"strategic_data": None}

# ---------------- ROUTES ----------------

@app.post("/upload-data")
async def upload_and_process(
    attribution_file: UploadFile = File(...), 
    mmm_file: UploadFile = File(...)          
):
    try:
        # 1. Save Files locally using pipeline standard names
        with open(DATA_DIR / "ga_2.csv", "wb") as f:
            shutil.copyfileobj(attribution_file.file, f)
        with open(DATA_DIR / "mmm_data_2016_2017.csv", "wb") as f:
            shutil.copyfileobj(mmm_file.file, f)

        # 2. Trigger the CLI Pipeline logic
        results = run_analysis_pipeline() 

        # 3. Apply Strategy Layer & Cache results
        recommendations = generate_action_plan(results['prior_df'])
        
        forest = [{
            "channel": r['channel'],
            "weight": round(r['attr_weight'], 3),
            "lower": round(max(0, r['attr_weight'] - 0.05), 3),
            "upper": round(r['attr_weight'] + 0.05, 3),
            "confidence": "High" if r.get('confidence', 0.8) > 0.7 else "Medium"
        } for _, r in results['prior_df'].iterrows()]

        cache["strategic_data"] = {
            "forest": forest,
            "recommendations": recommendations,
            "funnel": [
                {"stage": "Awareness", "value": 100, "label": "Path Hits"},
                {"stage": "Conversion", "value": 15, "label": "Pipeline Wins"}
            ]
        }
        
        return {"status": "success"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/get-executive-data")
async def get_exec():
    return cache.get("strategic_data", {"forest": [], "funnel": [], "recommendations": []})

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "")
    context = "Nexus Strategy Dashboard active."
    if cache["strategic_data"]:
        context += f" Top Action: {cache['strategic_data']['recommendations'][0]['action']}"
    
    answer = nexus.chat_with_data(question, context)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
 
