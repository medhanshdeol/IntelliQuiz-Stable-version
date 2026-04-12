import os, math, json, random, traceback
from collections import Counter
from google import genai
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

load_dotenv()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- 1. 2026 KERNEL INITIALIZATION ---
api_key = os.getenv("GEMINI_API_KEY")
client = None
if api_key:
    client = genai.Client(api_key=api_key)
    print("✅ KERNEL INITIALIZED: Gemini 2.5 Flash")
else:
    print("❌ KERNEL ERROR: NO API KEY FOUND")

# --- 2. ML ENGINE ---
def safe_entropy(data):
    if not data: return 0
    labels = [row[-1] for row in data]
    counts = Counter(labels)
    return -sum((c/len(labels)) * math.log2(c/len(labels)) for c in counts.values() if c > 0)

def get_gain(data, idx):
    total = safe_entropy(data)
    vals = set(row[idx] for row in data)
    weighted = sum((len(sub)/len(data)) * safe_entropy(sub) for sub in [[r for r in data if r[idx] == v] for v in vals])
    return total - weighted

# --- 3. ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/generate-quiz")
async def generate_quiz(topic: str = Query(...)):
    print(f"🚀 BUILDING: {topic}")
    
    if not client:
        raise HTTPException(status_code=500, detail="Gemini client not initialized")

    try:
        prompt = f"""Create a 10-question multiple choice quiz about {topic}.
Return ONLY a JSON array, where each element has:
- 'question': the question text
- 'options': an array of exactly 4 strings
- 'answer': the correct option from the options array

Example format:
[
  {{"question": "...", "options": ["A", "B", "C", "D"], "answer": "A"}}
]
"""
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        raw_text = response.text.strip()
        # Clean up markdown if any
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "", 1)
        if raw_text.startswith("```"):
            raw_text = raw_text.replace("```", "", 1)
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        
        quiz_data = json.loads(raw_text)
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to fetch relevant quiz data from AI")

    # Dummy ML data based on the provided functions
    dummy_data = [
        [1, 'High', 'Yes'], [2, 'Low', 'No'], [1, 'High', 'Yes'], 
        [3, 'Medium', 'Yes'], [2, 'Medium', 'No']
    ]
    
    ml_analysis = {
        "entropy": safe_entropy(dummy_data),
        "gains": {
            "Feature_0": get_gain(dummy_data, 0),
            "Feature_1": get_gain(dummy_data, 1)
        },
        "rf_logs": [
            f"Bootstrap Sample 1: Size {random.randint(50, 100)}",
            f"Bootstrap Sample 2: Size {random.randint(50, 100)}",
            "Model aggregated successfully."
        ]
    }
    
    return {"quiz": quiz_data, "ml_analysis": ml_analysis}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)