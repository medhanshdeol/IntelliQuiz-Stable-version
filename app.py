import os
import json
from fastapi import FastAPI
from google import genai
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Ensure these names match your ml_algorithms.py exactly
from ml_algorithms import get_info_gain, get_gain_ratio, NaiveBayesScratch

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- QUIZ ENDPOINT (10 Questions) ---
@app.get("/generate-quiz")
async def generate_quiz(topic: str):
    current_model = "gemini-2.5-flash" 
    prompt = f"""
    Create a 10-question MCQ quiz about {topic}. 
    Return ONLY a valid JSON list of objects.
    Format: [{{ "question": "...", "options": ["a", "b", "c", "d"], "answer": "..." }}]
    """
    try:
        response = client.models.generate_content(
            model=current_model,
            contents=prompt,
            config={'response_mime_type': 'application/json'}
        )
        return response.parsed if response.parsed else json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

# --- DYNAMIC ML ENDPOINT ---
@app.get("/run-ml")
async def run_ml(topic: str):
    current_model = "gemini-2.5-flash"
    
    dataset_prompt = f"""
    Generate a synthetic dataset for the topic '{topic}' with 8 rows and 4 columns.
    First 3 columns: Binary features (1 or 0). 
    Last column: A category label related to the topic.
    Return ONLY as a JSON list of lists.
    Example: [[1,0,1,"Success"], [0,1,1,"Failure"]]
    """
    
    try:
        response = client.models.generate_content(
            model=current_model, contents=dataset_prompt,
            config={'response_mime_type': 'application/json'}
        )
        dynamic_data = response.parsed if response.parsed else json.loads(response.text)
        
        # Run your from-scratch math on the AI's data
        labels = list(set(row[-1] for row in dynamic_data))
        ig = get_info_gain(dynamic_data, 0)
        gr = get_gain_ratio(dynamic_data, 0)
        
        nb = NaiveBayesScratch()
        nb.train(dynamic_data, labels)
        prediction = nb.predict([1, 0, 1]) 
        
        return {
            "topic_dataset": f"AI-Generated {topic} Data",
            "results": {
                "id3_info_gain": round(ig, 4),
                "c45_gain_ratio": round(gr, 4),
                "naive_bayes_prediction": f"Predicted Label: {prediction}"
            },
            "data_preview": dynamic_data[:3]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)