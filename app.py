import os
import json
from fastapi import FastAPI
from google import genai
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# 1. Load the environment (Gemini API Key)
load_dotenv()

# 2. Initialize FastAPI
app = FastAPI()

# 3. Security (Allows your future frontend to talk to this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Setup Gemini Client
# Uses the key from your .env file
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.get("/generate-quiz")
async def generate_quiz(topic: str):
    # Using the 2026 stable free-tier model
    current_model = "gemini-2.5-flash" 
    
    # We use double curly braces {{ }} so Python doesn't confuse them with variables
    prompt = f"""
    Create a 3-question MCQ quiz about {topic}. 
    Return the response ONLY as a valid JSON list.
    Example format:
    [
      {{
        "question": "What is the brain of the computer?",
        "options": ["RAM", "CPU", "Hard Drive", "GPU"],
        "answer": "CPU"
      }}
    ]
    """
    
    try:
        response = client.models.generate_content(
            model=current_model,
            contents=prompt,
            config={'response_mime_type': 'application/json'}
        )
        
        # Priority 1: Use the SDK's built-in parser
        if response.parsed:
            return response.parsed
        
        # Priority 2: Manual cleanup if the AI includes markdown backticks
        if response.text:
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
            
        return {"error": "AI returned empty content. Try a different topic."}

    except Exception as e:
        print(f"Server Error: {e}")
        return {"error": str(e)}

# 5. Start the Server
if __name__ == "__main__":
    import uvicorn
    print("🚀 IntelliQuiz Backend is starting...")
    uvicorn.run(app, host="0.0.0.0", port=8000)