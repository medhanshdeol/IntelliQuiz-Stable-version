import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Modern client initialization
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_quiz(topic):
    print(f"\n🧠 Attempting to generate quiz for: {topic}...")
    
    # We'll use Gemini 2.5 Flash, the 2026 free-tier stable model
    current_model = "gemini-2.5-flash"
    
    try:
        response = client.models.generate_content(
            model=current_model,
            contents=f"Create a 3-question MCQ quiz about {topic}."
        )
        print("-" * 30)
        print(response.text)
        print("-" * 30)
        
    except Exception as e:
        print(f"❌ Error with {current_model}: {e}")
        print("\n🔍 Scanning for available models in your region...")
        try:
            # This list will show us EXACTLY what you can use
            for m in client.models.list():
                if 'generateContent' in m.supported_methods:
                    print(f"✅ Available: {m.name}")
        except:
            print("Failed to list models. Check your API key.")

if __name__ == "__main__":
    topic = input("Enter quiz topic: ")
    generate_quiz(topic)