from litellm import completion
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set API keys from .env (optional - mostly not needed if litellm handles this)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

def openai():
    try:
        response = completion(
            model="openai/gpt-4o",
            messages=[{"content": "Hello, how are you?", "role": "user"}]
        )
        print("✅ OpenAI Response:", response['choices'][0]['message']['content'])
    except Exception as e:
        print("⚠️ OpenAI Error:", e)

def gemini():
    try:
        response = completion(
            model="gemini/gemini-1.5-flash",
            messages=[{"content": "Hello, how are you?", "role": "user"}]
        )
        print("✅ Gemini 1.5 Response:", response['choices'][0]['message']['content'])
    except Exception as e:
        print("⚠️ Gemini 1.5 Error:", e)

def gemini2():
    try:
        response = completion(
            model="gemini/gemini-2.0-flash-exp",
            messages=[{"content": "Hello, how are you?", "role": "user"}]
        )
        print("✅ Gemini 2.0 Response:", response['choices'][0]['message']['content'])
    except Exception as e:
        print("⚠️ Gemini 2.0 Error:", e) 