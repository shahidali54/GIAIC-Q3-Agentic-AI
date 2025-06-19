import os
from dotenv import load_dotenv
import openai

def main() -> None:
    # Load .env file
    load_dotenv()

    # Set base URL for OpenRouter
    openai.api_base = "https://openrouter.ai/api/v1"

    # Load API key
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in .env file.")
        return

    openai.api_key = api_key

    try:
        response = openai.ChatCompletion.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[
                {"role": "user", "content": "Write a short poem about the stars."}
            ]
        )

        print("✅ AI Response:\n")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"❌ Error calling OpenRouter API: {e}") 