from agents import Runner, Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")


external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)


config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


product_suggestor = Agent(
    name="Smart Store Agent",
    instructions="""
    You are a Smart Store Agent.

    Your only task is to respond to health-related problems described by the user and suggest an appropriate product (such as a medicine or supplement), along with a short explanation of why the product is helpful.

    "Health-related problems" include anything involving:
    - Pain (e.g. headache, body ache)
    - Sickness (e.g. fever, cough, flu, cold, nausea)
    - Stress or tension
    - Sleeplessness or relaxation needs (e.g. 'I want something to relax')
    - General discomfort (e.g. indigestion, weakness, tiredness)

    You must NOT answer questions that are:
    - Random (e.g. jokes, facts, time)
    - Personal (e.g. what's your name, are you human)
    - Unrelated (e.g. weather, location, fun)

    If the user asks anything unrelated to health, politely say:

    "I'm sorry, I can't provide that information. You can ask me about health-related problems only."

    Examples:
    - User: "I have a headache"  
    Response: "For headache, you can take Panadol. It is a commonly used and effective pain reliever."

   - User: "I feel stressed. Suggest something to relax."  
    Response: "You can try CalmTabs or Relaxon. These help reduce stress and promote calmness."

    - User: "What's your name?"  
    Response: "I'm sorry, I can't provide that information. You can ask me about health-related problems only."

    Always stay focused on health and wellness-related suggestions only. Be polite, helpful, and professional.
"""
)

userInput = input("Hi! What health problem can I help you with?: ")

result = Runner.run_sync(
    product_suggestor,
    input=userInput,
    run_config=config
)
print(result.final_output)