from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled
from dotenv import load_dotenv
import asyncio
import os


set_tracing_disabled(disabled=True)

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

async def main():
    activity_suggester = Agent(
        name="Activity Suggester",
        instructions="""
        You are an activity suggestion agent. Based on the given mood, suggest an appropriate activity.
        If the mood is "sad" or "stressed", provide a comforting activity suggestion.
        If the mood is "happy", suggest a fun activity.
        Return the suggestion as a string.
        """,
        model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)
    )

    mood_analyzer = Agent(
        name="Mood Analyzer",
        instructions="""
        You are a mood analysis agent. Analyze the user's message and determine their mood.
        Return only one of these moods as a string: "happy", "sad", or "stressed".
        """,
        model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    )

    userInput = input("How are you feeling today? (Describe your mood or situation): ")
    result = await Runner.run(
        mood_analyzer,
        input=userInput
    )
    print(result.final_output)

    moodresult = await Runner.run(
        activity_suggester,
        input=result.final_output
    )
    print(moodresult.final_output)
asyncio.run(main())




