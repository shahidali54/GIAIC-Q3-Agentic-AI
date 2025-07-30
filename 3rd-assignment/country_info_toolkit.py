from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from dotenv import load_dotenv
import os

set_tracing_disabled(disabled=True)
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")


client=AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


capital_info_agent = Agent(
    name="Capital Info Agent",
    instructions="You are a helpful assistant that receives a country name as input and returns ONLY the capital city of that country. Respond with only the capital name.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
)

language_info_agent = Agent(
    name="Language Info Agent",
    instructions="You are a helpful assistant that receives a country name as input and returns ONLY the primary language spoken in that country. Respond with only the language name.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
)

population_info_agent = Agent(
    name="Population Info Agent",
    instructions="You are a helpful assistant that receives a country name as input and returns ONLY the population of that country. Respond with only the number.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
)

country_info_orchestrator = Agent(
    name="Country Info Orchestrator",
    instructions="You are a helpful assistant that provides information about a country. You will be given a country name and you will need to provide information about the country. You will need to use the CapitalInfoAgent, LanguageInfoAgent, and PopulationInfoAgent to provide the information.",
    tools=[capital_info_agent.as_tool(
        tool_name="capital_info",
        tool_description="Provide information about the capital of a country",
    ), language_info_agent.as_tool(
        tool_name="language_info",
        tool_description="Provide information about the language of a country",
    ), population_info_agent.as_tool(
        tool_name="population_info",
        tool_description="Provide information about the population of a country",
    )],
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
)

user_input = input("Enter a country name: ")

result = Runner.run_sync(
    country_info_orchestrator,
    input=user_input,    
)
print(result.final_output)