import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
import requests

# Load environment variables from .env file
load_dotenv()

# Get Gemini API key from environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize OpenAI provider with Gemini API settings
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)

# Configure the language model
model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=provider)

# Function Tool
@function_tool
def get_shahiali_info():
    response = requests.get("https://api.github.com/users/shahidali54")
    result = response.json()
    return result


# Create an greeting agent with instructions, and model
greeting_agent = Agent(
    name="Greeting Agent",
    instructions="""
    You are a greeting agent. Your task is to greet the user warmly and make them feel welcome.
    """,
    model=model,
)


# Agent 2: Summarizer Agent
summrizer_agent = Agent(
    name="Summarizer Agent",
    instructions="""
    You are a summarizer agent. Your task is to read the provided text and generate a concise summary that captures the main points.
    """,
    model=model,
)


# Agent 3: Shahid Ali Info Agent
shahidali_info_agent = Agent(
    name="Shahid Ali Info Agent",
    instructions="""
    You are a helpful agent that retrieves information about Shahid Ali.
    Use the function tool get_shahiali_info to fetch the information.
    """,
    tools=[get_shahiali_info],
    model=model,
)


# Main Coordinator Agent
coordinator_agent = Agent(
    name="Coordinator Agent",
    instructions="""
    You are a coordinator agent. Your task is to manage the workflow of other agents and ensure that tasks are completed efficiently.
    You can assign tasks to other agents and collect their results.
    """,
    handoffs=[greeting_agent, summrizer_agent, shahidali_info_agent],
    model=model,
)


# Get user input from the terminal
prompt_value = input("Enter a prompt for the coordinator agent: ")

# Run the agent with user input and get result
agent_result = Runner.run_sync(coordinator_agent, prompt_value)

# Print the result
print(f"\nAgent Result: {agent_result.final_output}")