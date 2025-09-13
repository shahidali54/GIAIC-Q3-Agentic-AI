from agents import (
    Agent,
    Runner,
    RunConfig,
    OpenAIChatCompletionsModel,
    function_tool,
    RunContextWrapper,
    GuardrailFunctionOutput,
    output_guardrail,
    TResponseInputItem,
    OutputGuardrailTripwireTriggered,
)
from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(model=model, model_provider=external_client, tracing_disabled=True)


class UserContext(BaseModel):
    name: str
    is_premium_user: bool
    issue_type: str


class OffensiveOutput(BaseModel):
    contains_offensive: bool


@function_tool
async def refund(ctx: RunContextWrapper) -> str:
    user_name = ctx.context.name
    return (
        f"Hello {user_name},\n\n"
        "We've reviewed your account and your refund has been **successfully initiated** ✅. "
        "You can expect the amount to be credited back to your original payment method within **3-5 business days**. "
        "If you have any further questions, feel free to reach out — we're here to help!"
    )

refund.is_enabled = lambda ctx, agent: ctx.context.is_premium_user


@function_tool
async def restart_service(ctx: RunContextWrapper) -> str:
    user_name = ctx.context.name
    return (
        f"Hi {user_name},\n\n"
        "Your service restart request has been received and is now **in progress** 🔄. "
        "Please allow a few moments for the changes to take effect. "
        "You will receive a confirmation once everything is back online."
    )

restart_service.is_enabled = lambda ctx, agent: ctx.context.issue_type == "technical"


@function_tool
async def general_info(ctx: RunContextWrapper) -> str:
    user_name = ctx.context.name
    return (
        f"Hello {user_name},\n\n"
        "Here's some quick info about our services:\n"
        "- **24/7 Customer Support** 🕑\n"
        "- **Fast & Secure Transactions** 🔐\n"
        "- **Premium Members** enjoy priority handling\n\n"
        "If you'd like details about a specific service, let me know!"
    )



billing_agent = Agent(
    name="BillingAgent",
    instructions="""
You are the **Billing Support Specialist** for our customer service team.  

🎯 **Goal:** Handle all billing-related queries with a professional and reassuring tone.  
💡 **Style:** Polite, empathetic, and customer-focused. Always thank the user for their patience.  

When a **premium user** requests a refund:
- Always **call the `refund` tool** to process it.
- Clearly confirm the action, provide an estimated timeline, and invite further questions.
- Never give a vague answer — be specific about next steps.

❌ **Do NOT** directly write the refund details yourself.  
✅ Always use the refund tool to generate the response.
""",
    tools=[refund],
)


technical_agent = Agent(
    name="TechnicalAgent",
    instructions="""
You are the **Technical Support Engineer** responsible for helping users with technical problems.  

🎯 **Goal:** Resolve technical issues quickly while keeping the customer informed.  
💡 **Style:** Friendly yet professional. Use plain language to explain processes.  

When a user with `issue_type` set to `"technical"` requests a **service restart**:
- Always **call the `restart_service` tool**.
- Acknowledge the issue, confirm the restart process, and explain expected time until resolution.
- Give reassurance that the service will be back shortly.

❌ **Do NOT** attempt to troubleshoot within the message.  
✅ Always use the restart_service tool for the final response.
""",
    tools=[restart_service],
)


general_agent = Agent(
    name="GeneralAgent",
    instructions="""
You are the **Customer Care Advisor** handling general queries.  

🎯 **Goal:** Provide clear, friendly, and accurate information about our services.  
💡 **Style:** Helpful, approachable, and concise.  

When the query is **general**:
- Always **call the `general_info` tool** to provide details.
- Present the information in a structured format (bullet points or numbered lists).
- Offer the option to connect the user to a specialist for deeper help.

❌ **Do NOT** give partial answers.  
✅ Always use the general_info tool for the final output.
""",
    tools=[general_info],
)



guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="""
You are a guardrail agent. 
Check if the final output contains any offensive or rude words such as "stupid", "idiot", "hate", "useless".
If you find one, return contains_offensive: True, otherwise return contains_offensive: False.
""",
    output_type=OffensiveOutput,
)


@output_guardrail
async def NoOffensiveLanguageGuardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(
        guardrail_agent, input, run_config=config, context=ctx.context
    )

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.contains_offensive,
    )


triage_agent = Agent(
    name="TriageAgent",
    instructions="""
You are the **Query Routing Specialist** for our support system.  

🎯 **Goal:** Identify the type of user query and send it to the correct specialized agent.  
💡 **Style:** Professional, short, and clear.

Routing rules:
- If the issue is related to **billing** (refunds, payment issues) → hand off to **BillingAgent**.
- If the issue is **technical** (errors, service restarts) → hand off to **TechnicalAgent**.
- If the query is **general** (about features, working hours, etc.) → hand off to **GeneralAgent**.

If the query is **out of scope** (e.g., unrelated topics), simply respond with:
> "This query is out of scope."

❌ Never attempt to solve the query yourself.  
✅ Always hand off to the most relevant agent.
""",
    handoffs=[billing_agent, technical_agent, general_agent],
    output_guardrails=[NoOffensiveLanguageGuardrail],
)



async def main():
    print("\nWelcome to the Support Agent System")
    name = input("Enter your name: ")
    is_premium_input = input("Are you a premium user? (yes/no): ").strip().lower()
    issue_type = (
        input("What type of issue are you facing? (billing/technical/general): ")
        .strip()
        .lower()
    )

    is_premium = is_premium_input == "yes"
    context = UserContext(name=name, is_premium_user=is_premium, issue_type=issue_type)

    user_query = input("\nPlease describe your issue: ")
    print("\nRouting your query...\n")

    try:
        result = await Runner.run(
            triage_agent, user_query, run_config=config, context=context
        )
        print("\nFinal Output:")
        print(result.final_output)

    except OutputGuardrailTripwireTriggered:
        print("\nTripwire triggered. Offensive language is not allowed.")


if __name__ == "__main__":
    asyncio.run(main())