
# 🧠 Multi-Agent Workflow using Gemini 2.0 Flash

This project demonstrates a multi-agent architecture powered by Google's Gemini 2.0 Flash model, designed to handle various conversational tasks using specialized agents coordinated via a central controller.

## 🚀 Features

- 🤖 **Greeting Agent**: Welcomes the user with a friendly message.
- 📄 **Summarizer Agent**: Takes a block of text and returns a concise summary.
- 🔍 **Shahid Ali Info Agent**: Fetches GitHub profile details using a tool.
- 🧩 **Coordinator Agent**: Manages workflow and delegates tasks to the right agents.

## 🧠 Agentic AI Pillars Coverage

This project adopts a modular multi-agent design inspired by the **4 Pillars of Agentic AI**. Here's how each pillar is addressed:

| Pillar         | Status           | Description                                                                 |
|----------------|------------------|-----------------------------------------------------------------------------|
| ✅ Agent        | Implemented      | Four distinct agents created for greeting, summarizing, info retrieval, and coordination. |
| ✅ Handoff      | Implemented      | Tasks are delegated among agents using the `handoffs` parameter in the coordinator agent. |
| ❌ Guardrails   | Not Implemented  | No content filtering or safety constraints are applied to agent outputs. |
| ❌ Tracing      | Not Implemented  | Tracing requires OpenAI’s paid API or custom logging. Currently not included. |

> ⚠️ Future versions may add **Guardrails** via validation layers and **Tracing** using third-party tools or OpenAI’s paid services.

## 🔧 Technologies Used

| Technology       | Purpose                                          |
|------------------|--------------------------------------------------|
| Python           | Programming language                             |
| [Gemini API](https://ai.google.dev/) | Language model (via OpenAI-compatible API)    |
| `dotenv`         | Secure API key management                        |
| `requests`       | For GitHub API call                              |
| Custom SDK Layer | Agent, Runner, and function tools abstraction    |

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/shahidali54/openai-agent-sdk.git
cd openai-agent-sdk
```

### 2. Create `.env` File

```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python main.py
```

### 5. Sample Prompt

```
Please summarize this text: "Generative AI is transforming the future of work..."
```

## 🛠️ Agent Design

Each agent is designed using a custom `Agent` class. Tasks are modularized, and the `Coordinator Agent` handles all communication and task delegation. Tools like `function_tool` allow external APIs to be integrated seamlessly.

## 🔐 Security

All API keys are loaded via `.env` using `python-dotenv` to ensure no secrets are exposed in code.

## 👨‍💻 Author

-  **Shahid Ali**
- A passionate full-stack developer and learner at **PIAIC** & **GIAIC**
  Feel free to connect or reach out for collaboration!

---


**Happy Coding!** 😎 