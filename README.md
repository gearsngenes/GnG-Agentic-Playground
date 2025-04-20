# Welcome to Agentic Playground
## Synopsis and Motivation
This repository implements a Flask-based application designed to simulate **multi-agent conversations** with **AutoGen** assistants. Unlike traditional a traditional chatbot, the agents here are built for **collaborative task completion**. Users initiate a task, and AutoGen-managed agents collaborate to resolve it, step by step.

You can:
- Define and manage multiple agents with specific reasoning roles
- Assign agents to a **specialist domain**, **maker-checker writing process**, or **chain-of-thought reasoning**
- Upload knowledge bases (documents/images), which specialist agents reference using RAG-style context retrieval
- Interact with agents via a turn-based interface, where you as the user can interject at any time

You can use this playground to simulate workflows and performed advanced chains of tasks with minimal specifications based on the agents you provide. You can effectively create your own mini-team. For instance, if you are a teacher, you might want an SME on the textbooks that you are using to be able to retrieve any information to answer a student's question, a lesson planner to effectively turn that content into modular lessons, and a TA assistant to review the content to anticipate what types of questions students might ask.

## The Tech Stack
The backend is built with **Flask** as the primary middleware/frontend of the application, but also incorporates:

- **AutoGen**: A multi-agent orchestration framework built on top of OpenAI’s APIs. It allows agents to exchange messages and reason through tasks collaboratively. This is the backbone of all conversational workflows in this playground. Feel free to read more about my research on AutoGen in my LinkedIn [here](https://www.linkedin.com/pulse/exploring-ai-orchestrators-sidharta-vadaparty-bklre/?trackingId=6BeVDHSoysX4%2BxCC3q%2B60g%3D%3D).
- **OpenAI**: All language models used in agent responses and embeddings are provided by OpenAI's GPT APIs (e.g., `gpt-4o`, `gpt-4o-mini`, `gpt-4`).
- **Pinecone**: A vector database used for indexing and retrieving document/image content. Text chunks and image descriptions are first embedded using OpenAI and then the vectors along with the text metadata are stored on the database. Various "topics" are created, with each topic corresponding to a Pinecone index and a potential knowledge base we can assign to our agents.
- **JavaScript + HTML (vanilla)**: Simple frontend interfaces allow users to chat, create/manage assistants, and upload/query files.

## Files of Note

### app.py
Main Flask application entrypoint. It:
- Hosts all routes for chat, file uploads, and agent creation
- Manages AutoGen group chat execution
- Injects retrieved context for specialist agents

Run this to start the playground:
```
python app.py
```

### agents.py
Handles:
- Defining the user agent
- Loading all assistant agents from `_agent_configs.json`
- Storing agent metadata (e.g., type, topic)
- Injecting RAG context when needed

### pinecone_utils.py
Manages all interactions with Pinecone:
- Creating/deleting indexes
- Upserting vectors (text and image alt-text)
- Tracking whether a file has been embedded
- Handling descriptions via a central Table-of-Contents index

### helpers.py
Provides utility functions:
- Text and image extraction from uploaded files
- Chunking text for embedding
- Generating image descriptions via GPT-4
- Embedding generation

### _agent_configs.json
Persistent store for assistant definitions. Agents created through the frontend are written here. This includes:
- System messages
- Pattern type (`general`, `specialist`, `maker`, `checker`, `chain_of_thought`)
- model (`gpt-4o`, `gpt-4o-mini`, `gpt-4`)
- Linked topic for specialist agents for PineCone RAG

### Frontend HTML Files
- `index.html`: Main chat interface where users start tasks and interject in a chatroom.
- `manage_assistants.html`: Update or delete assistant agents as well as edit their system messages and descriptions if they need to be changed.
- `create_assistants.html`: Create agents with one of several patterns
- `manage_vectorstore.html`: Create knowledge base topics, upload files, and embed content into Pinecone

## Supported Agentic Patterns

- **General**: Basic assistant with custom name, model, system message, and description. For free form tasks.
- **Specialist (SME)**: Agent linked to a knowledge base topic; uses RAG to retrieve info relevant to the current conversation.
- **Maker-Checker**: Two-agent writing pipeline. One generates content (Maker), the other critiques it (Checker).
- **Chain of Thought**: Single agent that follows a structured step-by-step reasoning process before outputting an answer.

Each agent is appended with a control tag (`<status:CONTINUE>`, `<status:REQUEST>`, `<status:TERMINATE>`) to allow the system to manage conversation flow, where CONTINUE allows the conversation to go forward without needing user input, REQUEST explicitly asking for user input, and TERMINATE indicating the user's goal has been sufficiently addressed. 

## Setup Instructions

1. Set up your OpenAI and Pinecone accounts.
2. In the project root, create a `.env` file like so:
```
OPENAI_API_KEY="your-openai-key"
PINECONE_API_KEY="your-pinecone-key"
UPLOAD_ROOT="your-uploads-directory"
```

3. Run the server:
```
python app.py
```

4. Navigate to:
```
http://127.0.0.1:5000
```

## Docker Setup (To be built)