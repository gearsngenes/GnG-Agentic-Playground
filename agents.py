from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager
import json
import os
from dotenv import load_dotenv
from pinecone_utils import vector_store_manager
from helpers import UPLOAD_FOLDER
from urllib.parse import quote

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_CONFIG_PATH = "_agent_configs.json"

config_list = [{"model": "gpt-4o", "api_key": OPENAI_API_KEY}]
agents_list = []
agent_metadata = {}

# === User agent
user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
    is_termination_msg=lambda msg: msg.get("content", "").strip().endswith("TERMINATE")
)
agents_list.append(user)
agent_metadata["user"] = {"pattern": "user"}

# === System prompt marker
STATUS_INSTRUCTION = (
    "At the end of every message, append one of the following markers:\n\n"
    "<status:CONTINUE>\n<status:REQUEST>\n<status:TERMINATE>\n\n"
    "Choose CONTINUE to pass control, REQUEST to ask the user for info,"
    " or TERMINATE if the user's goal is accomplished."
)

# === Load all assistant agents
def load_agents():
    # Create the config file if it doesn't exist
    if not os.path.exists(AGENT_CONFIG_PATH):
        with open(AGENT_CONFIG_PATH, "w") as f:
            json.dump([], f)
    with open(AGENT_CONFIG_PATH, "r") as f:
        config_data = json.load(f)

    for config in config_data:
        name = config.get("name")
        if name == "user":
            continue

        model = config.get("model", "gpt-4o")
        base_message = config.get("system_message", "").strip()
        description = config.get("description", "")
        pattern = config.get("pattern", "general")
        topic = config.get("topic")

        agent_metadata[name] = {"pattern": pattern}
        if pattern == "specialist" and topic:
            agent_metadata[name]["topic"] = topic

        full_message = base_message + "\n\n" + STATUS_INSTRUCTION

        agent = AssistantAgent(
            name=name,
            system_message=full_message,
            description=description,
            llm_config={"config_list": config_list}
        )
        agents_list.append(agent)

# === System prompt access/update
def get_agent_sysmsg_descr(agent_name):
    with open(AGENT_CONFIG_PATH, "r") as f:
        data = json.load(f)
    for agent in data:
        if agent.get("name") == agent_name:
            return agent.get("system_message", ""), agent.get("description","")
    return "<NO AGENT>", ""

def update_agent_system_message(agent_name, new_message, new_description=None):
    for agent in agents_list:
        if agent.name == agent_name:
            agent.update_system_message(new_message.strip() + "\n\n" + STATUS_INSTRUCTION)
            break
    else:
        return False

    with open(AGENT_CONFIG_PATH, "r") as f:
        data = json.load(f)
    updated = False
    for agent in data:
        if agent.get("name") == agent_name:
            agent["system_message"] = new_message
            if new_description is not None:
                agent["description"] = new_description
            updated = True
            break
    if updated:
        with open(AGENT_CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)
        return True
    return False

# === Chat creation
def create_chat():
    load_agents()
    group_chat = GroupChat(
        agents=agents_list,
        messages=[],
        speaker_selection_method="auto",
        allow_repeat_speaker=False,
        select_speaker_auto_llm_config={"config_list": config_list},
        select_speaker_message_template=(
            "You are coordinating a team of AI agents. Review the task and messages so far, "
            "and choose which agent should respond next.\n\nAgents: {agentlist}\n\nRoles:\n{roles}"
        )
    )
    manager = GroupChatManager(groupchat=group_chat)
    return group_chat, manager

def get_rag_contex(query, topic):
    chunks = vector_store_manager.query_at_index(topic, query, top_k=5)
    if chunks:
        content_chunks = [c.get('content','[No content]') for c in chunks]
        file_paths = [c.get("file_path").replace("\\", "/") for c in chunks]
        relative_paths = [p[len(f"{UPLOAD_FOLDER}"):] if p.startswith(f"{UPLOAD_FOLDER}/") else p for p in file_paths]
        chunk_types = [c.get("type", "text") for c in chunks]
        url_paths = [f"/{UPLOAD_FOLDER}{quote(rp)}" for rp in relative_paths]
        markdown_links = [f"[{os.path.basename(fp)}]({url})" for fp,url in zip(file_paths, url_paths)]
        image_links = []
        for _type, url in zip(chunk_types, url_paths):
            if _type == "image":
                image_links.append(f"![]{url}")
        context_text = "\n\n".join(f"- {context}\n\t-Source: {source}" for context, source in zip(content_chunks, markdown_links))
        retrieved_content = f"[RETRIEVED TEXTUAL CONTEXT]\n{context_text}\n\n[RETRIEVED IMAGE LINKS]{image_links}"
        return retrieved_content
    return "[No content retrieved]"

__all__ = [
    "agents_list",
    "user",
    "create_chat",
    "load_agents",
    "get_agent_system_message",
    "update_agent_system_message",
    "agent_metadata",
    "get_rag_contex"
]
