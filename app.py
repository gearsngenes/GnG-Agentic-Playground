from flask import Flask, request, jsonify, send_file, render_template
from agents import (
    user,
    create_chat,
    get_rag_contex,
    get_agent_sysmsg_descr,
    update_agent_system_message,
    agent_metadata,
    AGENT_CONFIG_PATH
)
import re
import datetime
import shutil


from pinecone_utils import vector_store_manager
from helpers import *

app = Flask(__name__)

conversation = []
group_chat = None
manager = None
sender = None
current_status = "CONTINUE"

def get_status_from_reply(text):
    match = re.search(r"<status:(CONTINUE|REQUEST|TERMINATE)>$", text.strip())
    return match.group(1) if match else "CONTINUE"

def strip_status_tag(text):
    return re.sub(r"<status:(CONTINUE|REQUEST|TERMINATE)>\s*$", "", text.strip(), flags=re.IGNORECASE).strip()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/manage")
def manage_assistants():
    return render_template("manage_assistants.html")

@app.route("/vectorstore")
def vectorstore_page():
    return render_template("manage_vectorstore.html")

@app.route("/creator")
def create_assistants():
    return render_template("create_assistants.html")

@app.route("/start", methods=["POST"])
def start_conversation():
    global conversation, group_chat, manager, sender, current_status
    current_status = "CONTINUE"
    task = request.json.get("task", "").strip()

    if not task:
        return jsonify({"error": "No task provided"}), 400

    # Reset session
    group_chat, manager = create_chat()
    conversation = []

    # Initialize with user message
    user_message = {"role" : "user", "speaker" : "user", "content" : task}
    group_chat.append(user_message, user)
    conversation.append(user_message)

    # Pre-send message to all other agents
    for agent in group_chat.agents:
        if agent != user:
            manager.send(user_message, agent, request_reply=False)

    sender = user
    return jsonify({"messages": conversation, "done": False})


@app.route("/step", methods=["POST"])
def step():
    global conversation, sender, current_status, group_chat, manager

    user_input = request.json.get("message", "").strip()

    if current_status == "REQUEST" and not user_input:
        return jsonify({"messages": conversation, "done": False, "require_user": True})

    # === User interjection
    if user_input:
        user_msg = {"role": "user", "speaker":"user","content": user_input}
        group_chat.append(user_msg, user)
        conversation.append(user_msg)
        sender = user
        for agent in group_chat.agents:
            if agent != sender:
                manager.send(user_msg, agent, request_reply=False)

    receiver = group_chat.select_speaker(sender, manager)

    # === Specialist logic: inject retrieved context
    receiver_meta = agent_metadata.get(receiver.name, {})
    if receiver_meta.get("pattern") == "specialist":
        topic = receiver_meta.get("topic")
        query = (
            f"Please read the following conversation between the user and other agents "
            f"that are trying to help decompose and complete the task set by the user. "
            f"Use this conversation as context to determine what information to retrieve."
        )
        #query = next((m["content"] for m in reversed(group_chat.messages) if m["role"] == "user"), "")
        retrieved_content = get_rag_contex(query=query, topic=topic)
        print(retrieved_content)
        group_chat.append(
            {"role": "system", "speaker": "system", "content": retrieved_content},
            receiver
        )

    # === Generate response
    reply = receiver.generate_reply(sender=manager, messages=group_chat.messages)

    reply_text = ""
    if isinstance(reply, str):
        reply_text = reply
    elif isinstance(reply, dict):
        reply_text = reply.get("content", "")

    if not reply_text:
        return jsonify({"messages": conversation, "done": False})

    current_status = get_status_from_reply(reply_text)
    clean_reply = strip_status_tag(reply_text)

    message = {
        "role": "assistant",
        "content": clean_reply,
        "speaker": receiver.name
    }
    group_chat.append(message, receiver)
    conversation.append(message)
    sender = receiver

    for agent in group_chat.agents:
        if agent != sender:
            manager.send(message, agent, request_reply=False)

    if current_status == "TERMINATE":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{timestamp}.txt"
        os.makedirs("conversations", exist_ok=True)
        filepath = os.path.join("conversations", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for msg in conversation:
                f.write(f"{msg['role']}: {msg['content']}\n")
        return jsonify({"messages": conversation, "done": True, "auto_finish": True, "file": filepath})

    return jsonify({"messages": conversation, "done": False})

from flask import send_from_directory

@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory("uploads", filename)

@app.route("/finish", methods=["POST"])
def finish():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_{timestamp}.txt"
    os.makedirs("conversations", exist_ok=True)
    filepath = os.path.join("conversations", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        for msg in conversation:
            f.write(f"{msg['role']}: {msg['content']}\n")
    return jsonify({"file": filepath})


@app.route("/download")
def download():
    file = request.args.get("file")
    if not file or not os.path.exists(file):
        return "File not found", 404
    return send_file(file, as_attachment=True)


#=== manage_assistants.html ===
@app.route("/agent_load", methods=["POST"])
def agent_load():
    name = request.json.get("agent_name")
    sys_msg, descr = get_agent_sysmsg_descr(name)

    if sys_msg != "<NO AGENT>":
       return jsonify({
            "system_message": sys_msg,
            "description": descr
       })
    return jsonify({"error": "Agent not found"}), 404



@app.route("/agent_save", methods=["POST"])
def agent_save():
    name = request.json.get("agent_name")
    message = request.json.get("system_message", "")
    description = request.json.get("description")

    updated = update_agent_system_message(name, message, description)
    if updated:
        return jsonify({"success": updated})
    else:
        return jsonify({"error": "Agent not found"}), 404



@app.route("/agent_delete", methods=["POST"])
def delete_agent():
    name = request.json.get("agent_name")
    if not name or name == "user":
        return jsonify({"error": "Invalid agent name"}), 400

    try:
        with open(AGENT_CONFIG_PATH, "r") as f:
            data = json.load(f)

        # Check if this is part of a maker-checker pair
        agent_to_delete = next((a for a in data if a.get("name") == name), None)
        if not agent_to_delete:
            return jsonify({"error": "Agent not found"}), 404

        to_delete = [name]

        if agent_to_delete.get("pattern") in ["maker", "checker"]:
            counterpart = agent_to_delete.get("counterpart")
            if counterpart:
                to_delete.append(counterpart)

        # Filter out both if needed
        new_data = [a for a in data if a.get("name") not in to_delete]

        with open(AGENT_CONFIG_PATH, "w") as f:
            json.dump(new_data, f, indent=2)

        return jsonify({
            "message": f"Agent(s) deleted: {', '.join(to_delete)}",
            "agents": [a["name"] for a in new_data]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/agents", methods=["GET"])
def list_agents():
    with open(AGENT_CONFIG_PATH, "r") as f:
        data = json.load(f)
    return jsonify([a["name"] for a in data])


@app.route("/agent_create", methods=["POST"])
def create_agent():
    data = request.json
    pattern = data.get("pattern")

    if pattern == "specialist":
        name = data.get("name", "").strip()
        if not name:
            return jsonify({"error": "Missing required agent name."}), 400
        model = data.get("model", "gpt-4o")
        # description = data.get("description", "").strip()
        system_message, description = create_specialist_prompt(data)
        topic = data.get("topic")
        return write_agent_to_config(name, model, system_message, description, pattern="specialist", topic=topic)

    elif pattern == "maker_checker":
        return create_maker_checker_agents(data)

    elif pattern == "chain_of_thought":
        return create_chain_of_thought_agent(data)

    elif pattern == "general":
        name = data.get("name", "").strip()
        if not name:
            return jsonify({"error": "Missing required agent name."}), 400
        model = data.get("model", "gpt-4o")
        system_message = data.get("system_message", "").strip()
        description = data.get("description", "").strip()
        return write_agent_to_config(name, model, system_message, description, pattern="general")

    else:
        return jsonify({"error": f"Unsupported pattern: {pattern}"}), 400

def create_chain_of_thought_agent(data):
    name = data.get("name", "").strip()
    model = data.get("model", "gpt-4o")
    role = data.get("role", "").strip()
    thought_chain = data.get("thought_chain", [])

    if not name or not role or not isinstance(thought_chain, list) or not all(isinstance(step, str) and step.strip() for step in thought_chain):
        return jsonify({"error": "Missing or invalid Chain-of-Thought data."}), 400

    steps_formatted = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(thought_chain)])
    system_message = (
        f"You are a reasoning assistant who solves tasks using a logical, structured thought process.\n\n"
        f"**Scope of Reasoning:** {role}\n\n"
        f"You must always follow these steps in order:\n{steps_formatted}\n\n"
        f"Only provide a conclusion after completing all reasoning steps."
    )

    try:
        with open(AGENT_CONFIG_PATH, "r") as f:
            config_data = json.load(f)

        if name in [config["name"] for config in config_data]:
            return jsonify({"error": f"Agent '{name}' already exists."}), 400

        new_config = {
            "name": name,
            "model": model,
            "system_message": system_message,
            "description": f"Chain-of-thought agent for: {role}",
            "pattern": "chain_of_thought"
        }

        config_data.append(new_config)

        with open(AGENT_CONFIG_PATH, "w") as f:
            json.dump(config_data, f, indent=2)

        return jsonify({"success": True, "message": f"Chain-of-thought agent '{name}' created."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def write_agent_to_config(name, model, system_message, description, pattern="general", topic=None):
    try:
        with open(AGENT_CONFIG_PATH, "r") as f:
            config_data = json.load(f)
        agent_names = set([config["name"] for config in config_data])
        if name in agent_names:
            return jsonify({"error": f"Agent '{name}' already exists."}), 400
        new_config = {
            "name": name,
            "model": model,
            "system_message": system_message,
            "description": description,
            "pattern": pattern
        }
        if topic:
            new_config["topic"] = topic
        config_data.append(new_config)

        with open(AGENT_CONFIG_PATH, "w") as f:
            json.dump(config_data, f, indent=2)
        return jsonify({"success": True, "message": f"Agent '{name}' created successfully."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def create_maker_checker_agents(data):
    base_name = data.get("base_name", "").strip()
    model = data.get("model", "gpt-4o")
    scope = data.get("scope", "").strip()
    maker_rules = data.get("maker_rules", [])
    checker_criteria = data.get("checker_criteria", [])

    if not base_name or not scope or not maker_rules or not checker_criteria:
        return jsonify({"error": "Missing one or more required fields."}), 400

    maker_name = base_name + "-maker"
    checker_name = base_name + "-checker"

    maker_prompt = (
            f"You are **{maker_name}**, a maker agent for the writing task: {scope}\n\n"
            f"You must follow these rules when constructing your draft:\n" +
            "\n".join(f"- {r}" for r in maker_rules) +
            f"\n\nSend your draft to **{checker_name}** for review. Do not evaluate it yourself."
    )

    checker_prompt = (
            f"You are **{checker_name}**, responsible for reviewing drafts made by the "
            f"{maker_name} which creates drafts for the following scope:\n{scope}\n\n"
            f"You must assess the {maker_name}'s draft based on the following criteria:\n" +
            "\n".join(f"- {c}" for c in checker_criteria) +
            f"\n\nIf there are any key criteria that the draft is missing, send a notification back "
            f"to **{maker_name}** indicating what it is missing, along with any additional feedback"
            f"you might have. Do not create or edit drafts."
    )

    maker_config = {
        "name": maker_name,
        "model": model,
        "system_message": maker_prompt,
        "description": f"Maker agent for scope: {scope}",
        "pattern": "maker",
        "counterpart": checker_name
    }

    checker_config = {
        "name": checker_name,
        "model": model,
        "system_message": checker_prompt,
        "description": f"Checker agent for scope: {scope}",
        "pattern": "checker",
        "counterpart": maker_name
    }

    with open(AGENT_CONFIG_PATH, "r") as f:
        config_data = json.load(f)
    config_data.extend([maker_config, checker_config])
    with open(AGENT_CONFIG_PATH, "w") as f:
        json.dump(config_data, f, indent=2)

    return jsonify({"success": True, "message": f"Maker-Checker pair '{maker_name}' & '{checker_name}' created."})

def create_specialist_prompt(data):
    topic = data.get("topic", "").strip()
    audience_pairs = data.get("audience_interactions", [])

    if not topic or not audience_pairs:
        raise ValueError("Missing required fields for specialist.")

    # === Retrieve topic description from Pinecone
    topic_desc = vector_store_manager.get_index_description(topic)

    # === Default audience: other agents
    default_pairs = [{"audience": "other agents", "interaction": "requesting summaries, definitions, and data from your domain"}]
    full_pairs = default_pairs + audience_pairs

    # === Format into markdown-style bullet list
    formatted_pairs = "\n".join(f"- **{pair['audience']}**: {pair['interaction']}" for pair in full_pairs)

    system_message = (
        f"You are a subject matter expert (SME) in the domain of the following knowledge base: {topic}. "
        f"Below is the high-level description of the knowledge base information that you are equiped with"
        f"and how to use it in the scope of a conversation.\n"
        f"**Domain Description:**\n{topic_desc}\n\n"
        f"**Intended Audience & Response Instructions:**\n{formatted_pairs}\n"
        f"Additionally, if the query came from someone besides the user (from one of the types of audience above)"
        f"then check the conversation and make sure\n\n"
        f"**Domain-Jurisdiction**\n"
        f"Your ONLY role is to strictly provide information relevant to the user's query. Communicate it in an "
        f"effective way, but do not generate any additional content beyond simply providing information for the "
        f"user or other agents to use as reference.\n"
        f"Base your answers on information from this knowledge base only when responding. If the task/query is "
        f"interdisciplinary, then only provide information to the parts of the task that fall in the domain of "
        f"your knowledge base. Then defer the remainder of the task to the other agents to resolve.\n\n"
        f"**Citation Instructions**\n"
        f"If you receive chunks of text with their associated sources in the form of markdown links,"
        f"then use the links to refer to the sources when generating the response. Do not use the same"
        f"source repeatedly, and be sure to properly render any images with these links if they are "
        f"relevant to answering the query."
    )

    description = (
        f"Specialist agent for topic: {topic}, "
        f"whose domain primarily revolves around:\n{topic_desc}"
    )
    return system_message, description




#===manage_vectorstore.html
#===Topic-Management====
"""
The following methods are used for tracking, creating, &
removing pinecone indexes on Pinecone. These methods include

-   list_indexes()
        List existing indexes or "topics" for
        dropdown lists on the frontend.
        
-   get_index_description() & update_index_description()
        retrieve the general description for a
        given index, allowing for it to be
        edited and saved if changes are needed.
        
-   create_index()
        For creating new pinecone indices for a topic.
        Its description is saved in a Table of Contents
        index, separate from the other indexes, while a
        whole index with that name is created separately.
        
-   delete_index()
        For deleting an index on pinecone and removing
        its description from the Table of Contents
        index, ensuring total consistency
"""
@app.route('/list_indexes', methods=['GET'])
def list_indexes():
    indexes = vector_store_manager.list_indexes()
    return jsonify(indexes)

@app.route('/get_index_description', methods=['POST'])
def get_index_description():
    data = request.json
    index_name = data.get("index_name")
    if not index_name:
        return jsonify({"error": "Index name is required."}), 400
    try:
        description = vector_store_manager.get_index_description(index_name)
        return jsonify({"description": description})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update_index_description', methods=['POST'])
def update_index_description():
    data = request.json
    index_name = data.get("index_name")
    new_description = data.get("description", "")
    if not index_name:
        return jsonify({"error": "Index name is required."}), 400
    try:
        vector_store_manager.upsert_metadata(index_name, new_description)
        return jsonify({"message": f"Description for '{index_name}' updated successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/create_index', methods=['POST'])
def create_index():
    data = request.json
    index_name = data.get("index_name")
    description = data.get("description", "")
    if not index_name or not index_name.islower() or not all(c.isalnum() or c == '-' for c in index_name):
        return jsonify({"error": "Index name must be lowercase, alphanumeric, or contain '-' only."}), 400
    try:
        vector_store_manager.create_index(index_name)
        vector_store_manager.upsert_metadata(index_name, description)
        return jsonify({"message": f"Index '{index_name}' created successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_index', methods=['POST'])
def delete_index():
    index_name = request.json.get("index_name")
    if not index_name:
        return jsonify({"error": "Index name is required."}), 400
    try:
        vector_store_manager.delete_index(index_name)

        # Update affected specialist agents
        with open(agent_config_path, "r") as f:
            data = json.load(f)

        updated = False
        for agent in data:
            if agent.get("pattern") == "specialist" and agent.get("topic") == index_name:
                agent.pop("topic", None)
                agent["system_message"] += "\n\n(Note: this agent's topic was deleted and it now functions using general knowledge.)"
                updated = True

        if updated:
            with open(agent_config_path, "w") as f:
                json.dump(data, f, indent=2)

        return jsonify({"message": f"Index '{index_name}' deleted and agents updated."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

#===Document Management===
"""
The following methods are used for managing individual
& multiple files that have been uploaded locally for
a particular topic:

-   list_uploaded_files()
        This lists all files that have been uploaded
        for a specific topic.
        
-   upload_document()
        This method takes in a file from the frontend
        and saves it locally within a directory with
        the file's name as a subdirectory of the
        topic it was submitted under. Additionally,
        pdf's, docx's, and pptx's get an images/
        subfolder where any images in a document
        are extracted
        
-   embed_files()
        This method extracts the text of selected
        documents, and breaks them into chunks. Each
        chunk is then embedded as a vector onto
        the corresponding pinecone index. For images,
        we first create a description using the
        GPT-4-Turbo model, we can then embedd like
        other text chunks. Appropriate metadata
        is stored so that we can retrieve the text,
        source file, or image, as well.

-   delete_files()
        This method deletes vectors from selected
        files on the local directory, as well as
        any vectors they were embedded in on
        Pinecone. This ensures that we cannot
        use them for context, going forward.
"""
@app.route('/list_uploaded_files', methods=['POST'])
def list_uploaded_files():
    """Lists the names of uploaded files for a given topic/index with their embedding status."""
    data = request.get_json()
    index_name = data.get("index_name")
    if not index_name:
        return jsonify({"error": "Index name is required."}), 400
    folder_path = os.path.join(UPLOAD_FOLDER, index_name)
    if not os.path.exists(folder_path):
        return jsonify({"files": []})

    files = os.listdir(folder_path)
    file_info = []

    for file_name in files:
        embedded = vector_store_manager.is_embedded(index_name, file_name)
        file_info.append({"name": file_name, "embedded": embedded})

    return jsonify({"files": file_info})

@app.route('/upload_document', methods=['POST'])
def upload_document():
    index_name = request.form.get('index_name')
    file = request.files.get('file')
    user_description = request.form.get('image_description', '').strip()

    if not index_name:
        return jsonify({"error": "Invalid index selection."}), 400
    if not file:
        return jsonify({"error": "No file provided."}), 400

    topic_dir = os.path.join(UPLOAD_FOLDER, index_name)
    os.makedirs(topic_dir, exist_ok=True)
    file_name = file.filename.replace(" ","_")
    document_dir = os.path.join(topic_dir, file_name.replace(".","_dot_"))
    os.makedirs(document_dir, exist_ok=True)
    file_path = os.path.join(document_dir, file_name)
    file.save(file_path)

    ext = "." + file_name.split(".")[-1].lower()
    images_saved = []

    if ext not in IMG_EXTENSIONS:
        # For docs (PDF/DOCX/PPTX), extract embedded images
        document_image_dir = os.path.join(document_dir, "images")
        os.makedirs(document_image_dir, exist_ok=True)
        if ext.endswith(".pdf"):
            images_saved = extract_images_from_pdf(document_dir, file_path, document_image_dir)
        elif ext.endswith(".docx"):
            images_saved = extract_images_from_docx(document_dir, file_path, document_image_dir)
        elif ext.endswith(".pptx"):
            images_saved = extract_images_from_pptx(document_dir, file_path, document_image_dir)
    else:
        # For standalone images
        images_saved = [file_path]
        if user_description:
            alt_map_path = os.path.join(document_dir, "alt_image_map.json")
            with open(alt_map_path, "w", encoding="utf-8") as f:
                json.dump([{
                    "path": file_path,
                    "alt_text": user_description
                }], f, indent=4)

    return jsonify({
        "message": f"Document '{file.filename}' and {len(images_saved)} images saved successfully."
    })

@app.route('/embed_files', methods=['POST'])
def embed_files():
    data = request.json
    index_name = data.get("index_name")
    files_to_embed = list(data.get("files", []))
    chunk_size = int(data.get("chunk_size", 500))

    if not index_name or not files_to_embed:
        return jsonify({"error": "Index name and files are required."}), 400

    topic_path = os.path.join(UPLOAD_FOLDER, index_name)
    total_text_vectors = 0
    total_image_vectors = 0

    for file_name in files_to_embed:
        file_dir = os.path.join(topic_path, file_name)
        _file_name = file_name.replace("_dot_", ".")
        file_path = os.path.join(file_dir, _file_name)

        ext = "." + _file_name.split(".")[-1].lower()
        # === 🟢 First: TEXT-BASED EMBEDDINGS ===
        if ext in DOC_EXTENSIONS:
            try:
                text_chunks = extract_text(file_path, chunk_size)
                total_text_vectors += len(text_chunks)
                file_paths = [file_path] * len(text_chunks)

                if text_chunks:
                    vector_store_manager.upsert_vectors(
                        index_name,
                        src_doc=file_name,
                        file_paths=file_paths,
                        chunks=text_chunks,
                        embed_type="text"
                    )
            except Exception as e:
                print(f"Error extracting text from {file_name}: {e}")

        # === 🟢 Second: IMAGE-BASED EMBEDDINGS via alt_image_map.json ===
        alt_map_path = os.path.join(file_dir, "alt_image_map.json")
        if os.path.exists(alt_map_path):
            try:
                with open(alt_map_path, "r", encoding="utf-8") as f:
                    alt_images_info = json.load(f)

                image_paths = []
                image_descriptions = []

                for entry in alt_images_info:
                    img_path = entry.get("path")
                    alt_text = entry.get("alt_text")
                    if os.path.exists(img_path) and alt_text:
                        image_paths.append(img_path)
                        image_descriptions.append(alt_text)

                if image_descriptions:
                    vector_store_manager.upsert_vectors(
                        index_name,
                        src_doc=file_name,
                        file_paths=image_paths,
                        chunks=image_descriptions,
                        embed_type="image"
                    )
                    total_image_vectors += len(image_descriptions)

            except Exception as e:
                print(f"Error reading alt_image_map.json for {file_name}: {e}")

    return jsonify({
        "message": f"Embedding complete: {total_text_vectors} text chunks and {total_image_vectors} images processed."
    })

@app.route('/unembed_files', methods=['POST'])
def unembed_files():
    data = request.json
    index_name = data.get("index_name")
    files_to_unembed = list(data.get("files", []))
    if not index_name or not files_to_unembed:
        return jsonify({"error": "Index name and files to unembed are required."}), 400

    for file_name in files_to_unembed:
        _file_name = file_name.replace("_dot_",".")
        try:
            vector_store_manager.delete_vectors_by_source(index_name, file_name)
        except Exception as e:
            return jsonify({"error": f"Failed to unembed '{file_name}': {str(e)}"}), 500

    return jsonify({"message": f"Vectors for selected files have been removed from '{index_name}'."})

@app.route('/delete_files', methods=['POST'])
def delete_files():
    data = request.json
    index_name = data.get("index_name")
    files_to_delete = data.get("files", [])
    if not index_name or not files_to_delete:
        return jsonify({"error": "Index name and files to delete are required."}), 400

    for file_name in files_to_delete:
        document_dir = os.path.join(UPLOAD_FOLDER, index_name, file_name)
        if os.path.exists(document_dir):
            shutil.rmtree(document_dir)
        vector_store_manager.delete_vectors_by_source(index_name, file_name)

    return jsonify({"message": f"Selected files and associated data have been deleted from '{index_name}'."})


if __name__ == "__main__":
    try:
        shutil.rmtree(".cache/")
    except:
        pass
    app.run("0.0.0.0", debug=True)
