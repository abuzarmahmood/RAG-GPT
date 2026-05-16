"""
This is the main app file for the Streamlit app.
"""

############################################################
## Imports 
############################################################

import os
import hashlib
from datetime import datetime
import streamlit as st
from dataclasses import dataclass
from rag_llm import run_query, return_vectordb
from utils import load_config, return_paths

############################################################
# Vector Store Change Detection
############################################################

def get_vector_store_mtime(vector_persist_dir):
    """
    Return the most recent modification time across all files in the
    vector store directory, or 0 if the directory does not exist.
    """
    if not os.path.exists(vector_persist_dir):
        return 0
    mtimes = []
    for root, _, files in os.walk(vector_persist_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                mtimes.append(os.path.getmtime(fpath))
            except OSError:
                pass
    return max(mtimes) if mtimes else 0


def calculate_vector_store_hash(vector_persist_dir):
    """
    Calculate a hash of all files in the vector store directory.
    Returns a hex digest string representing the state of the vector store.
    """
    if not os.path.exists(vector_persist_dir):
        return "no_vector_store"
    
    hash_md5 = hashlib.md5()
    
    # Get all files sorted for consistent hashing
    all_files = []
    for root, _, files in os.walk(vector_persist_dir):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            all_files.append(fpath)
    
    # Hash each file's content
    for fpath in sorted(all_files):
        try:
            with open(fpath, 'rb') as f:
                # Hash file path (relative to vector_persist_dir) and content
                rel_path = os.path.relpath(fpath, vector_persist_dir)
                hash_md5.update(rel_path.encode())
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except OSError:
            pass
    
    return hash_md5.hexdigest()


############################################################
# Get Vector Database 
############################################################
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing RAG-GPT application...")

_, _, _, vector_persist_dir = return_paths()

# Load vectordb once per process; track mtime and hash so we can detect updates.
if "vectordb" not in st.session_state:
    st.session_state["vectordb"] = return_vectordb()
    st.session_state["vector_store_mtime"] = get_vector_store_mtime(vector_persist_dir)
    st.session_state["vector_store_hash"] = calculate_vector_store_hash(vector_persist_dir)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Application ready!")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Vector store hash: {st.session_state['vector_store_hash']}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Vector store mtime: {datetime.fromtimestamp(st.session_state['vector_store_mtime']).strftime('%Y-%m-%d %H:%M:%S')}")

# Check whether the vector store has been updated since last load.
current_mtime = get_vector_store_mtime(vector_persist_dir)
if current_mtime > st.session_state["vector_store_mtime"]:
    current_hash = calculate_vector_store_hash(vector_persist_dir)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Vector store updated — reloading...")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Old hash: {st.session_state['vector_store_hash']}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] New hash: {current_hash}")
    st.session_state["vectordb"] = return_vectordb()
    st.session_state["vector_store_mtime"] = current_mtime
    st.session_state["vector_store_hash"] = current_hash
    st.rerun()

vectordb = st.session_state["vectordb"]

############################################################
# Run Chat
############################################################


config = load_config()

# Display model information in sidebar
st.sidebar.title("Configuration")

# Display vector store information
st.sidebar.subheader("Vector Store Status")
if "vector_store_hash" in st.session_state:
    st.sidebar.text(f"Hash: {st.session_state['vector_store_hash'][:16]}...")
if "vector_store_mtime" in st.session_state and st.session_state['vector_store_mtime'] > 0:
    mtime_str = datetime.fromtimestamp(st.session_state['vector_store_mtime']).strftime('%Y-%m-%d %H:%M:%S')
    st.sidebar.text(f"Last modified: {mtime_str}")
st.sidebar.markdown("---")

# Add editable model name
st.sidebar.subheader("Model Settings")
if "model_name" not in st.session_state:
    st.session_state["model_name"] = config.get('model_name', 'gpt-4o-mini')

st.session_state["model_name"] = st.sidebar.text_input(
    "Model name:",
    value=st.session_state["model_name"],
    help="Specify the OpenAI model to use (e.g., gpt-4o-mini, gpt-4, gpt-3.5-turbo)"
)

# Add editable k parameter
if "k" not in st.session_state:
    st.session_state["k"] = config.get('k', 5)

st.session_state["k"] = st.sidebar.number_input(
    "Documents to retrieve (k):",
    min_value=1,
    max_value=20,
    value=st.session_state["k"],
    help="Number of relevant documents to retrieve for context"
)

# Add editable system context
st.sidebar.subheader("System Context")
if "system_context" not in st.session_state:
    st.session_state["system_context"] = config['system_context']

st.session_state["system_context"] = st.sidebar.text_area(
    "Edit system context:",
    value=st.session_state["system_context"],
    height=150,
    help="Modify the system context to change how the AI assistant behaves"
)

@dataclass
class Message:
    actor: str
    payload: str


USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"
if MESSAGES not in st.session_state:
    welcome_message = f"Welcome to RAG-GPT! {config['system_context'].split('.')[0]}. Ask me a question about taste processing and I'll try to answer it."
    st.session_state[MESSAGES] = [Message(
        actor=ASSISTANT,
        payload=welcome_message)]

msg: Message
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)

prompt: str = st.chat_input("Enter a prompt here")

if prompt:
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)
    # response: str = f"You wrote {prompt}"
    response: str = run_query(
        prompt, 
        vectordb, 
        k=st.session_state["k"],
        system_context=st.session_state["system_context"],
        model_name=st.session_state["model_name"]
    )
    st.session_state[MESSAGES].append(
        Message(actor=ASSISTANT, payload=response))
    st.chat_message(ASSISTANT).write(response)
    st.rerun()
