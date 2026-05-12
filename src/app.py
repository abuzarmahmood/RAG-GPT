"""
This is the main app file for the Streamlit app.
"""

############################################################
## Imports 
############################################################

import streamlit as st
from dataclasses import dataclass
from rag_llm import run_query, return_vectordb

############################################################
# Get Vector Database 
############################################################
print("Initializing RAG-GPT application...")
vectordb = return_vectordb()
print("Application ready!")

############################################################
# Run Chat
############################################################


from utils import load_config

config = load_config()

# Display model information in sidebar
st.sidebar.title("Configuration")
st.sidebar.info(f"**Model:** {config.get('model_name', 'Not specified')}")
st.sidebar.info(f"**Documents Retrieved (k):** {config.get('k', 5)}")

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
    response: str = run_query(prompt, vectordb, system_context=st.session_state["system_context"])
    st.session_state[MESSAGES].append(
        Message(actor=ASSISTANT, payload=response))
    st.chat_message(ASSISTANT).write(response)
