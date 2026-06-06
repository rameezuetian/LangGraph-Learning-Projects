from uuid import uuid4

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from chatbot_backend import ChatbotBackend


st.set_page_config(
    page_title="LangGraph Streaming Chatbot",
    page_icon=":speech_balloon:",
    layout="wide",
)


def get_backend() -> ChatbotBackend:
    if "backend" not in st.session_state:
        st.session_state.backend = ChatbotBackend()
    return st.session_state.backend


def ensure_thread_id() -> str:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())
    return st.session_state.thread_id


def reset_chat() -> None:
    st.session_state.thread_id = str(uuid4())


backend = get_backend()
thread_id = ensure_thread_id()

st.title("LangGraph Streaming Chatbot")
st.caption("A Streamlit chat UI backed by LangGraph memory and Groq.")

with st.sidebar:
    st.subheader("Session")
    st.text_input("Thread ID", key="thread_id")
    st.caption(f"Model: `{backend.model_name}`")
    if st.button("Start New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

messages = backend.get_messages(st.session_state.thread_id)

if not messages:
    st.info("Set your `GROQ_API_KEY`, then start chatting.")

for message in messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

prompt = st.chat_input("Ask anything")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for chunk in backend.stream_reply(st.session_state.thread_id, prompt):
            full_response += chunk
            placeholder.markdown(full_response or " ")
