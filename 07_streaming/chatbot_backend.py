import os
from typing import Annotated, Iterator, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class ChatbotBackend:
    def __init__(self, model_name: str | None = None, temperature: float = 0.0) -> None:
        if load_dotenv is not None:
            load_dotenv()

        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.llm = ChatGroq(model=self.model_name, temperature=temperature)
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        graph = StateGraph(ChatState)
        graph.add_node("chat_node", self._chat_node)
        graph.add_edge(START, "chat_node")
        graph.add_edge("chat_node", END)
        return graph.compile(checkpointer=self.checkpointer)

    def _chat_node(self, state: ChatState) -> dict[str, list[AIMessage]]:
        response = self.llm.invoke(state["messages"])
        return {"messages": [response]}

    def _config(self, thread_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": thread_id}}

    def get_messages(self, thread_id: str) -> list[BaseMessage]:
        snapshot = self.workflow.get_state(self._config(thread_id))
        if not snapshot or not snapshot.values:
            return []
        return list(snapshot.values.get("messages", []))

    def invoke_reply(self, thread_id: str, user_message: str) -> str:
        result = self.workflow.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=self._config(thread_id),
        )
        return result["messages"][-1].content

    def stream_reply(self, thread_id: str, user_message: str) -> Iterator[str]:
        config = self._config(thread_id)
        collected_chunks: list[str] = []

        try:
            for message_chunk, metadata in self.workflow.stream(
                {"messages": [HumanMessage(content=user_message)]},
                config=config,
                stream_mode="messages",
            ):
                if metadata.get("langgraph_node") != "chat_node":
                    continue
                text = self._extract_text(message_chunk)
                if text:
                    collected_chunks.append(text)
                    yield text
        except Exception:
            if collected_chunks:
                raise
            yield self.invoke_reply(thread_id, user_message)
            return

        if not collected_chunks:
            messages = self.get_messages(thread_id)
            if messages and isinstance(messages[-1], AIMessage):
                yield messages[-1].content
                return
            yield self.invoke_reply(thread_id, user_message)

    @staticmethod
    def _extract_text(message_chunk: BaseMessage) -> str:
        content = getattr(message_chunk, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)
        return str(content)
