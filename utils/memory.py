"""Conversational memory utilities using Streamlit session state."""

from typing import Any, Dict, List, MutableMapping, Optional

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


MAX_EXCHANGES = 6
CHAT_HISTORY_KEY = "chat_history"


def _get_state(state: Optional[MutableMapping[str, Any]] = None) -> MutableMapping[str, Any]:
    """Return the provided state dict or Streamlit's session_state."""
    return state if state is not None else st.session_state


def init_memory(state: Optional[MutableMapping[str, Any]] = None) -> None:
    """Initialize chat history in session state if not already present."""
    store = _get_state(state)
    if CHAT_HISTORY_KEY not in store:
        store[CHAT_HISTORY_KEY] = []


def add_message(
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    activity: Optional[List[Dict[str, Any]]] = None,
    verification: Optional[Dict[str, Any]] = None,
    state: Optional[MutableMapping[str, Any]] = None,
) -> None:
    """Append a user or assistant message entry to session state."""
    store = _get_state(state)
    init_memory(store)
    store[CHAT_HISTORY_KEY].append(
        {
            "role": role,
            "content": content,
            "sources": sources or [],
            "activity": activity or [],
            "verification": verification or {},
        }
    )


def get_display_messages(
    state: Optional[MutableMapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return all messages stored for rendering in the chat UI."""
    store = _get_state(state)
    init_memory(store)
    return list(store[CHAT_HISTORY_KEY])


def get_recent_history_messages(
    max_exchanges: int = MAX_EXCHANGES,
    state: Optional[MutableMapping[str, Any]] = None,
) -> List[BaseMessage]:
    """Return up to the last `max_exchanges` Q&A pairs as LangChain messages.

    One exchange equals one user question and one AI response (2 messages),
    so `max_exchanges=6` preserves up to the last 12 messages.
    """
    store = _get_state(state)
    init_memory(store)

    max_messages = max_exchanges * 2
    recent_entries = store[CHAT_HISTORY_KEY][-max_messages:]

    lc_messages: List[BaseMessage] = []
    for entry in recent_entries:
        role = entry.get("role")
        content = entry.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    return lc_messages


def clear_chat_history(state: Optional[MutableMapping[str, Any]] = None) -> None:
    """Clear only the conversation history while keeping the active PDF intact."""
    store = _get_state(state)
    store[CHAT_HISTORY_KEY] = []
