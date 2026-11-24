"""LangChain tool-calling agent, 3 PDF tools, ReAct activity log, and verification."""

import os
from typing import Any, Dict, List, Optional

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from utils.vector_store import search_chunks


DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are DocQuery, a helpful conversational PDF Research Assistant.
You follow a clear Reason -> Act -> Observe -> Respond workflow to answer questions about the currently uploaded PDF document.

You have access to ONLY three tools:
1. search_pdf: Use this for direct questions or finding specific facts in the PDF.
2. summarize_topic: Use this when the user asks to summarize, overview, or list key points of a topic or section in the PDF.
3. compare_topics: Use this when the user asks to compare or contrast two topics, concepts, or methods from the PDF.

Rules you MUST follow:
- Use the conversation history (chat_history) to understand incomplete follow-up questions (for example, "What are its three types?" or "Explain the first one with an example").
- When calling a tool for a follow-up question, rewrite the query/topic clearly using context from chat_history (for example, replace "the first one" with "supervised learning").
- You may answer directly from chat_history ONLY when the user is asking about something already answered in recent conversation and no new PDF lookup is needed. For any new fact, summary, comparison, or detail request, call the appropriate tool first.
- Prioritize information retrieved from the uploaded PDF. Never invent facts, add outside definitions, or include background details not present in the retrieved PDF chunks.
- Always include page number references (such as `(Page 2)`) for retrieved information.
- Give simple, clear, beginner-friendly explanations.
- Use a Markdown table when comparing two topics.
- Use bullet points for summaries where appropriate.
- If the retrieved PDF chunks do not contain the answer to the user's question, clearly state that the information could not be found in the uploaded PDF document.
- Do not reveal private internal reasoning; provide a clean final answer."""


class VerificationResult(BaseModel):
    """Structured output schema for the simple answer verification step."""

    is_supported_by_pdf: bool = Field(
        description=(
            "True if the core answer is supported by the retrieved PDF content "
            "(or prior verified conversation context), allowing normal paraphrasing "
            "and summarization, OR if the answer explicitly states that the "
            "information is not in the document."
        )
    )
    addresses_question: bool = Field(
        description="True if the answer directly addresses the user's question."
    )
    contains_unsupported_info: bool = Field(
        description=(
            "True ONLY if the answer introduces concrete facts, technical terms, "
            "numbers, or claims that are not supported by the retrieved PDF evidence."
        )
    )
    states_not_found: bool = Field(
        description=(
            "True if the answer states that the requested information could not "
            "be found in the uploaded PDF document."
        )
    )
    reason: str = Field(
        description="One-sentence explanation of the verification decision."
    )


def build_tools(
    vector_store: Chroma,
    doc_id: str,
    retrieved_sources: List[Dict[str, Any]],
) -> List[Any]:
    """Create the three LangChain tools bound to the active PDF vector store."""

    def _record_sources(sources: List[Dict[str, Any]]) -> None:
        existing = {(src["page"], src["snippet"]) for src in retrieved_sources}
        for src in sources:
            key = (src["page"], src["snippet"])
            if key not in existing:
                retrieved_sources.append(src)
                existing.add(key)

    @tool
    def search_pdf(query: str) -> str:
        """Search the uploaded PDF for relevant information matching a query.

        Use this tool to answer factual questions or look up specific details
        in the uploaded PDF document. Returns the top 4 chunks with page numbers.
        """
        result = search_chunks(vector_store, query=query, k=4, doc_id=doc_id)
        _record_sources(result["sources"])
        return result["formatted_text"]

    @tool
    def summarize_topic(topic: str) -> str:
        """Retrieve relevant PDF sections about a topic so it can be summarized.

        Use this tool when the user asks to summarize, give an overview of,
        or list main points about a topic from the uploaded PDF.
        """
        result = search_chunks(vector_store, query=topic, k=4, doc_id=doc_id)
        _record_sources(result["sources"])
        return (
            f"Retrieved PDF sections for summarizing '{topic}':\n\n"
            f"{result['formatted_text']}"
        )

    @tool
    def compare_topics(topic_a: str, topic_b: str) -> str:
        """Retrieve PDF information about two topics separately for comparison.

        Use this tool when the user asks to compare or contrast two concepts,
        methods, or topics from the uploaded PDF.
        """
        res_a = search_chunks(vector_store, query=topic_a, k=4, doc_id=doc_id)
        res_b = search_chunks(vector_store, query=topic_b, k=4, doc_id=doc_id)
        _record_sources(res_a["sources"])
        _record_sources(res_b["sources"])
        return (
            f"=== Information on Topic A: {topic_a} ===\n"
            f"{res_a['formatted_text']}\n\n"
            f"=== Information on Topic B: {topic_b} ===\n"
            f"{res_b['formatted_text']}"
        )

    return [search_pdf, summarize_topic, compare_topics]


def verify_answer(
    llm: ChatGroq,
    question: str,
    answer: str,
    retrieved_context: str,
    conversation_context: str,
    used_tools: bool,
) -> Dict[str, Any]:
    """Perform a simple verification check before showing the final answer."""
    evidence = retrieved_context.strip()
    if not used_tools and conversation_context.strip():
        evidence = (
            "Answered from prior conversation history:\n"
            + conversation_context.strip()
        )

    if not evidence:
        return {
            "passed": False,
            "status": "Not Found in Document",
            "final_answer": (
                "I could not find supporting information for this question in the "
                "uploaded PDF document."
            ),
            "reason": "No PDF content or prior document context was available to verify the answer.",
        }

    verifier_prompt = (
        "You are a fact-checker verifying an AI answer against retrieved PDF content.\n\n"
        f"User Question:\n{question}\n\n"
        f"Evidence (Retrieved PDF Chunks / Prior Verified Context):\n{evidence}\n\n"
        f"Proposed Answer:\n{answer}\n\n"
        "Check:\n"
        "1. Is the core answer supported by the evidence above (normal paraphrasing or summarizing of retrieved facts is supported), OR does it clearly state the info is not in the document?\n"
        "2. Does it address the user's question?\n"
        "3. Does it introduce concrete outside facts, claims, or technical terms not present in the evidence?\n"
    )

    try:
        structured_llm = llm.with_structured_output(VerificationResult)
        check: VerificationResult = structured_llm.invoke(verifier_prompt)

        if check.states_not_found:
            return {
                "passed": True,
                "status": "Not Found in Document",
                "final_answer": answer,
                "reason": check.reason,
            }

        if not check.is_supported_by_pdf:
            return {
                "passed": False,
                "status": "Not Found in Document",
                "final_answer": (
                    "I could not find enough supporting information in the uploaded PDF "
                    "to answer this question reliably."
                ),
                "reason": check.reason,
            }

        if check.contains_unsupported_info:
            rewrite_prompt = (
                "Rewrite the answer below so that EVERY claim, term, and example is "
                "strictly supported by the provided PDF evidence. Remove any outside "
                "terms or claims not present in the evidence, keep accurate `(Page X)` "
                "citations, and keep a clean, clear format (tables for comparisons, "
                "bullets for summaries).\n\n"
                f"User Question:\n{question}\n\n"
                f"PDF Evidence:\n{evidence}\n\n"
                f"Draft Answer:\n{answer}\n\n"
                "Rewritten Grounded Answer:"
            )
            rewritten = llm.invoke(rewrite_prompt).content.strip()
            if rewritten:
                return {
                    "passed": True,
                    "status": "Verified",
                    "final_answer": rewritten,
                    "reason": "Grounded strictly in the retrieved PDF chunks.",
                }

        status = "Verified" if check.addresses_question else "Partially Addressed"
        return {
            "passed": True,
            "status": status,
            "final_answer": answer,
            "reason": check.reason,
        }
    except Exception as exc:
        # Fallback cleanly if structured output fails unexpectedly
        return {
            "passed": True,
            "status": "Verification Skipped",
            "final_answer": answer,
            "reason": f"Verification check could not run: {exc}",
        }


def run_docquery_agent(
    question: str,
    vector_store: Chroma,
    doc_id: str,
    chat_history: List[BaseMessage],
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the DocQuery tool-calling agent, log tool activity, and verify the answer."""
    resolved_key = api_key or os.getenv("GROQ_API_KEY")
    if not resolved_key:
        raise ValueError(
            "Groq API key is missing. Set GROQ_API_KEY in your .env file."
        )

    resolved_model = model_name or os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    llm = ChatGroq(
        model_name=resolved_model,
        temperature=0,
        groq_api_key=resolved_key,
    )

    retrieved_sources: List[Dict[str, Any]] = []
    tools = build_tools(
        vector_store=vector_store,
        doc_id=doc_id,
        retrieved_sources=retrieved_sources,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    def _run_executor(active_llm: ChatGroq) -> Dict[str, Any]:
        agent = create_tool_calling_agent(active_llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            return_intermediate_steps=True,
            max_iterations=5,
            handle_parsing_errors=True,
        )
        return executor.invoke(
            {
                "input": question,
                "chat_history": chat_history,
            }
        )

    try:
        result = _run_executor(llm)
    except Exception as exc:
        if "model_not_found" in str(exc) and resolved_model != DEFAULT_GROQ_MODEL:
            retrieved_sources.clear()
            llm = ChatGroq(
                model_name=DEFAULT_GROQ_MODEL,
                temperature=0,
                groq_api_key=resolved_key,
            )
            result = _run_executor(llm)
        else:
            raise

    raw_answer = result.get("output", "").strip()
    intermediate_steps = result.get("intermediate_steps", [])

    activity_log: List[Dict[str, Any]] = []
    observation_texts: List[str] = []

    for idx, step in enumerate(intermediate_steps, start=1):
        action, observation = step
        obs_str = str(observation)
        observation_texts.append(obs_str)
        activity_log.append(
            {
                "step": idx,
                "tool": action.tool,
                "action_input": action.tool_input,
                "observation": obs_str,
            }
        )

    used_tools = len(activity_log) > 0
    if not used_tools:
        activity_log.append(
            {
                "step": 1,
                "tool": "direct_context_response",
                "action_input": {"reason": "Answered from recent conversation history"},
                "observation": "No new PDF tool call was needed.",
            }
        )

    conversation_context = "\n".join(
        f"{msg.type.upper()}: {msg.content}" for msg in chat_history[-4:]
    )

    verification = verify_answer(
        llm=llm,
        question=question,
        answer=raw_answer,
        retrieved_context="\n\n".join(observation_texts),
        conversation_context=conversation_context,
        used_tools=used_tools,
    )

    return {
        "answer": verification["final_answer"],
        "sources": retrieved_sources,
        "activity": activity_log,
        "verification": {
            "status": verification["status"],
            "reason": verification["reason"],
            "passed": verification["passed"],
        },
    }
