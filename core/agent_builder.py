# core/agent_builder.py
import json
import re
import os
import logging

from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq

from core.agent_state import AgentState
from core.tools import (
    create_rag_tool,
    create_summarization_tool,
    create_web_search_tool,
    call_gemini_summary,
)
from config.config import load_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.05  # when PDF confidence is below this, merge with WebSearch


def _decision_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         """You are a routing policy. Respond ONLY in natural language.
- You may write in full sentences, paragraphs, bullet points, or numbered lists.
- Do NOT return JSON, code, or backticks.
- Keep answers clear and structured for readability.

Actions (choose exactly one):
- "AnswerQuestionAboutPDFs"  -> for specific questions likely answerable from the PDFs.
- "SummarizePDF"             -> when the user asks for a summary of a document.
- "WebSearch"                -> when the question is unrelated to PDFs or PDFs insufficient.

Rules:
- If unsure whether PDFs contain the answer, prefer AnswerQuestionAboutPDFs first.
- For SummarizePDF, set "input" to the best-matching filename (string).
- For other actions, set "input" to the user's original query verbatim.

Available documents: {available_docs}

User request: {input}
Respond in the format:
""")
    ])


def _safe_json_parse(s: str):
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE)

    if s in {"WebSearch", "AnswerQuestionAboutPDFs", "SummarizePDF"}:
        return {"action": s, "input": ""}

    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

    # --- Heuristic fallback ---
    lower_s = s.lower()

    # If the LLM is clearly suggesting search or the query is general knowledge
    if any(word in lower_s for word in ["search", "web", "internet", "weather", "news", "kathmandu"]):
        return {"action": "WebSearch", "input": s}

    # Otherwise, treat it as a direct model-generated answer
    return {"action": "DirectAnswer", "input": s}



def compile_agent(vector_store, raw_documents):
    """Build and compile the LangGraph agent with robust routing."""
    try:
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.2,
            max_tokens=1024,
            api_key=load_api_key(),
        )

        # Tools
        websearch_tool = create_web_search_tool()
        rag_tool = create_rag_tool(vector_store, llm)
        summarize_tool = create_summarization_tool(raw_documents, llm)

        tools = [rag_tool, summarize_tool, websearch_tool]
        tool_map = {t.name: t for t in tools}

        decision_prompt = _decision_prompt()
        decision_chain = decision_prompt | llm | StrOutputParser()

        # -------- Agent node --------
        def run_agent(state):
            logger.info("Running agent...")
            try:
                user_input = state["input"]
                # Get memory for this thread first, fallback to agent state
                thread_id = state.get("config", {}).get("thread_id")
                current_history = state.get("chat_history", [])


                available_docs = [os.path.basename(doc.metadata.get('source', 'unknown')) for doc in raw_documents]
                available_docs_str = ", ".join(available_docs) if available_docs else "No documents available"

                decision_text = decision_chain.invoke({"input": user_input, "available_docs": available_docs_str})
                logger.info(f"Agent decision raw: {decision_text}")

                parsed = _safe_json_parse(decision_text)
                action = (parsed.get("action") or "").strip()
                tool_input = parsed.get("input") or user_input

                if action == "SummarizePDF" and available_docs:
                    if not tool_input or not tool_input.endswith(".pdf"):
                        tool_input = available_docs[0]

                if action in {"AnswerQuestionAboutPDFs", "SummarizePDF", "WebSearch"}:
                    return {"agent_outcome": AgentAction(tool=action, tool_input=tool_input, log=decision_text)}
                elif action == "DirectAnswer":
                    return {"agent_outcome": AgentFinish(return_values={"output": tool_input}, log="Direct model answer")}
                else:
                    return {"agent_outcome": AgentFinish(return_values={"output": tool_input}, log="Direct response")}
            except Exception as e:
                logger.error(f"Error in agent execution: {str(e)}")
                return {"agent_outcome": AgentFinish(
                    return_values={"output": f"I encountered an error: {str(e)}."},
                    log="Error in agent"
                )}

        # -------- Tool execution node --------
        def execute_tools(state):
            agent_outcome = state.get("agent_outcome")
            chat_history = state.get("chat_history", [])
            if not isinstance(agent_outcome, AgentAction):
                return {"intermediate_steps": [], "confidence": 0.0}

            tool_name = agent_outcome.tool
            tool_input = agent_outcome.tool_input

            if tool_name == "AnswerQuestionAboutPDFs":
                answer, pdf_confidence = tool_map["AnswerQuestionAboutPDFs"].func(tool_input, chat_history=chat_history)
                if pdf_confidence < CONFIDENCE_THRESHOLD:
                    web_answer = tool_map["WebSearch"].func(tool_input)
                    answer = call_gemini_summary(answer, web_answer, tool_input)
                    log_text = f"PDF weak (confidence={pdf_confidence:.3f}); merged with WebSearch."
                else:
                    log_text = f"Used RAG with confidence={pdf_confidence:.3f}"

                final = AgentFinish(return_values={"output": answer}, log=log_text)
                return {"agent_outcome": final, "intermediate_steps": [(agent_outcome, answer)], "confidence": pdf_confidence}

            elif tool_name == "SummarizePDF":
                summary = tool_map["SummarizePDF"].func(tool_input, chat_history=chat_history)
                summary_confidence = 0.0 if len(summary.strip()) < 30 else 1.0

                if summary_confidence < CONFIDENCE_THRESHOLD:
                    web_answer = tool_map["WebSearch"].func(tool_input)
                    summary = call_gemini_summary(summary, web_answer, tool_input)
                    log_text = f"Summary weak (confidence={summary_confidence:.3f}); merged with WebSearch."
                    summary_confidence = 1.0
                else:
                    log_text = f"Used SummarizePDF with confidence={summary_confidence:.3f}"

                final = AgentFinish(return_values={"output": summary}, log=log_text)
                return {"agent_outcome": final, "intermediate_steps": [(agent_outcome, summary)], "confidence": summary_confidence}

            elif tool_name == "WebSearch":
                original_user_input = state.get("input", "")
                ws = tool_map["WebSearch"].func(original_user_input)
                final = AgentFinish(return_values={"output": ws}, log="Used WebSearch")
                return {"agent_outcome": final, "intermediate_steps": [(agent_outcome, ws)], "confidence": 0.0}

            final = AgentFinish(return_values={"output": "Unhandled action."}, log="Unhandled")
            return {"agent_outcome": final, "intermediate_steps": [], "confidence": 0.0}

        # -------- Continuation logic --------
        def should_continue(state):
            agent_outcome = state.get("agent_outcome")
            if isinstance(agent_outcome, AgentFinish):
                return "end"
            elif isinstance(agent_outcome, AgentAction):
                return "continue"
            return "end"

        # -------- Build graph --------
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", run_agent)
        workflow.add_node("action", execute_tools)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
        workflow.add_edge("action", END)

        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        logger.info("Agent compiled successfully with RAG + Summarization + Web fallback!")
        return app

    except Exception as e:
        logger.error(f"Error compiling agent: {str(e)}")
        raise e
