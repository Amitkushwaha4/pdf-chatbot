from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from core.agent_state import AgentState
from core.tools import create_rag_tool, create_realtime_tool, create_summarization_tool, create_web_search_tool
import re
import os
import logging
from config.config import load_api_key

# Gemini
from google import genai
from google.genai import types

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.3  # minimum confidence to skip WebSearch


def call_gemini_summary(pdf_answer: str, web_answer: str, query: str) -> str:
    """Use Gemini .5 Flash to merge PDF and WebSearch data into a human-style answer."""
    try:
        client = genai.Client()  # GEMINI_API_KEY picked up from env
        prompt = (
            f"User query: {query}\n\n"
            f"Here is the information I found:\n\n"
            f"From PDF:\n{pdf_answer}\n\n"
            f"From Web:\n{web_answer}\n\n"
            f"Task: Summarize this into a single cohesive, human-style answer. "
            f"Make it clear, concise, and natural, combining both sources where possible. "
            f"If they disagree, mention both perspectives briefly."
        )
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        # response.text is expected; fallback to full object if not present
        return getattr(response, "text", str(response)).strip()
    except Exception as e:
        logger.error(f"Gemini summarization failed: {e}")
        return f"PDF:\n{pdf_answer}\n\nWeb:\n{web_answer}"


def compile_agent(vector_store, raw_documents):
    """
    Builds and compiles a custom LangGraph agent with proper PDF confidence logic and Gemini merging.
    """
    try:
        # Initialize LLM
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.2,
            max_tokens=2048,
            api_key=load_api_key()
        )

        # Create tools
        tools = [
            create_rag_tool(vector_store, llm),
            create_summarization_tool(raw_documents, llm),
            create_web_search_tool(),
            create_realtime_tool()
        ]

        # Decision-making prompt
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that helps users with PDF document analysis. 

Available tools:
1. AnswerQuestionAboutPDFs - Use this for answering specific questions about document content
2. SummarizePDF - Use this to summarize specific PDF documents
3. WebSearch - Use this if the user's question is not related to the uploaded PDFs and requires information from the internet
4. RealtimeData - Use this for real-time queries like weather, stock prices, and live news

Available documents: {available_docs}

Analyze the user's request and decide what to do:

If the user is asking a specific question about document content (like "what are the skills", "what companies", etc.), respond with:
ACTION: AnswerQuestionAboutPDFs
INPUT: [the user's question]

If the user wants a summary, overview, or general information about a document, respond with:
ACTION: SummarizePDF
INPUT: [pick the most relevant document from available docs, or if only one document, use that]

If the user's question is NOT related to the uploaded PDFs, or if you cannot find an answer in the PDFs, respond with:
ACTION: WebSearch
INPUT: [the user's question]

If you can answer directly without tools, respond with:
ACTION: DIRECT
INPUT: [your direct answer]
             
If the user asks for real-time data like weather, stock prices, or live news, respond with:
ACTION: RealtimeData
INPUT: [user query]

User request: {input}

Decision:"""),
        ])

        # --- Parsing Agent Response ---
        def parse_agent_response(response_text, available_docs):
            """Parse the agent's decision from the response text."""
            if "ACTION: AnswerQuestionAboutPDFs" in response_text:
                input_match = re.search(r"INPUT:\s*(.+)", response_text)
                if input_match:
                    return AgentAction(
                        tool="AnswerQuestionAboutPDFs",
                        tool_input=input_match.group(1).strip(),
                        log=response_text
                    )

            elif "ACTION: SummarizePDF" in response_text:
                input_match = re.search(r"INPUT:\s*(.+)", response_text)
                if input_match:
                    tool_input = input_match.group(1).strip()
                    # If user didn't provide exact filename, try to resolve
                    if not tool_input.endswith('.pdf') and available_docs:
                        for doc in available_docs:
                            if tool_input.lower() in doc.lower():
                                tool_input = doc
                                break
                        else:
                            tool_input = available_docs[0]
                    return AgentAction(
                        tool="SummarizePDF",
                        tool_input=tool_input,
                        log=response_text
                    )

            elif "ACTION: WebSearch" in response_text:
                input_match = re.search(r"INPUT:\s*(.+)", response_text)
                if input_match:
                    return AgentAction(
                        tool="WebSearch",
                        tool_input=input_match.group(1).strip(),
                        log=response_text
                    )

            elif "ACTION: RealtimeData" in response_text:
                input_match = re.search(r"INPUT:\s*(.+)", response_text)
                if input_match:
                    return AgentAction(
                        tool="RealtimeData",
                        tool_input=input_match.group(1).strip(),
                        log=response_text
                    )

            elif "ACTION: DIRECT" in response_text:
                input_match = re.search(r"INPUT:\s*(.+)", response_text, re.DOTALL)
                if input_match:
                    return AgentFinish(
                        return_values={"output": input_match.group(1).strip()},
                        log=response_text
                    )

            # Fallback: treat whole response as direct answer
            return AgentFinish(
                return_values={"output": response_text},
                log="Direct response"
            )

        # --- Agent Execution ---
        def run_agent(state):
            """Execute the agent to decide on the next action."""
            logger.info("Running agent...")
            try:
                user_input = state["input"]

                # Safety: avoid looping too much
                if len(state.get("intermediate_steps", [])) > 2:
                    return {
                        "agent_outcome": AgentFinish(
                            return_values={"output": "I apologize, but I'm having trouble processing your request with the available tools. Please try rephrasing your question or check if you've uploaded the correct documents."},
                            log="Max steps reached"
                        )
                    }

                # Collect available docs
                available_docs = [os.path.basename(doc.metadata.get('source', 'unknown')) for doc in raw_documents]
                available_docs_str = ", ".join(available_docs) if available_docs else "No documents available"

                # Run decision chain
                chain = decision_prompt | llm | StrOutputParser()
                decision = chain.invoke({
                    "input": user_input,
                    "available_docs": available_docs_str
                })

                logger.info(f"Agent decision: {decision}")
                agent_outcome = parse_agent_response(decision, available_docs)
                return {"agent_outcome": agent_outcome}

            except Exception as e:
                logger.error(f"Error in agent execution: {str(e)}")
                return {
                    "agent_outcome": AgentFinish(
                        return_values={"output": f"I encountered an error: {str(e)}. Please try again."},
                        log="Error in agent"
                    )
                }

        # --- Tool Execution ---
        def execute_tools(state):
            agent_outcome = state.get("agent_outcome")
            if not isinstance(agent_outcome, AgentAction):
                return {"intermediate_steps": [], "confidence": 0.0}

            tool_name = agent_outcome.tool
            tool_input = agent_outcome.tool_input

            # Map tools by name once
            tool_map = {t.name: t for t in tools}

            def run_rag(q: str):
                rag_tool = tool_map.get("AnswerQuestionAboutPDFs")
                if not rag_tool:
                    return "RAG tool missing.", 0.0
                result = rag_tool.func(q)
                if isinstance(result, tuple) and len(result) == 2:
                    return result
                return str(result), 0.0

            def run_web(q: str):
                web_tool = tool_map.get("WebSearch")
                if not web_tool:
                    return "WebSearch tool missing."
                return str(web_tool.func(q))

            def run_summary(doc: str):
                s_tool = tool_map.get("SummarizePDF")
                if not s_tool:
                    return "Summarize tool missing."
                return str(s_tool.func(doc))

            def run_realtime(q: str):
                r_tool = tool_map.get("RealtimeData")
                if not r_tool:
                    return "Realtime tool missing."
                return str(r_tool.func(q))

            # Dispatch based on the agent's chosen tool
            if tool_name == "AnswerQuestionAboutPDFs":
                answer, pdf_confidence = run_rag(tool_input)
                final_outcome = AgentFinish(
                    return_values={"output": answer},
                    log=f"Used RAG (confidence={pdf_confidence})"
                )
                return {"agent_outcome": final_outcome, "intermediate_steps": [(agent_outcome, answer)], "confidence": pdf_confidence}

            elif tool_name == "SummarizePDF":
                summary = run_summary(tool_input)
                final_outcome = AgentFinish(
                    return_values={"output": summary},
                    log="Used SummarizePDF"
                )
                return {"agent_outcome": final_outcome, "intermediate_steps": [(agent_outcome, summary)], "confidence": 1.0}

            elif tool_name == "WebSearch":
                # Try to answer from PDFs first; if weak, merge with web
                pdf_answer, pdf_confidence = run_rag(tool_input)
                if pdf_confidence >= CONFIDENCE_THRESHOLD:
                    final_outcome = AgentFinish(
                        return_values={"output": pdf_answer},
                        log=f"RAG strong enough (confidence={pdf_confidence}), skipped web"
                    )
                    return {"agent_outcome": final_outcome, "intermediate_steps": [(agent_outcome, pdf_answer)], "confidence": pdf_confidence}
                else:
                    web_answer = run_web(tool_input)
                    merged = call_gemini_summary(pdf_answer, web_answer, tool_input)
                    final_outcome = AgentFinish(
                        return_values={"output": merged},
                        log=f"RAG weak (confidence={pdf_confidence}), merged with Web"
                    )
                    return {"agent_outcome": final_outcome, "intermediate_steps": [(agent_outcome, merged)], "confidence": pdf_confidence}

            elif tool_name == "RealtimeData":
                # Realtime queries should go straight to realtime tool
                rt = run_realtime(tool_input)
                final_outcome = AgentFinish(
                    return_values={"output": rt},
                    log="Used RealtimeData"
                )
                return {"agent_outcome": final_outcome, "intermediate_steps": [(agent_outcome, rt)], "confidence": 0.0}

            else:
                # Fallback: try RAG, otherwise pass through the agent text
                answer, pdf_confidence = run_rag(tool_input)
                final_outcome = AgentFinish(
                    return_values={"output": answer},
                    log=f"Fallback RAG (confidence={pdf_confidence})"
                )
                return {"agent_outcome": final_outcome, "intermediate_steps": [(agent_outcome, answer)], "confidence": pdf_confidence}

        # --- Continuation Logic ---
        def should_continue(state):
            agent_outcome = state.get("agent_outcome")
            if isinstance(agent_outcome, AgentFinish):
                logger.info("Agent finished")
                return "end"
            elif isinstance(agent_outcome, AgentAction):
                logger.info("Agent will execute tool")
                return "continue"
            else:
                logger.info("Unknown outcome, ending")
                return "end"

        # --- Build Workflow ---
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", run_agent)
        workflow.add_node("action", execute_tools)
        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"continue": "action", "end": END},
        )

        workflow.add_edge("action", END)

        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)

        logger.info("Agent compiled successfully with PDF confidence logic and Gemini merging!")
        return app

    except Exception as e:
        logger.error(f"Error compiling agent: {str(e)}")
        raise e
