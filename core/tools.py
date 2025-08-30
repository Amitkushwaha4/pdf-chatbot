# core/tools.py
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from tavily import TavilyClient
import os
import logging
import requests
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.3


def create_rag_tool(vector_store, llm):
    """RAG tool that returns (answer_text, confidence)."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    qa_system_prompt = (
        "You are an AI assistant specialized in analyzing and answering questions about PDF documents. "
        "Use the following retrieved context to provide accurate, detailed answers. "
        "If the information is not available in the context, clearly state that you don't know.\n\n"
        "Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        ("human", "{input}"),
    ])

    # Not directly used for the returned text; we compute confidence from similarity.
    _question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    _rag_chain = create_retrieval_chain(retriever, _question_answer_chain)

    def rag_tool_function(query: str, chat_history=None) -> tuple[str, float]:
        history_text = ""
        if chat_history:
            history_text = "\n".join([f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}" for m in chat_history])
        full_query = f"{history_text}\nCurrent question: {query}" if history_text else query

        try:
            logger.info(f"RAG tool processing query: {query}")
            results = vector_store.similarity_search_with_score(query, k=5)

            if results:
                docs, scores = zip(*results)
                # Concatenate top chunks as answer context
                answer_text = "\n".join([doc.page_content for doc in docs])
                # scores are distances → lower = closer
                best_score = min(scores)
                # confidence in [0,1)
                confidence = 1.0 / (1.0 + float(best_score))
            else:
                answer_text, confidence = "No relevant information found in PDFs.", 0.0

            return answer_text, float(confidence)

        except Exception as e:
            logger.error(f"Error in RAG tool: {str(e)}")
            return f"Error processing query: {str(e)}", 0.0

    return Tool(
        name="AnswerQuestionAboutPDFs",
        func=rag_tool_function,
        description=(
            "Answer specific questions using the uploaded PDFs. "
            "Returns a tuple (answer, confidence[0..1])."
        )
    )


def create_summarization_tool(raw_documents, llm):
    """Summarize a specific PDF by filename (or partial match)."""
    doc_map = {}
    for doc in raw_documents:
        source_name = os.path.basename(doc.metadata.get('source', 'unknown'))
        doc_map.setdefault(source_name, []).append(doc)

    def summarize_tool_function(doc_name: str) -> str:
        try:
            logger.info(f"Summarization tool processing: {doc_name}")
            if doc_name in doc_map:
                docs_to_summarize = doc_map[doc_name]
            else:
                # Partial match
                matching_docs = []
                clean_requested = doc_name.replace('.pdf', '').lower()
                for available_doc, doc_list in doc_map.items():
                    clean_available = available_doc.replace('.pdf', '').lower()
                    if clean_requested in clean_available or clean_available in clean_requested:
                        matching_docs.extend(doc_list)
                        break
                if matching_docs:
                    docs_to_summarize = matching_docs
                else:
                    available_docs = ", ".join(doc_map.keys())
                    return f"Document '{doc_name}' not found. Available documents: {available_docs}"

            summarize_chain = load_summarize_chain(
                llm,
                chain_type="map_reduce",
                verbose=True
            )
            summary_result = summarize_chain.invoke({"input_documents": docs_to_summarize})
            return summary_result.get('output_text', 'No summary generated.')

        except Exception as e:
            logger.error(f"Error in summarization tool: {str(e)}")
            return f"Error summarizing document: {str(e)}"

    return Tool(
        name="SummarizePDF",
        func=summarize_tool_function,
        description="Generate a summary of a PDF. Provide the filename (supports partial match)."
    )


def create_web_search_tool():
    """Search the web via Tavily. Safe against None values."""
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def web_search_tool_function(query: str) -> str:
        try:
            logger.info(f"Web search tool processing query: {query}")
            result = tavily_client.search(query)
            if not result:
                return "No web answer found."

            answer_text = ""
            urls = []
            summary_points = []

            if isinstance(result, dict):
                raw_answer = result.get("answer")
                answer_text = (raw_answer or "").strip()

                results_list = result.get("results") or []
                if isinstance(results_list, list):
                    for item in results_list:
                        if not isinstance(item, dict):
                            continue
                        url = item.get("url")
                        content = item.get("content")
                        if url:
                            urls.append(str(url))
                        if content:
                            summary_points.append(str(content).strip())

                if not answer_text and summary_points:
                    # Take top 2-3 snippets to avoid verbosity
                    answer_text = " ".join(summary_points[:3])

                if answer_text and urls:
                    answer_text += "\n\nSources:\n" + "\n".join(urls)

            else:
                # Fallback stringify
                answer_text = str(result)

            return answer_text if (answer_text and answer_text.strip()) else "No useful information found."
        except Exception as e:
            logger.error(f"Error in web search tool: {str(e)}")
            return f"Error searching the web: {str(e)}"

    return Tool(
        name="WebSearch",
        func=web_search_tool_function,
        description="Search the internet for information not found in PDFs."
    )



def call_gemini_summary(pdf_answer: str, web_answer: str, query: str) -> str:
    """Merge PDF + Web info into one concise answer using Gemini 1.5 Flash."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client()  # GEMINI_API_KEY from env
        prompt = (
            f"User query: {query}\n\n"
            f"From PDF:\n{pdf_answer}\n\n"
            f"From Web:\n{web_answer}\n\n"
            f"Task: Combine into one clear, concise answer. "
            f"If they disagree, note both briefly."
        )
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        return getattr(response, "text", str(response)).strip()
    except Exception as e:
        logger.error(f"Gemini summarization failed: {e}")
        # Safe fallback: return both parts
        return f"PDF:\n{pdf_answer}\n\nWeb:\n{web_answer}"
