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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_rag_tool(vector_store, llm):
    """Creates the RAG tool for answering specific questions with confidence tracking."""

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    qa_system_prompt = (
        "You are an AI assistant specialized in analyzing and answering questions about PDF documents. "
        "Use the following retrieved context to provide accurate, detailed answers. "
        "If the information is not available in the context, clearly state that you don't know. "
        "Be precise, informative, and cite specific details when possible.\n\n"
        "Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    def rag_tool_function(query: str) -> tuple[str, float]:
        """
        Return a tuple (answer, confidence) based on retrieved chunks.
        Confidence = max similarity among top retrieved chunks.
        """
        try:
            logger.info(f"RAG tool processing query: {query}")
            response = rag_chain.invoke({"input": query})

            # Default answer & confidence
            answer_text = response.get("answer", response.get("output_text", "No answer generated."))
            confidence = 0.0

            # Compute max similarity from retrieved source_documents
            if "source_documents" in response:
                docs = response["source_documents"]
                if docs:
                    similarities = []
                    for doc in docs:
                        sim = 0.0
                        if hasattr(doc, "metadata"):
                            if "similarity" in doc.metadata:
                                sim = float(doc.metadata["similarity"])
                            elif "score" in doc.metadata:
                                sim = float(doc.metadata["score"])
                        similarities.append(sim)
                    if similarities:
                        confidence = max(similarities)
            # Ensure confidence is 0–1
            confidence = min(max(confidence, 0.0), 1.0)

            return answer_text, confidence

        except Exception as e:
            logger.error(f"Error in RAG tool: {str(e)}")
            return f"Error processing query: {str(e)}", 0.0

    return Tool(
        name="AnswerQuestionAboutPDFs",
        func=rag_tool_function,
        description=(
            "Use this tool to answer specific questions about PDF content. "
            "Returns a tuple (answer, confidence) where confidence is based on retrieved chunk similarity."
        )
    )


def create_summarization_tool(raw_documents, llm):
    """Creates the summarization tool."""

    # Create document mapping
    doc_map = {}
    for doc in raw_documents:
        source_name = os.path.basename(doc.metadata.get('source', 'unknown'))
        doc_map.setdefault(source_name, []).append(doc)

    def summarize_tool_function(doc_name: str) -> str:
        """Summarize a specific document."""
        try:
            logger.info(f"Summarization tool processing: {doc_name}")
            if doc_name in doc_map:
                docs_to_summarize = doc_map[doc_name]
            else:
                # Partial match
                matching_docs = []
                for available_doc, doc_list in doc_map.items():
                    clean_requested = doc_name.replace('.pdf', '').lower()
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
        description=(
            "Use this tool to generate a summary of a PDF document. "
            "Provide the exact filename (including .pdf extension)."
        )
    )


def create_web_search_tool():
    """Creates a tool for searching the web using Tavily."""
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def web_search_tool_function(query: str) -> str:
        try:
            logger.info(f"Web search tool processing query: {query}")
            result = tavily_client.search(query)
            answer_text = ""
            urls = []

            if isinstance(result, dict):
                answer_text = result.get("answer", "").strip()
                summary_points = []
                if "results" in result and isinstance(result["results"], list):
                    for item in result["results"]:
                        if isinstance(item, dict):
                            if "url" in item:
                                urls.append(item["url"])
                            if "content" in item and item["content"]:
                                summary_points.append(item["content"].strip())
                if not answer_text and summary_points:
                    answer_text = " ".join(summary_points[:3])
                if answer_text:
                    answer_text = f"{answer_text}\n\nSources:\n" + "\n".join(urls) if urls else answer_text
                else:
                    answer_text = "No useful information found."
            else:
                answer_text = str(result)
            return answer_text if answer_text.strip() else "No web answer found."

        except Exception as e:
            logger.error(f"Error in web search tool: {str(e)}")
            return f"Error searching the web: {str(e)}"

    return Tool(
        name="WebSearch",
        func=web_search_tool_function,
        description=(
            "Use this tool to search the internet for information not found in the uploaded PDF documents."
        )
    )
def create_realtime_tool():
    """Tool for fetching real-time information like weather."""

    import re

    city_pat = re.compile(r"(?:weather\s+(?:in|of)\s+)(?P<city>[a-zA-Z\s\-\.\,]+)$", re.IGNORECASE)

    def extract_city(q: str) -> str | None:
        m = city_pat.search(q.strip())
        if m:
            return m.group("city").strip(" .,")

        # fallback heuristics: last token after 'weather'
        if "weather" in q.lower():
            tail = q.lower().split("weather")[-1]
            tail = tail.replace("in", "").replace("of", "").strip(" .,:;")
            if tail:
                return tail.title()
        return None

    def realtime_tool_function(query: str) -> str:
        try:
            logger.info(f"Realtime tool processing query: {query}")

            ql = query.lower()
            if "weather" in ql:
                city = extract_city(query)
                if not city:
                    return "Please specify a city, e.g., 'weather in Kathmandu'."

                api_key = os.getenv("OPENWEATHER_API_KEY")
                if not api_key:
                    return "Realtime weather API key is not configured. Set OPENWEATHER_API_KEY in your environment."

                url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={api_key}"
                try:
                    resp = requests.get(url, timeout=10)
                    data = resp.json()
                except Exception as e:
                    return f"Network error while contacting weather service: {e}"

                if data.get("cod") == 200 and data.get("main"):
                    temp = data["main"].get("temp")
                    desc = data.get("weather", [{}])[0].get("description", "unknown conditions")
                    return f"The current temperature in {city.title()} is {temp}°C with {desc}."
                else:
                    msg = data.get("message") or "Unknown error."
                    return f"Could not fetch weather for {city.title()}: {msg}"

            # Add more dynamic queries here (stocks, live news, etc.)
            return "No real-time data found for your query."

        except Exception as e:
            logger.error(f"Error in Realtime tool: {str(e)}")
            return f"Error fetching real-time data: {str(e)}"

    return Tool(
        name="RealtimeData",
        func=realtime_tool_function,
        description="Use this tool for real-time information like weather, stock prices, or live news."
    )
