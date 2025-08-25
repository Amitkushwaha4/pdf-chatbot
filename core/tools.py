# core/tools.py
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_rag_tool(vector_store, llm):
    """Creates the RAG tool for answering specific questions."""
    
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
    
    def rag_tool_function(query: str) -> str:
        """Execute RAG query and return answer."""
        try:
            logger.info(f"RAG tool processing query: {query}")
            result = rag_chain.invoke({"input": query})
            return result.get("answer", "No answer generated.")
        except Exception as e:
            logger.error(f"Error in RAG tool: {str(e)}")
            return f"Error processing query: {str(e)}"

    return Tool(
        name="AnswerQuestionAboutPDFs",
        func=rag_tool_function,
        description=(
            "Use this tool to answer specific questions about the content of uploaded PDF documents. "
            "Provide the user's question as input. This tool searches through the document content "
            "and provides detailed, contextual answers."
        )
    )

def create_summarization_tool(raw_documents, llm):
    """Creates the summarization tool."""
    
    # Create document mapping with flexible matching
    doc_map = {}
    for doc in raw_documents:
        source_name = os.path.basename(doc.metadata.get('source', 'unknown'))
        if source_name not in doc_map:
            doc_map[source_name] = []
        doc_map[source_name].append(doc)
    
    def summarize_tool_function(doc_name: str) -> str:
        """Summarize a specific document."""
        try:
            logger.info(f"Summarization tool processing: {doc_name}")
            
            # First try exact match
            if doc_name in doc_map:
                docs_to_summarize = doc_map[doc_name]
            else:
                # Try partial matching - find documents that contain the requested name
                matching_docs = []
                for available_doc, doc_list in doc_map.items():
                    # Remove file extensions for comparison
                    clean_requested = doc_name.replace('.pdf', '').lower()
                    clean_available = available_doc.replace('.pdf', '').lower()
                    
                    # Check if the clean requested name is in the available name or vice versa
                    if clean_requested in clean_available or clean_available in clean_requested:
                        matching_docs.extend(doc_list)
                        break
                
                if matching_docs:
                    docs_to_summarize = matching_docs
                else:
                    available_docs = ", ".join(doc_map.keys())
                    return f"Document '{doc_name}' not found. Available documents: {available_docs}"
            
            # Use map_reduce for longer documents
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
            "Use this tool to generate a comprehensive summary of a specific PDF document. "
            "Provide the exact filename (including .pdf extension) as input. "
            "This tool will create a detailed overview of the document's main points, "
            "key findings, and important information."
        )
    )
