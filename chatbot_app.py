# chatbot_app.py
import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.agents import AgentFinish 
from services.document_processor import process_pdf
from services.vector_store import create_vector_store
from core.agent_builder import compile_agent
import tempfile
import os
import logging 

# --- Basic Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Page Configuration ---
st.set_page_config(page_title="InsightAgent", layout="wide")
st.title("🤖 InsightAgent: Agentic PDF Chatbot")

# --- Session State Initialization ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = {st.session_state.thread_id: "New Chat"}
if "raw_documents" not in st.session_state:
    st.session_state.raw_documents = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# --- Thread-based memory ---
if "chat_history_by_thread" not in st.session_state:
    st.session_state.chat_history_by_thread = {}

# Ensure current thread has a history list
thread_id = st.session_state.thread_id
st.session_state.chat_history_by_thread.setdefault(thread_id, [])

# --- Helper Functions ---
def get_thread_name(thread_id):
    """Get a meaningful name for the chat thread"""
    return st.session_state.chat_threads.get(thread_id, "Unknown Chat")

def create_new_thread():
    """Create a new chat thread"""
    new_thread_id = str(uuid.uuid4())
    st.session_state.chat_threads[new_thread_id] = "New Chat"
    st.session_state.thread_id = new_thread_id
    st.session_state.chat_history_by_thread.setdefault(new_thread_id, [])
    return new_thread_id

# --- Sidebar ---
with st.sidebar:
    
    # New Chat Button
    if st.button("New Chat", use_container_width=True):
        new_thread_id = create_new_thread()
        logging.info(f"Created new chat thread: {new_thread_id}")
        st.rerun()

    st.divider()
    
    # Document Upload Section
    st.header("📁 Upload Documents")
    pdf_docs = st.file_uploader(
        "Upload PDF files", 
        accept_multiple_files=True, 
        type="pdf",
        help="Select one or more PDF files to process"
    )

    if st.button("🔄 Process Documents", use_container_width=True) and pdf_docs:
        with st.spinner("Processing documents into an agent..."):
            try:
                temp_files = []
                doc_names = [file.name for file in pdf_docs]
                logging.info(f"Starting document processing for: {', '.join(doc_names)}")
                
                # Save uploaded files temporarily
                for uploaded_file in pdf_docs:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix=uploaded_file.name) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_files.append(tmp_file.name)

                # Process documents
                st.session_state.raw_documents = process_pdf(temp_files)
                st.session_state.processed_files = doc_names
                
                # Create vector store and agent (persistent for all threads)
                vector_store = create_vector_store(st.session_state.raw_documents)
                st.session_state.agent = compile_agent(vector_store, st.session_state.raw_documents)

                # Clean up temp files
                for file_path in temp_files:
                    try:
                        os.remove(file_path)
                    except:
                        pass
                        
                st.success(f"✅ Processed {len(doc_names)} document(s) successfully!")
                logging.info("Document processing successful.")
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                logging.error(f"Error during document processing: {str(e)}", exc_info=True)
    
    st.divider()

    # Chat History Section
    if st.session_state.agent and len(st.session_state.chat_threads) > 1:
        st.header("💬 Chat History")
        
        # Display chat threads
        for t_id, thread_name in st.session_state.chat_threads.items():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button(
                    f"🗨️ {thread_name}", 
                    key=f"thread_{t_id}",
                    use_container_width=True,
                    type="primary" if t_id == st.session_state.thread_id else "secondary"
                ):
                    st.session_state.thread_id = t_id
                    st.session_state.chat_history_by_thread.setdefault(t_id, [])
                    logging.info(f"Switched to thread: {t_id}")
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"delete_{t_id}", help="Delete thread"):
                    if len(st.session_state.chat_threads) > 1:
                        logging.warning(f"Deleting thread: {t_id}")
                        del st.session_state.chat_threads[t_id]
                        if t_id == st.session_state.thread_id:
                            st.session_state.thread_id = list(st.session_state.chat_threads.keys())[0]
                        st.rerun()

# --- Main Chat Interface ---
if st.session_state.agent is None:
    st.info("👆 Please upload and process PDF documents in the sidebar to begin chatting!")
    
    # Show example queries
    st.markdown("### 💡 What you can do:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Ask Questions:**
        - "What is the main topic of the document?"
        - "Explain the methodology used in chapter 3"
        - "What are the key findings?"
        """)
    
    with col2:
        st.markdown("""
        **Get Summaries:**
        - "Summarize document.pdf"
        - "Give me an overview of the research paper"
        - "What are the main points discussed?"
        """)

else:
    # Display current thread info
    thread_id = st.session_state.thread_id  # Always use latest
    current_thread_name = get_thread_name(thread_id)
    st.caption(f"💬 Current chat: {current_thread_name}")
    
    # Load and display chat history
    current_history = st.session_state.chat_history_by_thread.get(thread_id, [])

    for msg in current_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # Chat input
    user_query = st.chat_input("Ask a question or request a summary...")
    
    if user_query:
        # Update thread name if it's still "New Chat"
        if get_thread_name(thread_id) == "New Chat":
            thread_name = user_query[:30] + "..." if len(user_query) > 30 else user_query
            st.session_state.chat_threads[thread_id] = thread_name
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Process with agent
        with st.chat_message("assistant"):
            with st.spinner("thinking..."):
                try:
                    config = {"configurable": {"thread_id": thread_id}}
                    
                    logging.info(f"Invoking agent for thread '{thread_id}' with query: '{user_query}'")
                    
                    # Prepare input data
                    input_data = {
                        "input": user_query,
                        "chat_history": current_history 
                    }
                    
                    # Get agent response
                    final_state = st.session_state.agent.invoke(input_data, config=config)
                    
                    # Extract response and confidence
                    agent_outcome = final_state.get('agent_outcome')
                    confidence = final_state.get('confidence', 0.0)
                    
                    if agent_outcome and isinstance(agent_outcome, AgentFinish):
                        final_answer = agent_outcome.return_values.get('output', 'No response generated.')
                        
                        # Display response
                        st.markdown(final_answer)
                        
                        # Display confidence
                        if confidence > 0:
                            st.markdown(f"**📊 PDF Confidence Level:** {confidence*100:.2f}%")
                        
                        # --- Thread-based memory update ---
                        st.session_state.chat_history_by_thread.setdefault(thread_id, [])
                        st.session_state.chat_history_by_thread[thread_id].extend([
                            HumanMessage(content=user_query),
                            AIMessage(content=final_answer)
                        ])

                        # Update AgentState with full history
                        full_history = st.session_state.chat_history_by_thread[thread_id]
                        st.session_state.agent.update_state(config, {"chat_history": full_history})

                        logging.info("Agent invocation successful.")
                    else:
                        st.error("❌ Agent did not provide a proper response.")
                        logging.warning(f"Agent finished with unexpected outcome: {agent_outcome}")
                            
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
                    st.exception(e)
                    logging.error(f"Error during agent invocation: {str(e)}", exc_info=True)
        
        st.rerun()
