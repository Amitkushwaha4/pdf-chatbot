from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from core.agent_state import AgentState
from core.tools import create_rag_tool, create_summarization_tool
import re
import os
import logging
from config.config import load_api_key

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def compile_agent(vector_store, raw_documents):
    """
    Builds and compiles a custom LangGraph agent with simplified logic.
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
            create_summarization_tool(raw_documents, llm)
        ]
        
        # Create a simple decision-making prompt
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that helps users with PDF document analysis. 

Available tools:
1. AnswerQuestionAboutPDFs - Use this for answering specific questions about document content
2. SummarizePDF - Use this to summarize specific PDF documents

Available documents: {available_docs}

Analyze the user's request and decide what to do:

If the user is asking a specific question about document content (like "what are the skills", "what companies", etc.), respond with:
ACTION: AnswerQuestionAboutPDFs
INPUT: [the user's question]

If the user wants a summary, overview, or general information about a document, respond with:
ACTION: SummarizePDF  
INPUT: [pick the most relevant document from available docs, or if only one document, use that]

Examples:
- "summarize the document" → ACTION: SummarizePDF, INPUT: [first available document]
- "give me an overview" → ACTION: SummarizePDF, INPUT: [first available document] 
- "what are the main points" → ACTION: SummarizePDF, INPUT: [first available document]
- "summarize resume.pdf" → ACTION: SummarizePDF, INPUT: resume.pdf

If you can answer directly without tools, respond with:
ACTION: DIRECT
INPUT: [your direct answer]

User request: {input}

Decision:"""),
        ])
        
        def parse_agent_response(response_text, available_docs):
            """Parse the agent's decision from the response text."""
            if "ACTION: AnswerQuestionAboutPDFs" in response_text:
                input_match = re.search(r"INPUT: (.+)", response_text)
                if input_match:
                    return AgentAction(
                        tool="AnswerQuestionAboutPDFs",
                        tool_input=input_match.group(1).strip(),
                        log=response_text
                    )
            
            elif "ACTION: SummarizePDF" in response_text:
                input_match = re.search(r"INPUT: (.+)", response_text)
                if input_match:
                    # If the input doesn't contain .pdf, try to match with available docs
                    tool_input = input_match.group(1).strip()
                    
                    # If no specific filename provided, use the first available document
                    if not tool_input.endswith('.pdf') and available_docs:
                        tool_input = available_docs[0]
                    # If still no .pdf extension and we have docs, try to find a match
                    elif not tool_input.endswith('.pdf') and available_docs:
                        # Look for partial matches
                        for doc in available_docs:
                            if tool_input.lower() in doc.lower():
                                tool_input = doc
                                break
                        else:
                            tool_input = available_docs[0]  # Default to first doc
                    
                    return AgentAction(
                        tool="SummarizePDF",
                        tool_input=tool_input,
                        log=response_text
                    )
            
            elif "ACTION: DIRECT" in response_text:
                input_match = re.search(r"INPUT: (.+)", response_text, re.DOTALL)
                if input_match:
                    return AgentFinish(
                        return_values={"output": input_match.group(1).strip()},
                        log=response_text
                    )
            
            # If parsing fails, try to answer directly
            return AgentFinish(
                return_values={"output": response_text},
                log="Direct response"
            )
        
        def run_agent(state):
            """Execute the agent to decide on the next action."""
            logger.info("Running agent...")
            
            try:
                # Get the user input
                user_input = state["input"]
                
                # If we've already tried tools and failed, give a direct response
                if len(state.get("intermediate_steps", [])) > 2:
                    return {
                        "agent_outcome": AgentFinish(
                            return_values={"output": "I apologize, but I'm having trouble processing your request with the available tools. Please try rephrasing your question or check if you've uploaded the correct documents."},
                            log="Max steps reached"
                        )
                    }
                
                # Get available document names
                available_docs = []
                for doc in raw_documents:
                    doc_name = os.path.basename(doc.metadata.get('source', 'unknown'))
                    if doc_name not in available_docs:
                        available_docs.append(doc_name)
                
                available_docs_str = ", ".join(available_docs) if available_docs else "No documents available"
                
                # Create decision chain
                chain = decision_prompt | llm | StrOutputParser()
                
                # Get agent decision
                decision = chain.invoke({
                    "input": user_input,
                    "available_docs": available_docs_str
                })
                logger.info(f"Agent decision: {decision}")
                
                # Parse the decision
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
        
        def execute_tools(state):
            """Execute the selected tool."""
            logger.info("Executing tools...")
            
            agent_outcome = state.get("agent_outcome")
            if not isinstance(agent_outcome, AgentAction):
                return {"intermediate_steps": []}
            
            tool_name = agent_outcome.tool
            tool_input = agent_outcome.tool_input
            
            # Find and execute the tool
            for tool in tools:
                if tool.name == tool_name:
                    try:
                        logger.info(f"Executing {tool_name} with input: {tool_input}")
                        result = tool.func(tool_input)
                        
                        # Create the final answer
                        final_outcome = AgentFinish(
                            return_values={"output": result},
                            log=f"Tool {tool_name} executed successfully"
                        )
                        
                        return {
                            "agent_outcome": final_outcome,
                            "intermediate_steps": [(agent_outcome, result)]
                        }
                        
                    except Exception as e:
                        logger.error(f"Error executing {tool_name}: {str(e)}")
                        error_outcome = AgentFinish(
                            return_values={"output": f"Error using {tool_name}: {str(e)}"},
                            log=f"Tool error: {str(e)}"
                        )
                        return {
                            "agent_outcome": error_outcome,
                            "intermediate_steps": [(agent_outcome, f"Error: {str(e)}")]
                        }
            
            # Tool not found
            error_outcome = AgentFinish(
                return_values={"output": f"Tool {tool_name} not found."},
                log="Tool not found"
            )
            return {
                "agent_outcome": error_outcome,
                "intermediate_steps": [(agent_outcome, "Tool not found")]
            }
        
        def should_continue(state):
            """Determine whether to continue or finish."""
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
        
        # Build the workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", run_agent)
        workflow.add_node("action", execute_tools)
        
        workflow.set_entry_point("agent")
        
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        
        workflow.add_edge("action", END)
        
        # Use MemorySaver for persistence
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        
        logger.info("Agent compiled successfully!")
        return app
        
    except Exception as e:
        logger.error(f"Error compiling agent: {str(e)}")
        raise e