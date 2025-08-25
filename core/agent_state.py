# core/agent_state.py
from typing import List, TypedDict, Annotated, Sequence, Union
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator
from langchain_core.agents import AgentAction, AgentFinish

class AgentState(TypedDict):
    """
    Represents the state of our agent.
    
    Attributes:
        input: The user's original question
        chat_history: The conversation history
        agent_outcome: The outcome of the agent's decision
        intermediate_steps: A sequence of (action, observation) pairs
    """
    input: str
    chat_history: Annotated[List[BaseMessage], add_messages]
    agent_outcome: Union[List[AgentAction], AgentFinish, None]
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]