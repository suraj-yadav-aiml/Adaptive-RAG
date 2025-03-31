from typing import Dict, List
import streamlit as st
from src.AdaptiveRag.retriever import Retriever
from langchain.schema import Document
from src.AdaptiveRag.state.state import AdaptiveRAGState

class RetrieverNode:
    """Retrieves relevant documents based on a user question."""
    
    def __init__(self, user_input: Dict[str, str]):
        """Initialize the retriever node with user input configuration.
        
        Args:
            user_input: Dictionary containing user configuration
        """
        self._initialize_retriever(user_input)
    
    def _initialize_retriever(self, user_input: Dict[str, str]):
        """Initialize or retrieve the retriever from session state.
        
        Creates a new retriever if one doesn't exist in the session state,
        otherwise uses the existing one for consistency across reruns.
        
        Args:
            user_input: Dictionary containing user configuration
        """
        if "retriever" not in st.session_state:
            st.session_state["retriever"] = Retriever(user_input).get_retriever()
        
        self.retriever = st.session_state["retriever"]
    
    def retriever_node(self, state:AdaptiveRAGState):
        """Retrieve documents relevant to the question.
        
        Args:
            state: Dictionary containing the current state with the question
            
        Returns:
            dict: Updated state with retrieved documents
        """
        print("---RETRIEVE DOCUMENTS---")

        question = state["question"]
        documents = self.retriever.invoke(question)
        
        return {
            "documents": documents, 
            "question": question
        }