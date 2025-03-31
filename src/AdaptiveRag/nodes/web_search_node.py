from typing import Dict, List
from langchain.schema import Document
from src.AdaptiveRag.tools import get_tool_mananger
from src.AdaptiveRag.state.state import AdaptiveRAGState

class WebSearchNode:
    """Performs web searches and formats results as Document objects for the RAG pipeline."""
    
    def __init__(self):
        """Initialize the web search node with the web search tool."""
        self.web_search_tool = get_tool_mananger().get_tool("web_search")
        
        if self.web_search_tool is None:
            raise ValueError("Web search tool not found. Please ensure it's properly registered.")
    
    def web_search_node(self, state: AdaptiveRAGState):
        """Execute a web search based on the user question.
        
        Args:
            state: Dictionary containing the current state with the question
            
        Returns:
            dict: Updated state with search results as documents
        """
        print("---PERFORMING WEB SEARCH---")

        question = state["question"]
        
        search_results = self.web_search_tool.invoke({"query": question})
        
        documents = [Document(page_content=doc["content"]) for doc in search_results]

        return {
            "documents": documents,
            "question": question
        }