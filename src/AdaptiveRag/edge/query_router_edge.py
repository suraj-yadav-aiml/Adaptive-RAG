from pydantic import BaseModel, Field
from typing import Literal, Dict
from langchain_core.prompts import ChatPromptTemplate
from src.AdaptiveRag.llm import get_llm

class QueryRouter(BaseModel):
    """Model for routing a user query to the most relevant data source."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        description="Route to 'vectorstore' for stored embeddings or 'web_search' for real-time information."
    )

class QueryRouterEdge:
    """Routes user queries to either vectorstore or web search based on the query content."""
    
    # Topics available in the vectorstore
    VECTORSTORE_TOPICS = ["Agents", "Prompt Engineering", "Adversarial Attacks"]
    
    def __init__(self, user_input: Dict[str, str]):
        """Initialize the router with user input configuration.
        
        Args:
            user_input: Dictionary containing user configuration
        """
        self.llm = get_llm(user_input=user_input)
        self.structured_llm_router = self.llm.with_structured_output(QueryRouter)
        self.query_router = None
        self._setup_router()
    
    def _setup_router(self):
        """Set up the query router with system prompt and template."""
        system = f"""You are an expert in intelligently routing user queries to the most appropriate data source.

        The vectorstore contains documents specifically related to the following topics:
        {', '.join(self.VECTORSTORE_TOPICS)}

        For queries related to these topics, use the vectorstore.
        For all other queries, use web search to fetch real-time information from the internet."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{question}")
        ])
        
        self.query_router = prompt | self.structured_llm_router
    
    def query_router_edge(self, state):
        """Route the query based on its content.
        
        Args:
            state: Dictionary containing the current state with the question
            
        Returns:
            str: Either "vectorstore" or "web_search" based on routing decision
        """
        print("---ROUTE QUESTION---")
        
        question = state["question"]
        source = self.query_router.invoke({"question": question})
        
        if source.datasource == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"