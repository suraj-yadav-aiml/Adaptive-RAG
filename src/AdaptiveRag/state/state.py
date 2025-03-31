from langchain.schema import Document
from typing import TypedDict, List, Optional

class AdaptiveRAGState(TypedDict):
    """Represents the state of the adaptive RAG processing pipeline.
    
    This state object is passed between nodes in the RAG graph and tracks
    the question, retrieved documents, and generated answers throughout
    the processing pipeline.
    
    Attributes:
        question: The user's original question or query
        documents: List of retrieved documents relevant to the question
        generation: The final generated answer based on the documents
    """
    
    question: str
    documents: List[Document]
    generation: Optional[str]