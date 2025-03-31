from typing import Dict, List
from langchain import hub
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from src.AdaptiveRag.llm import get_llm
from src.AdaptiveRag.state.state import AdaptiveRAGState

class AnswerGeneratorNode:
    """Generates answers to user questions based on retrieved documents."""
    
    def __init__(self, user_input: Dict[str, str]):
        """Initialize the answer generator with user input configuration.
        
        Args:
            user_input: Dictionary containing user configuration
        """
        self.llm = get_llm(user_input=user_input)
        self.rag_chain = None
        self._get_rag_chain()
        
    def _format_docs(self, documents: List[Document]) -> str:
        """Format a list of documents into a single string.
        
        Args:
            documents: List of Document objects to format
            
        Returns:
            str: Concatenated document contents separated by double newlines
        """
        return "\n\n".join(doc.page_content for doc in documents)
    
    def _get_rag_chain(self):
        """Set up the RAG chain using a prompt from LangChain Hub.
        
        Creates a pipeline that takes context and question and generates an answer.
        """
        prompt = hub.pull("rlm/rag-prompt")
        self.rag_chain = prompt | self.llm | StrOutputParser()

    def answer_generator_node(self, state:AdaptiveRAGState):
        """Generate an answer based on the question and retrieved documents.
        
        Args:
            state: Dictionary containing question and documents
            
        Returns:
            dict: Updated state with generated answer
        """
        print("---GENERATE ANSWER---")

        question = state["question"]
        documents = state["documents"]

        generation = self.rag_chain.invoke({
            "context": self._format_docs(documents), 
            "question": question
        })
        
        return {
            "documents": documents, 
            "question": question, 
            "generation": generation
        }