from pydantic import BaseModel, Field
from typing import Literal, Dict, List
from langchain_core.prompts import ChatPromptTemplate
from src.AdaptiveRag.llm import get_llm
from src.AdaptiveRag.state.state import AdaptiveRAGState

class GradeDocument(BaseModel):
    """Model for assigning a binary relevance score to retrieved documents."""

    binary_score: str = Field(
        description="Document relevance: 'yes' (relevant) or 'no' (not relevant)."
    )

class DocumentGraderNode:
    """Evaluates and filters documents based on their relevance to a user question."""
    
    RELEVANT = "yes"
    NOT_RELEVANT = "no"
    
    def __init__(self, user_input: Dict[str, str]):
        """Initialize the document grader with user input configuration.
        
        Args:
            user_input: Dictionary containing user configuration
        """
        self.llm = get_llm(user_input=user_input)
        self.structured_llm_grader = self.llm.with_structured_output(GradeDocument)
        self.retrieval_grader = None
        self._get_document_grader()
    
    def _get_document_grader(self):
        """Create and configure the document grading prompt template.
        
        Returns:
            A configured grader pipeline that evaluates document relevance
        """
        system = """You are an expert grader evaluating the relevance of a retrieved document to a user question.

        Criteria for grading:
        - If the document contains keywords, phrases, or semantic meaning related to the user question, classify it as relevant.
        - The evaluation does not need to be overly strict; the primary goal is to filter out clearly incorrect retrievals.
        - Provide a binary score:
        - "yes" → Relevant to the question.
        - "no" → Not relevant to the question."""

        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Evaluate the relevance of the following document: \n\n"
                    "Retrieved Document: \n{document}\n\n"
                    "User Question: {question}\n\n"
                    "Provide a binary score ('yes' or 'no')."),
        ])

        self.retrieval_grader = grade_prompt | self.structured_llm_grader
    
    def document_grader_node(self, state:AdaptiveRAGState):
        """Filter documents based on their relevance to the question.
        
        Args:
            state: Dictionary containing the current state with question and documents
            
        Returns:
            dict: Updated state with filtered documents
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

        question = state["question"]
        documents = state["documents"]  # List[Document]

        # Score each document and filter based on relevance
        filtered_docs = []
        for doc in documents:
            score = self.retrieval_grader.invoke({
                "question": question, 
                "document": doc.page_content
            })
            
            if score.binary_score == self.RELEVANT:
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            elif score.binary_score == self.NOT_RELEVANT:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
            
        return {"documents": filtered_docs, "question": question}