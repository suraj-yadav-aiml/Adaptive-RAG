from typing import List, Dict, Any, Literal
from langchain.schema import Document
from src.AdaptiveRag.state import AdaptiveRAGState
from src.AdaptiveRag.edge.hallucination_grader import HallucinationGrader
from src.AdaptiveRag.edge.answer_grader import AnswerGrader

class HallucinationAnswerEdge:
    """Evaluates generated answers for hallucinations and relevance to the question."""
    
    def __init__(self, user_input: Dict[str, str]):
        """Initialize the edge evaluator with graders.
        
        Args:
            user_input: Dictionary containing user configuration
        """
        self.hallucination_grader = HallucinationGrader(user_input).get_hallucination_grader()
        self.answer_grader = AnswerGrader(user_input).get_answer_grader()
    
    def _format_docs(self, documents: List[Document]) -> str:
        """Format a list of documents into a single string.
        
        Args:
            documents: List of Document objects to format
            
        Returns:
            str: Concatenated document contents separated by double newlines
        """
        return "\n\n".join(document.page_content for document in documents)

    def hallucination_answer_edge(self, state: AdaptiveRAGState) -> Literal["useful", "not useful", "not supported"]:
        """Evaluate answer quality and determine the next routing step.
        
        Performs two evaluations:
        1. Whether the generation is factually grounded in the documents
        2. If factual, whether it adequately addresses the question
        
        Args:
            state: Current pipeline state with question, documents, and generation
            
        Returns:
            str: Routing decision - "useful", "not useful", or "not supported"
        """
        print("---EVALUATING ANSWER FOR HALLUCINATIONS AND RELEVANCE---")
        
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        # Step 1: Check for hallucinations
        print("---CHECKING IF ANSWER IS FACTUALLY GROUNDED---")
        hallucination_score = self.hallucination_grader.invoke({
            "documents": self._format_docs(documents), 
            "generation": generation
        })
        
        # If the answer contains hallucinations, route for regeneration
        if hallucination_score.binary_score != "yes":
            print("---DECISION: ANSWER CONTAINS UNSUPPORTED INFORMATION---")
            return "not supported"
        
        # Step 2: Check if the answer addresses the question
        print("---ANSWER IS FACTUAL, CHECKING IF IT ADDRESSES THE QUESTION---")
        answer_score = self.answer_grader.invoke({
            "question": question, 
            "generation": generation
        })
        
        # Determine final routing based on whether answer addresses question
        if answer_score.binary_score == "yes":
            print("---DECISION: ANSWER IS FACTUAL AND ADDRESSES THE QUESTION---")
            return "useful"
        else:
            print("---DECISION: ANSWER IS FACTUAL BUT DOES NOT ADDRESS THE QUESTION---")
            return "not useful"