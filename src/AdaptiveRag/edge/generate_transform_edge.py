from typing import Literal
from src.AdaptiveRag.state import AdaptiveRAGState

class GenerateOrRewriterEdge:
    """Decision node that determines whether to generate an answer or rewrite the question."""
    
    def generate_or_rewriter_node(self, state: AdaptiveRAGState) -> Literal["question_rewriter_node", "answer_generator_node"]:
        """Determine the next step based on document relevance assessment.
        
        If there are relevant documents, proceed to answer generation.
        If no relevant documents, route to question rewriting to improve retrieval.
        
        Args:
            state: Dictionary containing the current state with filtered documents
            
        Returns:
            str: The name of the next node to route to in the pipeline
        """
        print("---ASSESS DOCUMENT RELEVANCE FOR ROUTING DECISION---")

        filtered_documents = state.get("documents", [])
        
        if not filtered_documents:
            print("---DECISION: NO RELEVANT DOCUMENTS FOUND, REWRITING QUERY---")
            return "question_rewriter_node"
        
        doc_count = len(filtered_documents)
        print(f"---DECISION: {doc_count} RELEVANT DOCUMENTS FOUND, GENERATING ANSWER---")
        return "answer_generator_node"