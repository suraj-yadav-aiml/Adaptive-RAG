from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.AdaptiveRag.llm import get_llm
from src.AdaptiveRag.state.state import AdaptiveRAGState

class QuestionRewriterNode:
    """Optimizes user questions to improve vector database retrieval accuracy."""
    
    def __init__(self, user_input: Dict[str, str]):
        """Initialize the question rewriter with user input configuration.
        
        Args:
            user_input: Dictionary containing user configuration
        """
        self.llm = get_llm(user_input=user_input)
        self.question_rewriter = None
        self._get_question_rewriter_chain()
        
    def _get_question_rewriter_chain(self):
        """Set up the question rewriting chain.
        
        Creates a pipeline that takes a user question and optimizes it
        for better vector retrieval performance.
        """
        system = """You are an expert question rewriter specializing in optimizing queries for vectorstore retrieval.

        Task:
        - Convert an input question into a more effective version that enhances retrieval accuracy.
        - Analyze the underlying semantic intent and meaning to refine the question.
        - Ensure the rewritten question is clear, specific, and optimized for retrieving relevant results from the vectorstore.
        """

        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Refine the following question to improve its effectiveness for vectorstore retrieval:\n\n"
                    "Original Question: {question}\n\n"
                    "Provide an optimized version of the question. I want only the optimized version of the question nothing else no explanation needed."),
        ])

        self.question_rewriter = rewrite_prompt | self.llm | StrOutputParser()
    
    def question_rewriter_node(self, state:AdaptiveRAGState):
        """Rewrite the user question to optimize it for vector retrieval.
        
        Args:
            state: Dictionary containing the current state with the original question
            
        Returns:
            dict: Updated state with optimized question
        """
        print("---TRANSFORM QUERY FOR BETTER RETRIEVAL---")

        original_question = state["question"]
        better_question = self.question_rewriter.invoke({"question": original_question})
        
        return {
            'question': better_question
        }