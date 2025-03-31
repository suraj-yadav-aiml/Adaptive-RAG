from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.AdaptiveRag.llm import get_llm

class GradeAnswer(BaseModel):
    """Model for evaluating whether an answer adequately addresses a given question."""

    binary_score: str = Field(
        description="Answer quality: 'yes' (resolves the question) or 'no' (inadequate response)."
    )

class AnswerGrader:
    """Evaluates whether a generated answer sufficiently addresses the user's question."""
    
    def __init__(self, user_input: Dict[str, str]):
        """Initialize the answer grader with user input configuration.
        
        Args:
            user_input: Dictionary containing user configuration
        """
        self.llm = get_llm(user_input)
        self.structured_answer_grader = self.llm.with_structured_output(GradeAnswer)
        self.answer_grader = None
        self._setup_answer_grader()
        
    def _setup_answer_grader(self):
        """Create and configure the answer grading prompt template."""
        system = """You are an expert grader evaluating whether an answer adequately addresses a given question.

        Evaluation Criteria:
        - If the answer fully resolves the question, classify it as 'yes'.
        - If the answer is incomplete, off-topic, or does not sufficiently resolve the question, classify it as 'no'.

        Provide a binary score:
        - "yes" → The answer sufficiently resolves the question.
        - "no" → The answer does not adequately address the question.
        """

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Evaluate whether the following response adequately addresses the user's question:\n\n"
                    "User Question: {question}\n\n"
                    "LLM-Generated Response: {generation}\n\n"
                    "Provide a binary score ('yes' or 'no')."),
        ])

        self.answer_grader = answer_prompt | self.structured_answer_grader

    def get_answer_grader(self):
        """Return the configured answer grader.
        
        Returns:
            The configured answer grader pipeline
        """
        return self.answer_grader