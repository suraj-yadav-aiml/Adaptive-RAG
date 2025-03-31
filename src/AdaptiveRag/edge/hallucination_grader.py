from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.AdaptiveRag.llm import get_llm

class GradeHallucinations(BaseModel):
    """Model for assessing hallucinations in an LLM-generated response."""

    binary_score: str = Field(
        description="Response factual accuracy: 'yes' (grounded in facts) or 'no' (contains hallucinations)."
    )

class HallucinationGrader:
    """Evaluates whether generated responses are factually grounded in retrieved documents."""
    
    # Constants for grading results
    FACTUAL = "yes"
    HALLUCINATION = "no"
    
    def __init__(self, user_input: Dict[str, str]):
        """Initialize the hallucination grader with user input configuration.
        
        Args:
            user_input: Dictionary containing user configuration
        """
        self.llm = get_llm(user_input)
        self.structured_llm_grader = self.llm.with_structured_output(GradeHallucinations)
        self.hallucination_grader = None
        self._setup_hallucination_grader()
    
    def _setup_hallucination_grader(self):
        """Create and configure the hallucination grading prompt template."""
        system = """You are an expert grader assessing whether an LLM-generated response is factually grounded 
        in a provided set of retrieved facts.

        Evaluation Criteria:
        - If the generated response is fully supported by the provided facts, classify it as 'yes'.
        - If the response contains information that is not explicitly supported or contradicts the facts, classify it as 'no'.

        Provide a binary score:
        - "yes" → The response is grounded in the given facts.
        - "no" → The response contains hallucinations or unsupported information.
        """

        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Evaluate the factual grounding of the following response: \n\n"
                    "Set of Facts: \n{documents}\n\n"
                    "LLM-Generated Response: {generation}\n\n"
                    "Provide a binary score ('yes' or 'no')."),
        ])

        self.hallucination_grader = hallucination_prompt | self.structured_llm_grader
    
    def get_hallucination_grader(self):
        """Return the configured hallucination grader.
        
        Args:
            state: Current pipeline state (unused, kept for API compatibility)
            
        Returns:
            The configured hallucination grader pipeline
        """
        return self.hallucination_grader