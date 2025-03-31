import streamlit as st
from typing import Dict, Optional, Type, Union
from langchain_core.language_models.chat_models import BaseChatModel
from src.AdaptiveRag.llm.base_llm import BaseLLMProvider
from src.AdaptiveRag.llm.groq_llm import GroqLLM
from src.AdaptiveRag.llm.openai_llm import OpenAILLM
from src.AdaptiveRag.llm.anthropic_llm import AnthropicLLM


def get_llm(user_input: Dict[str, str]) -> Optional[BaseChatModel]:
    """
    Function to get the appropriate LLM model instance.
    
    Args:
        user_input: Dictionary containing user configuration settings
        
    Returns:
        Configured LLM model instance or None if initialization fails
    """

    llm_providers: Dict[str, Type[BaseLLMProvider]] = {
        "Groq": GroqLLM,
        "OpenAI": OpenAILLM,
        "Anthropic": AnthropicLLM
    }
    selected_llm = user_input['selected_llm']
    # Get the appropriate LLM class
    llm_class = llm_providers.get(selected_llm)
    
    if llm_class:
        try:
            # Initialize the LLM provider and get the model
            llm_provider = llm_class(user_input)
            llm_model = llm_provider.get_llm_model()
            
            if llm_model:
                return llm_model
            else:
                return None
                
        except Exception as e:
            error_msg = f"Error initializing {selected_llm} LLM: {str(e)}"
            st.error(error_msg)
            return None
    else:
        # Handle unsupported LLM provider
        supported_providers = ', '.join(llm_providers.keys())
        error_msg = f"Unsupported LLM provider: {selected_llm}"
        
        st.error(error_msg)
        st.info(f"Supported providers: {supported_providers}")
        return None