import os
import streamlit as st
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    def __init__(self, user_input: Dict[str, str]):
        self.user_input = user_input
        self.llm = None
        self.error_messages: List[str] = []
    
    def _get_api_key(self, api_key_name):
        api_key = self.user_input[api_key_name] or os.getenv(api_key_name) or st.session_state[api_key_name]
        return api_key
    
    def _validate_requirements(self,
                               api_key: str,
                               model: str,
                               api_key_name: str,
                               model_name: str) -> bool:
        
        is_valid = True

        if not api_key:
            is_valid = False
            self.error_messages.append(f"Please enter the {api_key_name} API KEY.")
        
        if not model:
            is_valid = False
            self.error_messages.append(f"Please select a {model_name}.")
        
        return is_valid
    
    def display_errors(self):
        for message in self.error_messages:
            st.error(message)
    
    @abstractmethod
    def get_llm_model(self):
        pass