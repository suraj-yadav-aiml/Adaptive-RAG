import os
import streamlit as st
from typing import Dict, Optional
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from src.AdaptiveRag.llm.base_llm import BaseLLMProvider



class GroqLLM(BaseLLMProvider):

    def get_llm_model(self) -> Optional[BaseChatModel]:
        try:
            # Clear previous error messages
            self.error_messages = [] 

            groq_api_key = self._get_api_key(api_key_name="GROQ_API_KEY")
            groq_selected_model = self.user_input['selected_groq_model']

            if self._validate_requirements(api_key=groq_api_key,
                                           model=groq_selected_model,
                                           api_key_name="GROQ_API_KEY",
                                           model_name="Groq"):
                
                self.llm = ChatGroq(
                    api_key=groq_api_key,
                    model=groq_selected_model
                )
                
                return self.llm
            else:
                self.display_errors()
                return None
            
        except Exception as e:
            self.error_messages.append(f"Error initializing Groq LLM: {str(e)}")
            self.display_errors()
            return None