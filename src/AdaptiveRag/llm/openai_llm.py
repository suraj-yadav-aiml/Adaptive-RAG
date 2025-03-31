import os
import streamlit as st
from typing import Dict, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from src.AdaptiveRag.llm.base_llm import BaseLLMProvider



class OpenAILLM(BaseLLMProvider):

    def get_llm_model(self) -> Optional[BaseChatModel]:
        try:
            # Clear previous error messages
            self.error_messages = []

            openai_api_key = self._get_api_key(api_key_name="OPENAI_API_KEY")
            openai_selected_model = self.user_input['selected_openai_model']

            if self._validate_requirements(api_key=openai_api_key,
                                           model=openai_selected_model,
                                           api_key_name="OPENAI_API_KEY",
                                           model_name="OpenAI"):
                self.llm = ChatOpenAI(
                    api_key=openai_api_key,
                    model=openai_selected_model
                )
                return self.llm

            else:
                self.display_errors()
                return None
                                           
        except Exception as e:
            self.error_messages.append(f"Error initializing OpenAI LLM: {str(e)}")
            self.display_errors()
            return None

