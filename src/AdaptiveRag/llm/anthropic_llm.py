from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from src.AdaptiveRag.llm.base_llm import BaseLLMProvider



class AnthropicLLM(BaseLLMProvider):

    def get_llm_model(self) -> Optional[BaseChatModel]:
        try:
            # Clear previous error messages
            self.error_messages = []

            anthropic_api_key = self._get_api_key(api_key_name="ANTHROPIC_API_KEY")
            anthropic_selected_model = self.user_input['selected_anthropic_model']

            if self._validate_requirements(api_key=anthropic_api_key,
                                           model=anthropic_selected_model,
                                           api_key_name="ANTHROPIC_API_KEY",
                                           model_name="Anthropic"):
                self.llm = ChatAnthropic(
                    api_key=anthropic_api_key,
                    model=anthropic_selected_model
                )
                return self.llm

            else:
                self.display_errors()
                return None
                                           
        except Exception as e:
            self.error_messages.append(f"Error initializing OpenAI LLM: {str(e)}")
            self.display_errors()
            return None

