
import streamlit as st
from typing import Literal
from langgraph.graph.state import CompiledStateGraph

class DisplayResultStreamlit:
    """Displays chatbot interaction results in Streamlit UI."""

    def __init__(self, graph: CompiledStateGraph, user_message: str):
        self.graph = graph
        self.user_message = user_message

        # Initialize message history if not in session state
        if "message_history" not in st.session_state:
            st.session_state.message_history = []   
    
    def _display_message(self,role:Literal["user","assistant"], message:str) -> None:
        with st.chat_message(role):
            st.markdown(message)

    def _display_chat_history(self) -> None:
        for chat in st.session_state.message_history:
            self._display_message(chat["role"], chat["message"])
    
    def handle_adaptive_rag_conversation(self) -> None:
        self._display_chat_history()

        st.session_state.message_history.append({"role": "user", "message": self.user_message})
        self._display_message("user", self.user_message)
        
        try:
            response = self.graph.invoke(
                input= {
                    'question': self.user_message
                }
            )
            ai_response = response['generation']
            
            st.session_state.message_history.append({"role": "assistant", "message": ai_response})
            self._display_message("assistant", ai_response)
                

        except Exception as e:
            st.error(f"Error processing response: {str(e)}")
            