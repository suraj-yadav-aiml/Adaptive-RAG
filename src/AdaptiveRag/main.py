import streamlit as st

from src.AdaptiveRag.ui.streamlit.loadui import StreamlitUILoader
from src.AdaptiveRag.llm import get_llm
from src.AdaptiveRag.graph import GraphBuilder
from src.AdaptiveRag.ui.streamlit.display_result import DisplayResultStreamlit


def adaptive_rag():
    ui = StreamlitUILoader()
    user_input = ui.load_streamlit_ui()
    
    user_message = st.chat_input(placeholder="Enter your message")

    if user_message:
        try:
            llm = get_llm(user_input=user_input)
            graph_builder = GraphBuilder(user_input)

            if "graph" not in st.session_state:
                st.toast("Graph not found in session state, creating new Graph")
                graph = graph_builder.setup_graph()
                st.session_state["graph"] = graph

            graph = st.session_state["graph"]
            
            if graph is None:
                    st.error("Error: Graph setup failed.")
                    return
            
            DisplayResultStreamlit(graph, user_message).handle_adaptive_rag_conversation()

        except Exception as e:
            st.error(f"Error(main.py): Graph setup failed - {e}")