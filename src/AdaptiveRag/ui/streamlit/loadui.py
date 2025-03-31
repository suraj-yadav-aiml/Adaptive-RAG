import os
import streamlit as st
from streamlit_option_menu import option_menu
from src.AdaptiveRag.ui.uiconfigfile import Config

class StreamlitUILoader:
    def __init__(self):
        self.config = Config()
        self.user_input = {}
    
    def _validate_api_key(self, api_key: str, service: str, reference_link: str) -> None:
        """
        Validate and display warning for API keys.
        
        Args:
            api_key (str): API key to validate
            service (str): Name of the service
            reference_link (str): Link for obtaining API key
        """
        if not api_key:
            st.warning(f"⚠️ Please enter your {service} API key to proceed. Don't have one? Refer: {reference_link}")
    
    def _setup_groq_configuration(self) -> None:
        """Set up Groq LLM configuration in the sidebar."""
        groq_model_options = self.config.get_groq_model_options()
        self.user_input['selected_groq_model'] = st.selectbox(label="Select Groq Model", options=groq_model_options)

        groq_api_key = st.text_input(label="GROQ API KEY", type='password')
        self.user_input['GROQ_API_KEY'] = groq_api_key
        st.session_state['GROQ_API_KEY'] = groq_api_key
        os.environ['GROK_API_KEY'] = groq_api_key

        self._validate_api_key(
            groq_api_key, 
            "GROQ", 
            "https://console.groq.com/keys"
        )
    
    def _setup_openai_configuration(self) -> None:
        """Set up OpenAI LLM configuration in the sidebar."""
        openai_model_options = self.config.get_openai_model_options()
        self.user_input['selected_openai_model'] = st.selectbox(label="Select OpenAI Model",options=openai_model_options)

        openai_api_key = st.text_input(label="OPENAI API KEY", type='password')
        self.user_input['OPENAI_API_KEY'] = openai_api_key
        st.session_state['OPENAI_API_KEY'] = openai_api_key
        os.environ['OPENAI_API_KEY'] = openai_api_key

        self._validate_api_key(
            openai_api_key, 
            "OpenAI", 
            "https://platform.openai.com/settings/organization/api-keys"
        )
    
    def _setup_anthropic_configuration(self) -> None:
        """Set up Anthropic LLM configuration in the sidebar."""
        anthropic_model_options = self.config.get_anthropic_model_options()
        self.user_input['selected_anthropic_model'] = st.selectbox(label="Select Anthropic Model",options=anthropic_model_options)

        anthropic_api_key = st.text_input(label="ANTHROPIC API KEY", type='password')
        self.user_input['ANTHROPIC_API_KEY'] = anthropic_api_key
        st.session_state['ANTHROPIC_API_KEY'] = anthropic_api_key
        os.environ['ANTHROPIC_API_KEY'] = anthropic_api_key

        self._validate_api_key(
           anthropic_api_key, 
            "Anthropic", 
            "https://console.anthropic.com/settings/keys"
        )
    
    def _setup_tavily_configuration(self) -> None:
        """Set up Tavily configuration in the sidebar."""
        tavily_api_key = st.text_input(label="TAVILY API KEY", type="password")
        self.user_input['TAVILY_API_KEY'] = tavily_api_key
        st.session_state['TAVILY_API_KEY'] = tavily_api_key
        os.environ['TAVILY_API_KEY'] = tavily_api_key

        self._validate_api_key(
            tavily_api_key, 
            "TAVILY", 
            "https://app.tavily.com/home"
        )
    
    def _home_page(self):
        page_title = self.config.get_page_title()
        # st.set_page_config(page_title="🤖 " + page_title, layout="wide")
        st.header("🤖 " + page_title)
        st.image(image="./src/AdaptiveRag/ui/streamlit/images/AdaptiveRAG_o.png")

        st.title("Adaptive RAG")
    
        st.header("What it is:")
        st.write(
            "Adaptive RAG builds upon the traditional RAG approach by adding a layer of adaptability, "
            "allowing systems to dynamically adjust their retrieval and generation strategies based on the query's complexity."
        )
        
        st.header("How it works:")
        
        st.subheader("Query Analysis:")
        st.write("A classifier or similar mechanism analyzes the incoming query to determine its complexity.")
        
        st.subheader("Strategy Selection:")
        st.write("Based on the query complexity, the system selects the most appropriate retrieval strategy:")
        
        st.markdown("""
        - **No Retrieval:** For simple queries that can be answered directly by the LLM's internal knowledge.
        - **Single-Step Retrieval:** For moderate complexity queries requiring retrieval from a knowledge base.
        - **Multi-Step Retrieval:** For complex queries that require multiple retrieval steps or iterative refinement.
        """)
        
        st.subheader("Response Generation:")
        st.write("The LLM then uses the retrieved information (or its internal knowledge) to generate a response.")
        
        st.header("Benefits:")
        
        st.markdown("""
        - **Improved Accuracy:** By adapting to the query complexity, Adaptive RAG can improve the accuracy and relevance of the generated responses.
        - **Enhanced Efficiency:** It can avoid unnecessary retrieval for simple queries, leading to improved efficiency.
        - **Better User Experience:** Adaptive RAG can provide a more tailored and user-friendly experience by selecting the most appropriate retrieval strategy for each query.
        """)
        
        st.header("Examples of Adaptive RAG Strategies:")
        
        st.markdown("""
        - **Query Routing:** Routing queries to different retrieval strategies based on their complexity.
        - **Self-Evaluation:** Evaluating the retrieved documents and generated responses and iteratively refining the process.
        - **Iterative Generation:** Allowing for multiple attempts at generation if the initial response is not satisfactory.
        - **Additional Information Seeking:** Allowing the system to seek additional information when needed.
        """)
        
        st.header("Tools and Frameworks:")
        st.write("Adaptive RAG can be implemented using tools and frameworks like LangGraph.")

    
    def load_streamlit_ui(self) -> dict:
        """
        Load and configure Streamlit UI.
        
        Returns:
            dict: User input configuration
        """
        

        with st.sidebar:
            page = option_menu(
                menu_title="Menu",
                options=["Home", "Adaptive RAG"],
                icons='house robot'.split(),
                menu_icon='cast',
                default_index=0,
                orientation='vertical'
            )

        if page == 'Home':
            self._home_page()
        
        elif page == 'Adaptive RAG':
            st.title("Adaptive RAG with LangGraph")

            urls = st.text_area(label="Enter the url(s)  - (One per line)")
            self.user_input['urls'] = urls.split('\n')

            if not self.user_input['urls']:
                st.info("Please enter at least one URL.")
                return

            with st.sidebar:
                llm_options = self.config.get_llm_options()
                self.user_input['selected_llm'] = st.selectbox(label="Select LLM", options=llm_options)

                if self.user_input['selected_llm'] == "Groq":
                    self._setup_groq_configuration()
                elif self.user_input['selected_llm'] == "OpenAI":
                    self._setup_openai_configuration()
                elif self.user_input['selected_llm'] == "Anthropic":
                    self._setup_anthropic_configuration()
                
                self._setup_tavily_configuration()
        
        return self.user_input