import traceback
import streamlit as st
from typing import Dict, Optional
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.AdaptiveRag.state import AdaptiveRAGState
from src.AdaptiveRag.tools import get_tool_mananger
from src.AdaptiveRag.nodes import (
    AnswerGeneratorNode,
    DocumentGraderNode,
    QuestionRewriterNode,
    RetrieverNode,
    WebSearchNode
)
from src.AdaptiveRag.edge import (
    QueryRouterEdge,
    HallucinationAnswerEdge,
    GenerateOrRewriterEdge
)

class GraphBuilder:
    """Builds and compiles the Adaptive RAG workflow graph."""
    
    def __init__(self, user_input: Dict[str, str]):
        """Initialize the graph builder with user configuration.
        
        Args:
            user_input: Dictionary containing user configuration
        """
        self.user_input = user_input
        self.workflow = StateGraph(AdaptiveRAGState)
        self.tool_manager = get_tool_mananger()
        
        # Initialize node and edge references
        self.answer_generator = None
        self.document_grader = None
        self.question_rewriter = None
        self.retriever = None
        self.web_search = None
        self.query_router = None
        self.hallucination_answer = None
        self.generate_or_rewriter = None
    
    def _graph_nodes(self):
        """Initialize all processing nodes for the graph."""
        try:
            print("---INITIALIZING GRAPH NODES---")
            self.answer_generator = AnswerGeneratorNode(self.user_input).answer_generator_node
            self.document_grader = DocumentGraderNode(self.user_input).document_grader_node
            self.question_rewriter = QuestionRewriterNode(self.user_input).question_rewriter_node
            self.retriever = RetrieverNode(self.user_input).retriever_node
            self.web_search = WebSearchNode().web_search_node
            print("---NODES INITIALIZED SUCCESSFULLY---")
        except Exception as e:
            print(f"---ERROR INITIALIZING NODES: {str(e)}---")
            raise RuntimeError(f"Failed to initialize graph nodes: {str(e)}")
    
    def _graph_edges(self):
        """Initialize all edge functions for the graph."""
        try:
            print("---INITIALIZING GRAPH EDGES---")
            self.query_router = QueryRouterEdge(self.user_input).query_router_edge
            self.hallucination_answer = HallucinationAnswerEdge(self.user_input).hallucination_answer_edge
            self.generate_or_rewriter = GenerateOrRewriterEdge().generate_or_rewriter_node
            print("---EDGES INITIALIZED SUCCESSFULLY---")
        except Exception as e:
            print(f"---ERROR INITIALIZING EDGES: {str(e)}---")
            raise RuntimeError(f"Failed to initialize graph edges: {str(e)}")

    def _adaptive_rag_graph(self):
        """Build the complete Adaptive RAG graph with nodes and edges."""
        try:
            print("---BUILDING ADAPTIVE RAG GRAPH---")
            
            self.workflow.add_node("answer_generator", self.answer_generator)
            self.workflow.add_node("document_grader", self.document_grader)
            self.workflow.add_node("question_rewriter", self.question_rewriter)
            self.workflow.add_node("retriever", self.retriever)
            self.workflow.add_node("web_search", self.web_search)
            
            self.workflow.add_conditional_edges(
                START,
                self.query_router,
                {
                    "web_search": "web_search",
                    "vectorstore": "retriever"
                }
            )
            
            self.workflow.add_edge("web_search", "answer_generator")
            self.workflow.add_edge("retriever", "document_grader")
            
            self.workflow.add_conditional_edges(
                "document_grader",
                self.generate_or_rewriter,
                {
                    "question_rewriter_node": "question_rewriter",
                    "answer_generator_node": "answer_generator"
                }
            )
            
            self.workflow.add_edge("question_rewriter", "retriever")
            
            self.workflow.add_conditional_edges(
                "answer_generator",
                self.hallucination_answer,
                {
                    "useful": END,
                    "not useful": "question_rewriter",
                    "not supported": "answer_generator"
                }
            )
            
            print("---GRAPH BUILT SUCCESSFULLY---")
        except Exception as e:
            print(f"---ERROR BUILDING GRAPH: {str(e)}---")
            raise RuntimeError(f"Failed to build adaptive RAG graph: {str(e)}")
    
    def setup_graph(self) -> Optional[CompiledStateGraph]:
        """Build and compile the workflow graph.
        
        Returns:
            Compiled StateGraph or None if compilation fails
        """
        try:
            print("---SETTING UP ADAPTIVE RAG WORKFLOW---")
            
            self._graph_nodes()
            self._graph_edges()
            self._adaptive_rag_graph()
            
            compiled_graph = self.workflow.compile()
            print("---WORKFLOW SUCCESSFULLY COMPILED---")
            return compiled_graph
            
        except Exception as e:
            print(f"---CRITICAL ERROR: GRAPH COMPILATION FAILED: {str(e)}---")
            print(f"Full error: {traceback.format_exc()}")
            st.error(f"Full error: {traceback.format_exc()}")
            
            return None
