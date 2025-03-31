from typing import Dict
from itertools import chain
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class Retriever:
    def __init__(self, user_input: Dict[str, str]):
        """
        Initialize the Retriever class with user input.
        
        Args:
            user_input (Dict[str, str]): A dictionary containing user input, including URLs.
        """
        self.urls = user_input.get("urls", [])
        self.documents = []
        self.chunks = []
        self.vector_store = None
        self.retriever = None
    
    def _load_documents(self) -> None:
        """
        Load documents from the given URLs.
        """
        self.documents = list(chain.from_iterable(
            [WebBaseLoader(web_path=url).load() for url in self.urls]
        ))
    
    def _split_documents(self, chunk_size: int = 500, chunk_overlap: int = 0) -> None:
        """
        Split documents into smaller chunks using RecursiveCharacterTextSplitter.
        
        Args:
            chunk_size (int): The size of each chunk.
            chunk_overlap (int): The overlap between chunks.
        """
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.chunks = splitter.split_documents(self.documents)
    
    def _create_vector_store(self) -> None:
        """
        Create a FAISS vector store from the document chunks.
        """
        self.vector_store = FAISS.from_documents(
            documents=self.chunks, embedding=OpenAIEmbeddings()
        )
    
    def get_retriever(self, top_k: int = 4):
        """
        Get the retriever object.
        
        Args:
            top_k (int): Number of documents to retrieve.
        
        Returns:
            retriever: A retriever object for similarity search.
        """
        with st.status("Getting the retriever...", expanded=True) as status:
            st.write("Loading the documents...")
            self._load_documents()

            st.write("Splitting the documents...")
            self._split_documents()

            st.write("Creating the vector store...")
            self._create_vector_store()

            st.write("Creating the retriever...")
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
            
            status.update(label="Retriever created successfully!", state="complete", expanded=False)

        return self.retriever



