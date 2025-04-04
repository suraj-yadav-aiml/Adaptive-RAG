---
title: Adaptive RAG
emoji: 🤖
colorFrom: blue
colorTo: yellow
sdk: streamlit
sdk_version: 1.44.0
app_file: app.py
pinned: false
license: mit
short_description: Adaptive RAG builds upontraditional RAG
---


# Adaptive RAG 🤖

Adaptive Retrieval-Augmented Generation (Adaptive RAG) is an advanced implementation that builds upon traditional RAG systems by adding adaptability features. It dynamically adjusts retrieval and generation strategies based on query complexity, improving answer accuracy and relevance.

![Adaptive RAG](./src/AdaptiveRag/ui/streamlit/images/AdaptiveRAG_o.png)

## 🌟 Features

- **Dynamic Query Routing**: Routes queries to either web search or local vector database based on content
- **Document Relevance Grading**: Evaluates retrieved documents for relevance to the question
- **Question Rewriting**: Optimizes queries that don't yield relevant results
- **Answer Quality Control**: Validates generated answers for factual accuracy and completeness
- **Multi-LLM Support**: Works with OpenAI, Anthropic, and Groq models
- **Interactive UI**: Built with Streamlit for easy configuration and use

## 🧠 How It Works

Adaptive RAG utilizes a directed graph workflow powered by LangGraph:

1. **Query Analysis**: Initial query is routed to the appropriate data source (local vectorstore or web search)
2. **Document Retrieval & Grading**: Documents are retrieved and filtered based on relevance
3. **Question Rewriting**: If no relevant documents are found, the system rewrites the query
4. **Answer Generation**: Using retrieved context to generate a factual response
5. **Quality Control**: Evaluates answers for hallucinations and relevance, triggering regeneration if needed

## 📋 Prerequisites

- Python 3.9+
- API keys:
  - OpenAI (for LLM access)
  - Tavily (for web search capability)

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/adaptive-rag.git
   cd adaptive-rag
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

The system is configured through the Streamlit UI, but you can also modify settings in:
- `src/AdaptiveRag/ui/uiconfigfile.ini` - Configure available LLM models and UI settings

## 🏃‍♂️ Running the Application

Start the application with:

```bash
python app.py
```

This will launch the Streamlit interface, typically at `http://localhost:8501`.

## 📝 Usage Instructions

1. **Setup Environment**:
   - In the sidebar, select your preferred LLM provider (OpenAI, Anthropic, or Groq)
   - Enter your API keys for the selected LLM and Tavily
   - Select the specific model to use

2. **Configure Data Sources**:
   - Enter URLs to create your knowledge base (one URL per line)
   - The system will load, chunk, and embed these documents

3. **Ask Questions**:
   - Type your question in the chat input
   - The system will:
     - Determine the best data source for your query
     - Retrieve and evaluate relevant documents
     - Generate a factual answer based on the retrieved context
     - Verify the answer for quality before presenting it

## 🧩 System Architecture

The project structure follows a modular design:

```
suraj-yadav-aiml-adaptive-rag/
├── README.md
├── app.py                # Application entry point
├── requirements.txt      # Dependencies
└── src/
    └── AdaptiveRag/      # Main package
        ├── edge/         # Edge functions for graph routing 
        ├── graph/        # LangGraph implementation
        ├── llm/          # LLM provider integrations
        ├── nodes/        # Processing nodes
        ├── retriever/    # Document retrieval
        ├── state/        # State management
        ├── tools/        # Additional tools (web search)
        └── ui/           # Streamlit interface
```

## 🛣️ Workflow Graph

The system uses LangGraph to implement a state machine workflow:

![Adaptive RAG](./src/AdaptiveRag/ui/streamlit/images/AdaptiveRAG.png)

## 🔄 Adaptive Features

The system adapts in multiple ways:
- **Query Routing**: Determines whether to use local knowledge or web search
- **Document Relevance**: Filters out irrelevant documents
- **Answer Verification**: Checks for hallucinations and question relevance
- **Query Reformulation**: Rewrites questions when no good documents are found

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
