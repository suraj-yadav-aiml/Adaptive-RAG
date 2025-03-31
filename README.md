# Adaptive RAG

## What it is:
Adaptive RAG builds upon the traditional RAG approach by adding a layer of adaptability, allowing systems to dynamically adjust their retrieval and generation strategies based on the query's complexity.

## How it works:

### Query Analysis:
A classifier or similar mechanism analyzes the incoming query to determine its complexity.

### Strategy Selection:
Based on the query complexity, the system selects the most appropriate retrieval strategy:

- **No Retrieval:** For simple queries that can be answered directly by the LLM's internal knowledge.
- **Single-Step Retrieval:** For moderate complexity queries requiring retrieval from a knowledge base.
- **Multi-Step Retrieval:** For complex queries that require multiple retrieval steps or iterative refinement.

### Response Generation:
The LLM then uses the retrieved information (or its internal knowledge) to generate a response.

## Benefits:

- **Improved Accuracy:** By adapting to the query complexity, Adaptive RAG can improve the accuracy and relevance of the generated responses.
- **Enhanced Efficiency:** It can avoid unnecessary retrieval for simple queries, leading to improved efficiency.
- **Better User Experience:** Adaptive RAG can provide a more tailored and user-friendly experience by selecting the most appropriate retrieval strategy for each query.

## Examples of Adaptive RAG Strategies:

- **Query Routing:** Routing queries to different retrieval strategies based on their complexity.
- **Self-Evaluation:** Evaluating the retrieved documents and generated responses and iteratively refining the process.
- **Iterative Generation:** Allowing for multiple attempts at generation if the initial response is not satisfactory.
- **Additional Information Seeking:** Allowing the system to seek additional information when needed.

## Tools and Frameworks:
Adaptive RAG can be implemented using tools and frameworks like LangGraph.
