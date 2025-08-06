# Graph RAG + Knowledge Graph Implementation

This is my Graph RAG + Knowledge Graph implementation tryout with Neo4j AuraDB and LangChain.

## What I Did

### 1. Setup a Pipeline For:
- **1.1 Extracting data with LLM**  
  Transform unstructured text into structured graph data using LLM
- **1.2 Inserting the extracted data into graphDB with LLM**  
  Convert natural language queries into GraphQL (GQL)  
  Store transformed data in Neo4j
- **1.3 Updating appropriate embedding properties**  
  Maintain vector embeddings for semantic search

### 2. Querying with Proper Context Retrieval from GraphDB:
- **2.1 Extract query-specific nodes**  
  Retrieve relevant nodes and their details from the graph
- **2.2 Semantic search for relevant data**  
  Find contextually similar information using embeddings
- **2.3 Feed the LLM with these contexts**  
  Generate accurate, knowledge-grounded responses

### 3. Setup an Agentic Workflow:
- Automated orchestration of the entire process  
- From data ingestion to query response  
- With iterative refinement capabilities
