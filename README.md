# Graph RAG + Knowledge graph 
This is my Graph RAG + Knowledge graph implementation tryout with Neo4j AuraDB and langchain.

Basically what i did:
1. Setup a pipeline for 
  1.1 Extracting data with LLM.
  1.2 Inserting the extracted data into graphDB with LLM. Transform natural language query into GQL.
  1.3 Updating appropiate embedding properties.
2. Querying with Proper context retreival from graphDB
   2.1 Extract query specific node from the graphDB, extract the details.
   2.2 Semantic search for relevent data.
   2.3 Finally feed the LLM with these context.
3. Setup up a agentic workflow for all of these.



# Graph RAG + Knowledge Graph Implementation

A Retrieval-Augmented Generation (RAG) system built with Neo4j AuraDB and LangChain, leveraging knowledge graph capabilities for enhanced context retrieval.

## Architecture

### 1. Data Processing Pipeline

#### 1.1 LLM-Powered Data Extraction
- Uses LLMs to extract structured entities and relationships from unstructured text
- Transforms raw data into graph-compatible format

#### 1.2 Graph Database Integration
- Automatically generates and executes Cypher queries (GQL)
- Populates Neo4j with extracted knowledge
- Handles node/relationship creation and updates

#### 1.3 Embedding Management
- Generates and stores vector embeddings for graph entities
- Maintains hybrid (graph + vector) search capabilities

### 2. Intelligent Query Processing

#### 2.1 Graph-Aware Retrieval
- Identifies relevant subgraphs based on query intent
- Extracts neighboring nodes and relationships

#### 2.2 Semantic Search
- Augments graph results with vector similarity search
- Combines structural and semantic matching

#### 2.3 Contextual Generation
- Provides LLM with graph-structured context
- Generates knowledge-grounded responses

### 3. Agentic Workflow
- Orchestrates the complete RAG process
- Manages iterative query refinement
- Handles complex, multi-hop reasoning

## Technical Stack
- **Graph Database**: Neo4j AuraDB
- **LLM Framework**: LangChain
- **Vector Embeddings**: OpenAI/HuggingFace
- **Query Processing**: Cypher + Hybrid Search
