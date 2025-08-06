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
