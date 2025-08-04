from langchain_community.graphs import Neo4jGraph
import os
from dotenv import load_dotenv

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# print(graph.schema)

results = graph.query("MATCH (n) RETURN n LIMIT 3")
print(results)
