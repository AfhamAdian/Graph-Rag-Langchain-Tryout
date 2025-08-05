from langchain_community.graphs import Neo4jGraph
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import json

from langchain.chains import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# from langchain.schema import SystemMessage
# from langchain.schema.output_parser import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.schema.runnable import RunnableLambda
# from langchain.agents import initialize_agent, AgentType
# from langchain.tools import Tool

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

print("Connected to Neo4j database")
# print(graph.schema)
# results = graph.query("MATCH (n) RETURN n LIMIT 3")
# print(results)




# connect to gemini
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "invalid"

# Initialize the chat model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.environ.get("GOOGLE_API_KEY")
)
print("Model initialized")

# Initialize the embeddings model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)
print("Embeddings model initialized")

# Initialize Neo4j vector store
# vector_store = Neo4jVector.from_existing_graph(
#     embedding=embeddings,
#     url=os.getenv("NEO4J_URL"),
#     username=os.getenv("NEO4J_USERNAME"),
#     password=os.getenv("NEO4J_PASSWORD"),
#     index_name="visit_embeddings",
#     node_label="Visit",
#     text_node_properties=["text"],
#     embedding_node_property="embedding"
# )
# print("Vector store initialized")


# Read the data from data.txt
data_file_path = "data.txt"
with open(data_file_path, "r", encoding="utf-8") as file:
    data = file.read()
print("Data read from data.txt")
# print(data)


schema = """
Patient {
  patientID: STRING,
  name: STRING,
  gender: STRING,
  age: INTEGER,
  weight: INTEGER,
  height: INTEGER,
  heart_rate: INTEGER,
  blood_pressure: STRING,
  chronic_diseases: [STRING],
  allergies: [STRING],
  blood_type: STRING,
  major_events: [STRING],
  last_updated: STRING,
  lifestyle: [STRING], // smoking, alcohol, physical_activity, diet_type
  summary: STRING,
  // Optional for patient summary RAG
  embedding: [FLOAT]
}

Visit {
  visitID: STRING,
  visit_count: INTEGER,
  date: STRING,
  visit_type: STRING, // routine, emergency, follow-up, consultation
  chief_complaint: STRING,
  visit_summary: STRING,
  status: STRING, // completed, in-progress, cancelled
  embedding: [FLOAT] // For visit summary search
}

// ================================
// CLINICAL DATA ENTITIES
// ================================

Symptoms {
  name: STRING,
  severity: STRING, // mild, moderate, severe
  duration: STRING,
  onset: STRING, // sudden, gradual
  description: STRING,
  location: STRING, // if applicable
  triggers: [STRING],
  relieving_factors: [STRING],
  embedding: [FLOAT] // Critical for symptom matching
}

Vitals {
  vitalsID: STRING,
  condition: STRING,
  weight: STRING,
  height: STRING,
  bmi: FLOAT,
  temperature: STRING,
  heart_rate: INTEGER,
  blood_pressure: STRING,
  recorded_at: STRING,
  // Minimal embedding for abnormal patterns only
  summary: STRING,
  embedding: [FLOAT]
}

// ================================
// DIAGNOSTIC ENTITIES
// ================================

GivenTests {
  testBatchID: STRING,
  test_ids: [STRING],
  test_names: [STRING],
  ordered_date: STRING,
  urgency: STRING, // routine, urgent, stat
  summary: STRING,
  embedding: [FLOAT] // For test reasoning search
}

TestResult {
  testID: STRING,
  test_name: STRING,
  test_category: STRING, // lab, imaging, biopsy, etc.
  test_result: STRING,
  abnormal_flag: BOOLEAN,
  severity: STRING, // normal, borderline, abnormal, critical
  test_date: STRING,
  lab_name: STRING,
  technician: STRING,
  embedding: [FLOAT] // For result interpretation search
}

Diagnosis {
  diagnosisID: STRING,
  name: STRING,
  brief_description: STRING,
  detailed_description: STRING,
  confidence_level: STRING, // confirmed, probable, suspected, ruled-out
  diagnosis_type: STRING, // primary, secondary, differential
  severity: STRING,
  embedding: [FLOAT] // Critical for diagnosis matching
}

// ================================
// TREATMENT ENTITIES
// ================================

Prescription {
  prescriptionID: STRING,
  prescribed_date: STRING,
  status: STRING, // active, completed, discontinued, on-hold
  total_duration: STRING,
  special_instructions: STRING,
  embedding: [FLOAT] // For prescription pattern search
}

Medicine {
  medicineID: STRING,
  name: STRING,
  dosage: STRING,
  frequency: STRING,
  duration: STRING,
  instructions: STRING,
  embedding: [FLOAT] // For drug interaction and alternative search
}

DoctorAdvice {
  adviceID: STRING,
  advice_type: STRING, // lifestyle, follow-up, precaution, emergency
  text: STRING,
  priority: STRING, // low, medium, high, critical
  followup_in_days: INTEGER,
  completion_status: STRING, // pending, completed, overdue
  embedding: [FLOAT] // For advice similarity search
}

// ================================
// HEALTHCARE PROVIDER ENTITIES
// ================================

Doctor {
  doctorID: STRING,
  name: STRING,
  specialty: STRING,
  embedding: [FLOAT] // For doctor expertise matching
}
// ================================
// MEDICAL HISTORY ENTITIES
// ================================

FamilyHistory {
  familyHistoryID: STRING,
  relation: STRING, // parent, sibling, grandparent
  condition: [STRING],
  current_status: STRING, // alive, deceased
  summary: STRING,
  embedding: [FLOAT] // For genetic risk assessment
}

// ================================
// DOCUMENT MANAGEMENT
// ================================

Upload {
  uploadID: STRING,
  description: STRING,
  extracted_text: STRING,
  embedding: [FLOAT] // For document content search
}

// ================================
// RELATIONSHIPS
// ================================

// Core Patient Flow
(:Patient)-[:HAS_VISIT]->(:Visit)
(:Visit)-[:HAS_SYMPTOMS]->(:Symptoms)
(:Visit)-[:HAS_VITALS]->(:Vitals)
(:Visit)-[:ORDERED_TESTS]->(:GivenTests)
(:Visit)-[:DIAGNOSED_WITH]->(:Diagnosis)
(:Visit)-[:PRESCRIBED]->(:Prescription)
(:Visit)-[:RECEIVED_ADVICE]->(:DoctorAdvice)
(:Visit)-[:CONDUCTED_BY]->(:Doctor)
(:Visit)-[:AT_HOSPITAL]->(:Hospital)
(:Visit)-[:HAS_UPLOAD]->(:Upload)

// Test Relationships
(:GivenTests)-[:INCLUDES_TEST]->(:TestResult)
(:TestResult)-[:SUPPORTS]->(:Diagnosis)
(:TestResult)-[:CONTRADICTS]->(:Diagnosis)
(:TestResult)-[:INDICATES]->(:Symptoms)

// Diagnostic Relationships
(:Symptoms)-[:SUGGESTS]->(:Diagnosis)
(:Symptoms)-[:RULES_OUT]->(:Diagnosis)
(:Diagnosis)-[:TREATED_BY]->(:Medicine)
(:Diagnosis)-[:REQUIRES_FOLLOWUP]->(:DoctorAdvice)

// Treatment Relationships
//(:Prescription)-[:CONTAINS]->(:Medicine)
(:Prescription)-[:PRESCRIBED_BY]->(:Doctor)
(:Patient)-[:ALLERGIC_TO]->(:Medicine)
(:Medicine)-[:INTERACTS_WITH]->(:Medicine)

// Temporal Relationships
(:Visit)-[:FOLLOWS]->(:Visit)
(:Visit)-[:NEXT_VISIT]->(:Visit)
""" 





# Function to split data according to schema
def split_according_to_schema(data: str) -> str:
    """
    Split the patient data according to the defined schema and return JSON
    """
    try:
        # Create a prompt for the LLM to extract structured data
        extraction_prompt = f"""
        You are a medical data extraction specialist. Extract information from the following patient data and structure it according to the provided schema. Return ONLY valid JSON.

        PATIENT DATA:
        {data}

        SCHEMA REFERENCE:
        {schema}

        INSTRUCTIONS:
        1. Extract all relevant information from the patient data
        2. Structure it according to the schema provided
        3. Create appropriate IDs where needed (use format: PAT001, VIS001, etc.)
        4. Use proper data types (strings, integers, arrays, etc.)
        5. Include all applicable sections: Patient, Visit, Vitals, Symptoms, FamilyHistory, etc.
        6. Return ONLY the JSON object, no additional text
        7. For arrays, extract multiple items if present in the data

        Expected JSON structure:
        {{
            "Patient": {{...}},
            "Visit": {{...}},
            "Vitals": {{...}},
            "Symptoms": [...],
            "FamilyHistory": [...],
            "Medications": [...]
        }}
        """
        
        # Use the model to extract structured data
        response = model.invoke(extraction_prompt)
        extracted_data = response.content
        # print("Extracted Data: ") 
        # print(extracted_data)
        return extracted_data  # Always return the extracted JSON string

    except Exception as e:
        return f"❌ Error during data extraction: {str(e)}"

# Function to validate extracted data against schema
# tools = [
#     Tool(
#         name="ExtractDataToSchema",
#         func=split_according_to_schema,
#         description="Extract and structure patient data according to the medical schema. Input should be the raw patient data text from data.txt."
#     )
# ]

# Initialize a React agent
# agent = initialize_agent(
#     tools=tools,
#     llm=model,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True
# )
# print("React agent initialized")

# # Example usage: invoke the agent with a task
# result = agent.invoke(
#   {
#     "input": "Extract the patient data from data.txt"
#   }
# )
# print("Agent result:")
# print(result)

result = split_according_to_schema(data)
print("Result from split_according_to_schema:")

try:
  # Remove leading/trailing triple backticks and 'json' if present
  cleaned_result = result.strip()
  if cleaned_result.startswith("```json"):
      cleaned_result = cleaned_result[len("```json"):].strip()
  if cleaned_result.startswith("```"):
      cleaned_result = cleaned_result[len("```"):].strip()
  if cleaned_result.endswith("```"):
      cleaned_result = cleaned_result[:-len("```")].strip()
  result_json = json.loads(cleaned_result)
  print("Result as JSON:")
  print(json.dumps(result_json, indent=2))

except json.JSONDecodeError:
  print("Failed to parse result as JSON.")


data_json = cleaned_result
def generate_cypher_code(json_data: str) -> str:
    try:
        cypher_prompt = f"""
        Generate Neo4j Cypher queries to insert the following medical data into a graph database.
        
        SCHEMA GUIDELINES:
        1. Symptoms are only related to Visits, not directly to Patients.
            There can be multiple symptoms in a visit, dont miss any of them to relate.
        2. There can be many more nodes related to visit, dont miss them either.
        {schema}
        
        MEDICAL DATA (JSON):
        {json_data}
        
        REQUIREMENTS:
        1. Use MERGE instead of CREATE to avoid duplicates
        2. STRICTLY Follow schema to create node and relationships.
        3. Strictly follow the labels and properties defined in schema. Dont add any extra properties.
        4. Make sure all relations that can be done from the created nodes, are done. Also there can be multiple symptoms in a visit, dont miss any of them.
           For example

            MERGE (p)-[:HAS_VISIT]->(v)
            MERGE (v)-[:HAS_VITALS]->(vit)
            MERGE (visit)-[:HAS_SYMPTOMS]->(s1)
            MERGE (visit)-[:HAS_SYMPTOMS]->(s2)
            MERGE (visit)-[:HAS_SYMPTOMS]->(s3)
            MERGE (v)-[:ORDERED_TESTS]->(gt1)
            MERGE (v)-[:DIAGNOSED_WITH]->(d1)
            MERGE (v)-[:PRESCRIBED]->(pres1)
            MERGE (v)-[:RECEIVED_ADVICE]->(da1)
            MERGE (v)-[:CONDUCTED_BY]->(doc1)
            MERGE (s2)-[:SUGGESTS]->(d1)
            MERGE (s5)-[:SUGGESTS]->(d1)
            MERGE (p)-[:ALLERGIC_TO]->(m1)
            MERGE (p)-[:ALLERGIC_TO]->(m2)
            MERGE (p)-[:HAS_FAMILY_HISTORY]->(fh1)
            MERGE (p)-[:HAS_FAMILY_HISTORY]->(fh2)
            MERGE (p)-[:HAS_FAMILY_HISTORY]->(fh3)
            MERGE (pres1)-[:PRESCRIBED_BY]->(doc1)
            MERGE (d1)-[:REQUIRES_FOLLOWUP]->(da1)


        5. Return ONLY the Cypher code, no explanations
        
        Example format:
        MERGE (p:Patient {{patientID: "PAT001", name: "John Doe", ...}})
        MERGE (v:Visit {{visitID: "VIS001", date: "2024-01-01", ...}})
        """
        
        response = model.invoke(cypher_prompt)
        cypher_code = response.content
        
        # Clean up the response
        if cypher_code.startswith("```cypher"):
            cypher_code = cypher_code[len("```cypher"):].strip()
        elif cypher_code.startswith("```"):
            cypher_code = cypher_code[len("```"):].strip()
        if cypher_code.endswith("```"):
            cypher_code = cypher_code[:-len("```")].strip()
        return cypher_code
        
    except Exception as e:
        return f"Error generating Cypher code: {str(e)}"

def validate_cypher_code(cypher_code: str) -> str:
    try:
        prompt = f"""
        Validate and ADD if necessary the following Cypher code for correctness and adherence to the schema:
        
        SCHEMA:
        {schema}
        
        CODE TO VALIDATE:
        {cypher_code}

        Make Sure It has ALL the relationships with all the nodes created. 
        Make sure all relations that can be done from the created nodes, are done. For example :
          MERGE (p)-[:HAS_VISIT]->(v)
          MERGE (v)-[:HAS_VITALS]->(vit)
          MERGE (v)-[:HAS_SYMPTOMS]->(s1)
          MERGE (v)-[:ORDERED_TESTS]->(gt1)
          MERGE (v)-[:DIAGNOSED_WITH]->(d1)
          MERGE (v)-[:PRESCRIBED]->(pres1)
          MERGE (v)-[:RECEIVED_ADVICE]->(da1)
          MERGE (v)-[:CONDUCTED_BY]->(doc1)
          MERGE (s2)-[:SUGGESTS]->(d1)
          MERGE (s3)-[:SUGGESTS]->(d1)
          MERGE (s4)-[:SUGGESTS]->(d1)
          MERGE (s5)-[:SUGGESTS]->(d1)
          MERGE (p)-[:ALLERGIC_TO]->(m1)
          MERGE (p)-[:ALLERGIC_TO]->(m2)
          MERGE (p)-[:HAS_FAMILY_HISTORY]->(fh1)
          MERGE (p)-[:HAS_FAMILY_HISTORY]->(fh2)
          MERGE (p)-[:HAS_FAMILY_HISTORY]->(fh3)
          MERGE (pres1)-[:PRESCRIBED_BY]->(doc1)
          MERGE (d1)-[:REQUIRES_FOLLOWUP]->(da1)

        DOUBLE CHECK FOR RELATIONS. 
        SPLIT MERGE with SET for null values like this :
          MERGE (doc1:Doctor {{doctorID: "DOC001"}})
          ON CREATE SET doc1.name = "Dr. Smith", doc1.specialty = "General Physician"
          SET doc1.embedding = null

        RETURN only the cypher code
        """
        response = model.invoke(prompt)
        validated_code = response.content
        # Clean up the response 
        if validated_code.startswith("```cypher"):
            validated_code = validated_code[len("```cypher"):].strip()
        elif validated_code.startswith("```"):
            validated_code = validated_code[len("```"):].strip()
        if validated_code.endswith("```"):
            validated_code = validated_code[:-len("```")].strip()
        return validated_code
    except Exception as e:
        return f"Error validating Cypher code: {str(e)}"
    

# Generate the Cypher code
cypher_code = generate_cypher_code(data_json)
cypher_code = validate_cypher_code(cypher_code)

print("Final Code for insertion:")
print(cypher_code)


# Run the generated Cypher code to insert data into Neo4j
try:
    insertion_result = graph.query(cypher_code)
    print("Data inserted into Neo4j. Insertion result:")
    print(insertion_result)
except Exception as e:
    print(f"Error inserting data into Neo4j: {str(e)}")


