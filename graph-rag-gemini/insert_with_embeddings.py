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

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

print("Connected to Neo4j database")

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

# Read the data from data.txt
data_file_path = "data.txt"
with open(data_file_path, "r", encoding="utf-8") as file:
    data = file.read()
print("Data read from data.txt")

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

def create_embedding_text(node_type: str, node_data: dict) -> str:
    """
    Create meaningful text for embedding based on node type and data
    """
    if node_type == "Patient":
        # Combine key patient information for embedding
        chronic_diseases = ", ".join(node_data.get("chronic_diseases", []))
        allergies = ", ".join(node_data.get("allergies", []))
        lifestyle = ", ".join(node_data.get("lifestyle", []))
        
        text_parts = [
            f"Patient: {node_data.get('name', '')}",
            f"Age: {node_data.get('age', '')} Gender: {node_data.get('gender', '')}",
            f"Summary: {node_data.get('summary', '')}",
            f"Chronic diseases: {chronic_diseases}" if chronic_diseases else "",
            f"Allergies: {allergies}" if allergies else "",
            f"Lifestyle: {lifestyle}" if lifestyle else ""
        ]
        return " | ".join([part for part in text_parts if part])
        
    elif node_type == "Visit":
        text_parts = [
            f"Visit type: {node_data.get('visit_type', '')}",
            f"Chief complaint: {node_data.get('chief_complaint', '')}",
            f"Visit summary: {node_data.get('visit_summary', '')}"
        ]
        return " | ".join([part for part in text_parts if part])
        
    elif node_type == "Symptoms":
        text_parts = [
            f"Symptom: {node_data.get('name', '')}",
            f"Severity: {node_data.get('severity', '')}",
            f"Duration: {node_data.get('duration', '')}",
            f"Description: {node_data.get('description', '')}",
            f"Location: {node_data.get('location', '')}"
        ]
        return " | ".join([part for part in text_parts if part])
        
    elif node_type == "Vitals":
        text_parts = [
            f"Vitals condition: {node_data.get('condition', '')}",
            f"Summary: {node_data.get('summary', '')}",
            f"Weight: {node_data.get('weight', '')} Height: {node_data.get('height', '')}",
            f"Temperature: {node_data.get('temperature', '')} HR: {node_data.get('heart_rate', '')} BP: {node_data.get('blood_pressure', '')}"
        ]
        return " | ".join([part for part in text_parts if part])
        
    elif node_type == "GivenTests":
        test_names = ", ".join(node_data.get("test_names", []))
        text_parts = [
            f"Tests ordered: {test_names}",
            f"Urgency: {node_data.get('urgency', '')}",
            f"Summary: {node_data.get('summary', '')}"
        ]
        return " | ".join([part for part in text_parts if part])
        
    elif node_type == "TestResult":
        text_parts = [
            f"Test: {node_data.get('test_name', '')}",
            f"Category: {node_data.get('test_category', '')}",
            f"Result: {node_data.get('test_result', '')}",
            f"Abnormal: {node_data.get('abnormal_flag', '')} Severity: {node_data.get('severity', '')}"
        ]
        return " | ".join([part for part in text_parts if part])
        
    elif node_type == "Diagnosis":
        text_parts = [
            f"Diagnosis: {node_data.get('name', '')}",
            f"Description: {node_data.get('brief_description', '')}",
            f"Detailed: {node_data.get('detailed_description', '')}",
            f"Confidence: {node_data.get('confidence_level', '')} Severity: {node_data.get('severity', '')}"
        ]
        return " | ".join([part for part in text_parts if part])
        
    elif node_type == "Prescription":
        text_parts = [
            f"Prescription duration: {node_data.get('total_duration', '')}",
            f"Status: {node_data.get('status', '')}",
            f"Instructions: {node_data.get('special_instructions', '')}"
        ]
        return " | ".join([part for part in text_parts if part])
        
    elif node_type == "Medicine":
        text_parts = [
            f"Medicine: {node_data.get('name', '')}",
            f"Dosage: {node_data.get('dosage', '')}",
            f"Frequency: {node_data.get('frequency', '')}",
            f"Duration: {node_data.get('duration', '')}",
            f"Instructions: {node_data.get('instructions', '')}"
        ]
        return " | ".join([part for part in text_parts if part])
        
    elif node_type == "DoctorAdvice":
        text_parts = [
            f"Advice type: {node_data.get('advice_type', '')}",
            f"Priority: {node_data.get('priority', '')}",
            f"Advice: {node_data.get('text', '')}"
        ]
        return " | ".join([part for part in text_parts if part])
        
    elif node_type == "Doctor":
        text_parts = [
            f"Doctor: {node_data.get('name', '')}",
            f"Specialty: {node_data.get('specialty', '')}"
        ]
        return " | ".join([part for part in text_parts if part])
        
    elif node_type == "FamilyHistory":
        conditions = ", ".join(node_data.get("condition", []))
        text_parts = [
            f"Relation: {node_data.get('relation', '')}",
            f"Conditions: {conditions}",
            f"Status: {node_data.get('current_status', '')}",
            f"Summary: {node_data.get('summary', '')}"
        ]
        return " | ".join([part for part in text_parts if part])
        
    elif node_type == "Upload":
        text_parts = [
            f"Description: {node_data.get('description', '')}",
            f"Content: {node_data.get('extracted_text', '')}"
        ]
        return " | ".join([part for part in text_parts if part])
        
    return ""

def generate_embeddings_for_json(json_data: dict) -> dict:
    """
    Generate embeddings for all nodes that have embedding properties
    """
    nodes_with_embeddings = [
        "Patient", "Visit", "Symptoms", "Vitals", "GivenTests", 
        "TestResult", "Diagnosis", "Prescription", "Medicine", 
        "DoctorAdvice", "Doctor", "FamilyHistory", "Upload"
    ]
    
    updated_data = json_data.copy()
    
    for node_type in nodes_with_embeddings:
        if node_type in updated_data:
            node_data = updated_data[node_type]
            
            # Handle both single nodes and arrays of nodes
            if isinstance(node_data, list):
                for i, node in enumerate(node_data):
                    text_for_embedding = create_embedding_text(node_type, node)
                    if text_for_embedding.strip():
                        try:
                            embedding_vector = embeddings.embed_query(text_for_embedding)
                            updated_data[node_type][i]["embedding"] = embedding_vector
                            print(f"✅ Generated embedding for {node_type}[{i}]")
                        except Exception as e:
                            print(f"❌ Error generating embedding for {node_type}[{i}]: {str(e)}")
                            updated_data[node_type][i]["embedding"] = None
            else:
                text_for_embedding = create_embedding_text(node_type, node_data)
                if text_for_embedding.strip():
                    try:
                        embedding_vector = embeddings.embed_query(text_for_embedding)
                        updated_data[node_type]["embedding"] = embedding_vector
                        print(f"✅ Generated embedding for {node_type}")
                    except Exception as e:
                        print(f"❌ Error generating embedding for {node_type}: {str(e)}")
                        updated_data[node_type]["embedding"] = None
    
    return updated_data

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
        8. DO NOT include embedding fields in the JSON - they will be added later

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
        return extracted_data  # Always return the extracted JSON string

    except Exception as e:
        return f"❌ Error during data extraction: {str(e)}"

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
            MERGE (v)-[:ORDERED_TESTS]->(gt1)
            MERGE (v)-[:DIAGNOSED_WITH]->(d1)
            MERGE (v)-[:PRESCRIBED]->(pres1)
            MERGE (v)-[:RECEIVED_ADVICE]->(da1)
            MERGE (v)-[:CONDUCTED_BY]->(doc1)
            MERGE (s2)-[:SUGGESTS]->(d1)
            MERGE (p)-[:ALLERGIC_TO]->(m1)
            MERGE (p)-[:ALLERGIC_TO]->(m2)
            MERGE (p)-[:HAS_FAMILY_HISTORY]->(fh1)
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
          MERGE (s5)-[:SUGGESTS]->(d1)
          MERGE (p)-[:ALLERGIC_TO]->(m1)
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

def update_embeddings_from_json(embedded_json: dict):
    """
    Update embeddings for nodes in Neo4j using the embedded JSON data
    """
    print("\n🔄 Starting embedding updates...")
    
    # Define node types and their ID fields
    node_mappings = {
        "Patient": {"id_field": "patientID", "label": "Patient"},
        "Visit": {"id_field": "visitID", "label": "Visit"},
        "Vitals": {"id_field": "vitalsID", "label": "Vitals"},
        "GivenTests": {"id_field": "testBatchID", "label": "GivenTests"},
        "TestResult": {"id_field": "testID", "label": "TestResult"},
        "Diagnosis": {"id_field": "diagnosisID", "label": "Diagnosis"},
        "Prescription": {"id_field": "prescriptionID", "label": "Prescription"},
        "Medicine": {"id_field": "medicineID", "label": "Medicine"},
        "DoctorAdvice": {"id_field": "adviceID", "label": "DoctorAdvice"},
        "Doctor": {"id_field": "doctorID", "label": "Doctor"},
        "FamilyHistory": {"id_field": "familyHistoryID", "label": "FamilyHistory"},
        "Upload": {"id_field": "uploadID", "label": "Upload"}
    }
    
    for node_type, mapping in node_mappings.items():
        if node_type in embedded_json:
            node_data = embedded_json[node_type]
            
            # Handle both single nodes and arrays
            nodes_to_process = node_data if isinstance(node_data, list) else [node_data]
            
            for node in nodes_to_process:
                if "embedding" in node and node["embedding"] is not None:
                    node_id = node.get(mapping["id_field"])
                    if node_id:
                        try:
                            # Update embedding for the node
                            update_query = f"""
                            MATCH (n:{mapping["label"]} {{{mapping["id_field"]}: $id}})
                            SET n.embedding = $embedding
                            """
                            
                            graph.query(update_query, {
                                "id": node_id,
                                "embedding": node["embedding"]
                            })
                            
                            print(f"✅ Updated embedding for {node_type} ID: {node_id}")
                            
                        except Exception as e:
                            print(f"❌ Error updating embedding for {node_type} ID {node_id}: {str(e)}")
    
    # Handle Symptoms separately (they don't have a unique ID field, use internal Neo4j id)
    if "Symptoms" in embedded_json:
        symptoms_data = embedded_json["Symptoms"]
        symptoms_list = symptoms_data if isinstance(symptoms_data, list) else [symptoms_data]
        
        for i, symptom in enumerate(symptoms_list):
            if "embedding" in symptom and symptom["embedding"] is not None:
                try:
                    # Find symptoms by name and update embedding
                    symptom_name = symptom.get("name", "")
                    if symptom_name:
                        update_query = """
                        MATCH (s:Symptoms {name: $name})
                        SET s.embedding = $embedding
                        """
                        
                        graph.query(update_query, {
                            "name": symptom_name,
                            "embedding": symptom["embedding"]
                        })
                        
                        print(f"✅ Updated embedding for Symptoms: {symptom_name}")
                        
                except Exception as e:
                    print(f"❌ Error updating embedding for Symptoms[{i}]: {str(e)}")




# ====== MAIN WORKFLOW EXECUTION ======
print("\n" + "="*60)
print("STARTING MEDICAL DATA PROCESSING WORKFLOW")
print("="*60)

# Step 1: Extract JSON from raw data
print("\n📝 Step 1: Extracting structured JSON from raw data...")
result = split_according_to_schema(data)

try:
    # Clean the JSON result
    cleaned_result = result.strip()
    if cleaned_result.startswith("```json"):
        cleaned_result = cleaned_result[len("```json"):].strip()
    if cleaned_result.startswith("```"):
        cleaned_result = cleaned_result[len("```"):].strip()
    if cleaned_result.endswith("```"):
        cleaned_result = cleaned_result[:-len("```")].strip()
    
    result_json = json.loads(cleaned_result)
    print("✅ JSON extraction successful")
    

    # Step 2: Generate embeddings for the JSON
    print("\n🔮 Step 2: Generating embeddings for extracted data...")
    embedded_json = generate_embeddings_for_json(result_json)
     
    # Save embedded JSON
    with open("extracted_data_with_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(embedded_json, f, ensure_ascii=False, indent=2)
    print("✅ Embedded JSON saved to extracted_data_with_embeddings.json")

    
    # Step 3: Generate Cypher code from original JSON (without embeddings)
    print("\n⚙️ Step 3: Generating Cypher code from original JSON...")
    cypher_code = generate_cypher_code(cleaned_result)
    cypher_code = validate_cypher_code(cypher_code)
    print("✅ Cypher code generated and validated")
    

    # Step 4: Execute Cypher to insert nodes and relationships
    print("\n💾 Step 4: Inserting nodes and relationships into Neo4j...")
    try:
        insertion_result = graph.query(cypher_code)
        print("✅ Data inserted into Neo4j successfully")
        print("Insertion result:", insertion_result)
    except Exception as e:
        print(f"❌ Error inserting data into Neo4j: {str(e)}")
        # Don't continue if insertion failed
        exit(1)
    
    
    # Step 5: Update embeddings using embedded JSON
    print("\n🎯 Step 5: Updating embeddings in Neo4j...")
    update_embeddings_from_json(embedded_json)
    print("✅ Embedding updates completed")
    
    print("\n" + "="*60)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*60)
    
except json.JSONDecodeError as e:
    print(f"❌ Failed to parse result as JSON: {str(e)}")
except Exception as e:
    print(f"❌ Workflow failed with error: {str(e)}")