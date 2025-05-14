
import streamlit as st
from openai import OpenAI
import os
import chromadb
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
import pandas as pd
import csv


load_dotenv()
collection_name = "aircrafts_facts"
openai_api_key = os.getenv("OPENAI_API_KEY")
class EmbeddingModel:
    def __init__(self,model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(openai_api_key)
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model="text-embedding-3-small"
            )
        elif model_type == "chroma":
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        elif model_type == "nomic":
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                model="nomic-embed-text",
                api_base="http://localhost:11434/v1",
                api_key="ollama"
            )

class LLMModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(openai_api_key)
            self.model_name = "gpt-3.5-turbo"
        else:
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self.model_name = "phi3:latest"

    def generate_completions(self, messages):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error generating response: {e}")
                return f"Error generating response: {str(e)}"


def select_models():
    while True:
        choice = input("Select model type (1. openai :::: 2. ollama): ").strip().lower()
        if choice in ["1","2","3"]:
            embedding_type = {"1":"openai","2":"chroma","3":"nomic"}[choice]
            llm_type = {"1":"openai","2":"ollama","3":"ollama"}[choice]
        else:
            print("Invalid choice. Please select (1. openai :::: 2. ollama) ")

        return llm_type, embedding_type


def generate_csv(): 
    # Sample JSON data about fighter aircraft
    json_data = [
        {"id": 1, "name": "F-22 Raptor", "manufacturer": "Lockheed Martin", "role": "Air superiority","description": "The F-22 Raptor is a fifth-generation, single-seat, twin-engine, all-weather stealth tactical fighter aircraft developed for the United States Air Force."},
        {"id": 2, "name": "F-35 Lightning II", "manufacturer": "Lockheed Martin", "role": "Multirole stealth fighter","description": "The F-35 Lightning II is a family of stealth multirole fighters designed for ground attack and air superiority missions."},
    
        {"id": 3, "name": "Eurofighter Typhoon", "manufacturer": "Airbus, BAE Systems, Leonardo", "role": "Multirole fighter","description": "The Eurofighter Typhoon is a twin-engine, canard-delta wing, multirole fighter designed and manufactured by a consortium of European aerospace companies."},
        {"id": 4, "name": "Su-57", "manufacturer": "Sukhoi", "role": "Stealth multirole fighter","description": "The Su-57 is a fifth-generation, single-seat, twin-engine stealth multirole fighter developed for air superiority and ground attack."},
        {"id": 5, "name": "Chengdu J-20", "manufacturer": "Chengdu Aircraft Industry Group", "role": "Stealth air superiority fighter","description": "The Chengdu J-20 is a fifth-generation stealth fighter aircraft developed by China for the People's Liberation Army Air Force."},   
    ]

    # Specify the CSV file name
    csv_file = 'data.csv'

    # Write JSON data to CSV
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["id","name","manufacturer","role","description"])
        writer.writeheader()  # Write the header row
        writer.writerows(json_data)  # Write the data rows

    print(f"Data has been written to {csv_file}")


def load_csv():
    generate_csv()
    df = pd.read_csv("data.csv")
    documents = df["description"].tolist()
    for doc in documents:
        print("document loaded: ", doc)
    return documents


def setup_chroma(documents ,embedding_model):
    # Placeholder for Chroma setup logic
    try:
        client = chromadb.PersistentClient(path="./db/chroma_persist")
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(collection_name, embedding_function=embedding_model.embedding_fn)
    collection.add(documents=documents,ids=[str(i) for i in range(len(documents))])
    print("Chroma setup complete. and documents added to chroma")
    return collection


def find_related_chunks(query,collection,top_k=2):
    # Placeholder for finding related chunks logic
    results = collection.query(query_texts=query, n_results=top_k)
    print("\n related chunks found")
    for doc in results["documents"][0]:
        print("document: ", doc)
    return list(
        zip(
            results["documents"][0],
            (
                results["metadatas"][0]) if results["metadatas"][0] else [{}] *len(results["documents"][0])
            )
            
            )

def augment_prompt(query, related_chunks):
    # Placeholder for augmenting prompt logic
    print("Related Chunks ::")
    print(related_chunks)
    context = "\n".join(chunk[0] for chunk in related_chunks)
    augment_prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
    print("augmented_prompt: ", augment_prompt)
    return augment_prompt


def rag_pipeline(query, collection, llm_model,top_k=2):
    # Placeholder for RAG pipeline logic
    print("Processing Query: ", {query})
    related_chunks = find_related_chunks(query, collection, top_k)
    augmented_prompt = augment_prompt(query, related_chunks)
    response = llm_model.generate_completions(
        [{
            "role": "system",
            "content": "You are helpful assistant who can answer question based on the context provided."
        },{
            "role": "user",
            "content": augmented_prompt
        }]
    )
    print("Response: ", response)
    references = [chunk[0] for chunk in related_chunks]
    return response , references


def main():
    print("Welcome to the RAG pipeline!")
    llm_type, embedding_type = select_models()
    llm_model = LLMModel(model_type=llm_type)
    embedding_model = EmbeddingModel(model_type=embedding_type)
    print("Using LLM model: ", llm_model.model_type)
    print("Using embedding model: ", embedding_model.model_type)
    documents = load_csv()
    collection = setup_chroma(documents, embedding_model)
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        response, references = rag_pipeline(query, collection, llm_model)
        print("Response: ", response)
        print("References: ", references)


def streamlit_app():
    st.title("RAG Pipeline with Streamlit")
    
    # Model selection
    st.sidebar.header("Model Selection")
    llm_type = st.sidebar.selectbox("Select LLM Model", ["openai", "ollama"])
    embedding_type = st.sidebar.selectbox("Select Embedding Model", ["openai", "chroma", "nomic"])
    
    # Initialize models
    llm_model = LLMModel(model_type=llm_type)
    embedding_model = EmbeddingModel(model_type=embedding_type)
    st.sidebar.write(f"Using LLM model: {llm_model.model_type}")
    st.sidebar.write(f"Using embedding model: {embedding_model.model_type}")
    
    # Load documents
    st.write("Loading documents...")
    documents = load_csv()
    collection = setup_chroma(documents, embedding_model)
    st.write("Documents loaded and Chroma setup complete.")
    
    # Query input
    query = st.text_input("Enter your query:")
    if st.button("Submit Query"):
        if query.strip():
            st.write(f"Processing Query: {query}")
            response, references = rag_pipeline(query, collection, llm_model)
            st.subheader("Response:")
            st.write(response)
            st.subheader("References:")
            for ref in references:
                st.write(ref)
        else:
            st.warning("Please enter a valid query.")

if __name__ == "__main__":
    streamlit_app()