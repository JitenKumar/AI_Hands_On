
from openai import OpenAI
import os
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
import pandas as pd

load_dotenv()

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
                return response.choices[0].message["content"]
            except Exception as e:
                print(f"Error generating response: {e}")
                return f"Error generating response: {str(e)}"


def select_models():
    while True:
        choice = input("Select model type (openai/chroma/nomic): ").strip().lower()
        if choice in ["openai", "chroma", "nomic"]:
            return choice
        else:
            print("Invalid choice. Please select 'openai', 'chroma', or 'nomic'.")

        return choice, choice


def generate_csv():
    # Placeholder for CSV generation logic
    data = {
        "id": [1, 2, 3],
        "text": ["Document 1", "Document 2", "Document 3"]
    }

def load_csv():
    df = pd.read_csv("data.csv")
    documents = df.to_dict(orient="records")


def setup_chroma(documents ,embedding_model):
    # Placeholder for Chroma setup logic
    try:
        client.delete_collection("space_facts")
    except:
        pass
    collection = client.create_collection("space_facts", embedding_function=embedding_model.embedding_fn)
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
    context = "\n".join([chunk[0]] for chunk in related_chunks)
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
    return response