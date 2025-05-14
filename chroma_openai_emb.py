import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

load_dotenv()

open_ai_key  =  os.getenv("OPENAI_API_KEY")

default_ef = embedding_functions.DefaultEmbeddingFunction(
)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=open_ai_key, model_name="text-embedding-3-small")
chroma_client = chromadb.PersistentClient(path="./db/chroma_persist")

collection = chroma_client.get_or_create_collection(name="test",embedding_function=openai_ef)
docs = [
    {"id": "id1", "text": "Facebook-> FAISS"},
    {"id": "id2", "text": "o"},
    {"id": "id2", "text": "id2text"},
    {"id": "id2", "text": "id2text"},
    {"id": "id2", "text": "id2text"},
]
for doc in docs:
    collection.add(ids=doc["id"], documents=doc["text"])

query_text = "Micro"
result = collection.query(query_texts=query_text, n_results=3)
print(result)