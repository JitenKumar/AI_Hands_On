import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
import os

default_embedding_function = embedding_functions.DefaultEmbeddingFunction()

chroma_client = chromadb.PersistentClient(
    path="chroma_persist/chroma.db",
)

collection_name = "test_collection"

client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_function = embedding_functions.OpenAIEmbeddingFunction(client=client, model="text-embedding-ada-002")

collection = chroma_client.get_or_create_collection(name=collection_name,embedding_function=default_embedding_function)
docs = [
    {"id": "id1", "text": "id1text"},
    {"id": "id2", "text": "id2text"},
]
for doc in docs:
    collection.add(ids=doc["id"], documents=doc["text"])


query_text = "ageOf text"
result = collection.query(query_texts=query_text, n_results=3)
print(result)