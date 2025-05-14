import chromadb
from chromadb.utils import embedding_functions
chroma_client  = chromadb.client()

collection_name = "test_collection"

embedding_function  = embedding_functions.OpenAIEmbeddingFunction()

collection = chroma_client.create_collection(name=collection_name,embedding_functions=embedding_functions)

docs = {
    {"id":"id1" , "text": "id1text"}
}

for doc in docs:
    collection.add(ids=doc["id"],documents= doc["texts"])


query_text = "hellow"

result = collection.query(query_text=[query_text], n_result=3)