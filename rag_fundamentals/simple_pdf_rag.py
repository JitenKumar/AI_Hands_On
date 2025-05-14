
import chromadb
import os
import uuid
import PyPDF2
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from openai import OpenAI
import streamlit as st
load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
collection_name = "pdf_facts"
class SimpleModelSelector:
    def __init__(self):
        self.llm_model = {"openai": "GPT-4", "ollama": "phi3:latest"}

        self.embedding_model = {
            "openai": {
                "name": "openai",
                "dimensions": 1536,
                "model_name" : "text-embedding-3-small",
            },
            "chroma": {
                "name": "Chroma Default",
                "dimensions": 384,
                "model_name" : "None",
            },
            "nomic": {
                "name": "Nomic Embed Text",
                "dimensions": 768,
                "model_name" : "nomic-embed-text",
            },
        }

    def select_models(self):
        """Let user select model from streamlit"""
        embedding_model = st.selectbox(
            "Select embedding model",
            options=list(self.embedding_model.keys()),
            format_func=lambda x: self.embedding_model[x]["name"],
        )
        llm_model = st.selectbox(
            "Select LLM model",
            options=list(self.llm_model.keys()),
            format_func=lambda x: self.llm_model[x],
        )
        return llm_model, embedding_model
    
def SimplePdfProcessor():
    """Process PDF files and create chunking"""
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP

    def read_pdf(self):
        """Process PDF file and create chunks"""
        with open(self.pdf_file, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    self.text += text
        return text
    
    def create_chunks(self,text,pdf_file):
        """Create chunks from text"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i : i + self.chunk_size]
            chunks.append({"id": str(uuid.uuid4()), "text": chunk, "metadata": {"source": pdf_file.name}})
        return chunks
    

class SimpleRAGSystem:
    "Simple RAG system using ChromaDB and OpenAI"

    def __init__(self,embedding_model, llm_model):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.db = chromadb.PersistentClient(path="./db/chroma_persist")
        self.setup_embedding_function()

        if llm_model == "openai":
            self.llm = OpenAI(
                model="gpt-4",
                temperature=0,
                max_tokens=1000,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        else:
            self.llm = OpenAI(
                model="phi3:latest",
                temperature=0,
                max_tokens=1000,
                api_key="ollama",
                base_url="http://localhost:11434/v1",
            )

        
    def setup_collection(self):
        try:
            try:
                self.collection = self.db.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                )

            except:
                collection = self.db.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"model": self.embedding_model}
                )
                st.success("Collection new collection for {self.embedding_model} created")
        except:
            st.error("Error creating collection")
            return None
        return collection

    def setup_embedding_function(self):
        """Setup embedding function for ChromaDB"""
        if self.embedding_model == "openai":
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small",
            )
        elif self.embedding_model == "chroma":
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        elif self.embedding_model == "nomic":
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                model_name="nomic-embed-text",
                api_base="http://localhost:11434/v1",
                api_key="ollama",
            )
        else:
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

    def add_pdf_to_collection(self, chunks):
        """Add PDF chunks to collection"""
        try:
            if not self.collection:
                self.collection = self.setup_collection()
                self.collection.add(
                    documents=[chunk["text"] for chunk in chunks],
                    metadatas=[chunk["metadata"] for chunk in chunks],
                    ids=[chunk["id"] for chunk in chunks],
                )
            return True
        except Exception as e:
            st.error(f"Error adding PDF to collection: {e}")

    def query_collection(self, query,n_results):
        try:
            if not self.collection:
                raise ValueError("No collection available")
            
            results = self.collection.query(query_texts=[query],
                                            n_results=n_results)
            return results
        except Exception as e:
            st.error("Error Query collection : {str(e)}")
            return None
        
    def generate_response(self, query,context):
        """Generate response from LLM"""
        prompt = f"Answer the question based on the context provided. If the answer is not in the context, say 'I don't know'.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        try:
            messages = [
                {"role": "user", "content": query},
                {"role": "system", "content": prompt},
            ]
            model="gpt-4-mini" if self.llm_model=="openai" else "phi3:latest",
            response = self.llm.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return None
    def get_embedding_info(self):
        """Get embedding model info"""
        model_selector = SimpleModelSelector()
        model_info = model_selector.embedding_model[self.embedding_model]
        return {
            "name": model_info["name"],
            "dimensions": model_info["dimensions"],
            "model": self.embedding_model,
        }
    

def main():
    st.title("Simple RAG System with session state")
    model_selector = SimpleModelSelector()
    llm_model, embedding_model = model_selector.select_models()
    if embedding_model != st.session_state.current_embedding_model:
        st.session_state.processed_files.clear()
        st.session_state.current_embedding_model = embedding_model
        st.session_state.rag_system = None
        st.warning("Embedding model changed. Please re-upload PDF files.")

    try:
        if st.session_state.rag_system is None:
            st.session_state.rag_system = SimpleRAGSystem(embedding_model, llm_model)
            st.session_state.current_embedding_model = embedding_model
            st.success("RAG system initialized")

        embedding_info = st.session_state.rag_system.get_embedding_info()
        st.sidebar.info(f"Current Embedding Model:\n"
                        f"Using embedding model: {embedding_info['name']}\n"
                        f"-Dimensions: {embedding_info['dimensions']}\n"
                        f"-Model: {embedding_info['model']}\n")
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf")
    if uploaded_files:
        for pdf_file in uploaded_files:
            if pdf_file.name not in st.session_state.processed_files:
                st.session_state.processed_files.append(pdf_file.name)
                pdf_processor = SimplePdfProcessor(pdf_file)
                text = pdf_processor.read_pdf()
                chunks = pdf_processor.create_chunks(text, pdf_file)
                st.session_state.rag_system.add_pdf_to_collection(chunks)
                st.success(f"Processed {pdf_file.name} and added to collection")
            else:
                st.warning(f"{pdf_file.name} already processed")
    if st.session_state.processed_files:
       st.markdown("-----")
       st.subheader("Query your documents")
       query = st.text_input("Ask a question")
       if(query):
           with st.spinner("Generating response"):
               result =  st.session_state.rag_system.query_collection(query, n_results=3)
               if result:
                   response  = st.session_state.rag_system.generate_response(query, result["documents"][0])
                   
                   if response:
                          st.markdown("Answer:")
                          st.write(response)
                          with st.expander("View Sources passes"):
                              for idx, doc in enumerate(result["documents"][0],1):
                                  st.markdown(f"Passage {idx}:**")
                                  st.info(doc)
               else:
                   st.info("Please upload a PDF file to query")


if __name__ == "__main__":
    main()

    