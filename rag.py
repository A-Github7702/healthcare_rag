from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import os

class HealthcareRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = Ollama(model="llama3.2")  # Free local LLM
        
    def load_docs(self, pdf_folder="data"):
        docs = []
        for pdf in os.listdir(pdf_folder):
            if pdf.endswith('.pdf'): 
                loader = PyPDFLoader(f"{pdf_folder}/{pdf}")
                docs.extend(loader.load())
        return docs
    
    def create_vectorstore(self):
        docs = self.load_docs()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local("vector_store")
        return vectorstore
    
    def get_qa_chain(self):
        if not os.path.exists("vector_store"):
            self.create_vectorstore()
            
        vectorstore = FAISS.load_local("vector_store", self.embeddings)
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )