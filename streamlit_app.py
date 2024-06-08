import sys
import os
import warnings
import streamlit as st

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.load_data import openai_api_key, load_data
from scripts.embed_save import embed_chunks, save_to_vectorstore
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from retriever.retriever import create_rag_chain

def main():
    st.title("Document-based Question Answering")
    
    # Define model and vector store parameters
    hf_model_name = "BAAI/bge-base-en-v1.5"
    persist_directory = "./db"
    
    # Check if vector store already exists
    if os.path.exists(persist_directory):
        embedding_model = HuggingFaceEmbeddings(model_name=hf_model_name)
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    else:
        local_pdf_path = "./data/Rich-Dad-Poor-Dad.pdf"
        pages_and_chunks = load_data(local_pdf_path, num_sentence_chunk_size=12)
        pages_and_chunks = embed_chunks(pages_and_chunks, hf_model_name)
        save_to_vectorstore(pages_and_chunks, hf_model_name, persist_directory)
        embedding_model = HuggingFaceEmbeddings(model_name=hf_model_name)
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    
    # Initialize the LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
    
    # Create the RAG chain
    rag_chain = create_rag_chain(vectorstore, llm, embedding_model)
    
    # Streamlit input for user query
    question = st.text_input("Enter your question:", placeholder="enter a question?")
    
    if st.button("Get Answer"):
        # Get response from the model
        response = rag_chain(question)
        
        # Display the response
        st.write("Response:")
        st.write(response.content)

if __name__ == "__main__":
    main()
