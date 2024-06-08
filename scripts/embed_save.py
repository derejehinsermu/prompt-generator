from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import os

def embed_chunks(pages_and_chunks: list[dict], hf_model_name: str) -> list[dict]:
    embedding_model = HuggingFaceEmbeddings(model_name=hf_model_name)
    for i, chunk in enumerate(pages_and_chunks):
        chunk["embedding"] = embedding_model.embed_documents(chunk["sentence_chunk"])
        print(f"Processed chunk {i + 1}/{len(pages_and_chunks)}:")
    return pages_and_chunks

def save_to_vectorstore(pages_and_chunks: list[dict], hf_model_name: str, persist_directory: str):
    embedding_model = HuggingFaceEmbeddings(model_name=hf_model_name)
    documents = [
        Document(
            page_content=chunk["sentence_chunk"],
            metadata={"page_number": chunk["page_number"]}
        )
        for chunk in pages_and_chunks
    ]
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    # Save to vector store with progress
    print("Saving to vector store...")
    vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model, persist_directory=persist_directory)
    vectorstore.persist()
    return vectorstore
