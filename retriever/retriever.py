# retriever/retriever.py

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import LLMChain
from format_prompt.prompt_formatter import prompt_formatter

def retrieve_relevant_resources(query: str, vectorstore: Chroma, embedding_model, k: int = 5):
    """
    Embed the query, perform similarity search with scores, and return the top k results.
    """
    # Embed the query
    query_embedding = embedding_model.embed_query(query)
    
    # Perform similarity search
    results = vectorstore.similarity_search_by_vector(query_embedding, k)
    return results

def create_rag_chain(vectorstore, llm, embedding_model):
    def retrieve_and_format(query):
        docs = retrieve_relevant_resources(query, vectorstore, embedding_model, k=5)
        return docs

    def rag_chain(question):
        context_items = retrieve_and_format(question)
        formatted_prompt = prompt_formatter(question, context_items)
        response = llm.invoke(formatted_prompt)
        return response

    return rag_chain