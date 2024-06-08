import sys
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.load_data import openai_api_key, load_data
from scripts.embed_save import embed_chunks, save_to_vectorstore
import os
import fitz  # PyMuPDF
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain


def prompt_formatter(query: str, context_items: list):
    context = "- " + "\n- ".join([item.page_content for item in context_items])
    base_prompt = """You are an assistant specialized in generating optimized prompts.
    Give yourself room to think by extracting relevant passages from the context before answering the optimized prompt.
    Don't return the thinking, only return the optimized prompt.
    Make sure your answers are as explanatory as possible.
    Use the following examples as reference for the ideal answer style.
    example 1:
    Original query: Interior furniture design with rocks.
    Optimized prompt: Interior furniture design with rocks, rustic, earthy, minimalist, natural, organic, textured, contemporary, modern, Scandinavian, zen, Japanese, wood, stone, sustainable, eco-friendly, neutral colors, clean lines, spatial, cozy.
    
    example 2:
    Original prompt: Write me programming job candidate requirements.
    Optimized prompt: You are a senior software engineer responsible for assessing the ideal candidate for a programming job. Your role involves analyzing technical skills, experience, and personality traits that contribute to successful software development. With extensive knowledge of programming languages, frameworks, and algorithms, you can accurately evaluate candidates' potential to excel in the field. As an expert in this domain, you can easily identify the qualities necessary to thrive in a programming role. Please provide a detailed yet concise description of the ideal candidate, covering technical skills, personal qualities, communication abilities, and work experience. Focus your knowledge and experience on creating a guide for our recruiting process.
    
    example 3:
    Original query: who is Robert?
    optimized prompt: Provide a detailed overview of Robert Kiyosaki, the author of "Rich Dad Poor Dad." Include his background, key achievements, contributions to financial education, and his impact on personal finance and investment strategies.
    
    example 4:
    Original query: what does it mean poor and rich dad?
    Optimized prompt: Explain the concept of "Rich Dad, Poor Dad" by Robert Kiyosaki, highlighting the differences in financial philosophy and mindset between the rich dad and poor dad. Include key lessons about money management, investment, and financial independence.
    Based on the following context items, generate an optimized prompt for the given query:
    Do not use any outside knowledge. If you don't know the answer based on the context, just say that you don't know. don't forget comparing the Original query with contenxt "
    {context}
    Original query: {query}

    Optimized prompt:"""
    formatted_prompt = base_prompt.format(context=context, query=query)
    return formatted_prompt

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

def main():
    # Define model and vector store parameters
    hf_model_name = "BAAI/bge-base-en-v1.5"
    persist_directory = "../db"
    
    # Check if vector store already exists
    if os.path.exists(persist_directory):
        embedding_model = HuggingFaceEmbeddings(model_name=hf_model_name)
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    else:
        local_pdf_path = "../data/Rich-Dad-Poor-Dad.pdf"
        pages_and_chunks = load_data(local_pdf_path, num_sentence_chunk_size=12)
        pages_and_chunks = embed_chunks(pages_and_chunks, hf_model_name)
        save_to_vectorstore(pages_and_chunks, hf_model_name, persist_directory)
        embedding_model = HuggingFaceEmbeddings(model_name=hf_model_name)
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    
    # Initialize the LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
    
    # Create the RAG chain
    rag_chain = create_rag_chain(vectorstore, llm, embedding_model)
    
    # Example question
    question = "who is dereje"
    response = rag_chain(question)
    print("Response:")
    print(response.content)

if __name__ == "__main__":
    main()