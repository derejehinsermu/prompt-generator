import os
from dotenv import load_dotenv
load_dotenv()

# Access environment variables
langchain_tracing_v2 = os.getenv('LANGCHAIN_TRACING_V2')
langchain_endpoint = os.getenv('LANGCHAIN_ENDPOINT')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
langchain_project = os.getenv('LANGCHAIN_PROJECT')
openai_api_key = os.getenv('OPENAI_API_KEY')


import os
# PyMuPDF
import fitz  
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader


# Load PDF Document from local machine
local_pdf_path = "../data/Rich-Dad-Poor-Dad.pdf" 

# Make sure the file exists
if not os.path.exists(local_pdf_path):
    raise FileNotFoundError(f"The file at {local_pdf_path} was not found.")

# Load PDF document
loader = PyPDFLoader(local_pdf_path)
docs = loader.load()

# split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
splits = text_splitter.split_documents(docs)

# embedding = HuggingFaceEmbeddings(model_name = "BAAI/bge-base-en-v1.5")
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(documents=splits, embedding= embedding)

retriever = vectorstore.as_retriever()