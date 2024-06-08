import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.open_read import open_and_read_pdf
from scripts.split_page import split_text_into_sentences
from scripts.chunk import chunk_sentences, create_chunks

# Load environment variables from a .env file
load_dotenv()

# Access environment variables
langchain_tracing_v2 = os.getenv('LANGCHAIN_TRACING_V2')
langchain_endpoint = os.getenv('LANGCHAIN_ENDPOINT')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
langchain_project = os.getenv('LANGCHAIN_PROJECT')
openai_api_key = os.getenv('OPENAI_API_KEY')

def load_data(pdf_path: str, num_sentence_chunk_size: int):
    pages_and_texts = open_and_read_pdf(pdf_path)
    pages_and_texts = split_text_into_sentences(pages_and_texts)
    pages_and_texts = chunk_sentences(pages_and_texts, num_sentence_chunk_size=num_sentence_chunk_size)
    pages_and_chunks = create_chunks(pages_and_texts)
    return pages_and_chunks
