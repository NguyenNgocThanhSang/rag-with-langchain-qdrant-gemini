from google import genai
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain_docling import DoclingLoader

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

file_path = "documents/69_2024_ND-CP_597437.pdf"

loader = PyPDFLoader(file_path=file_path)
# pages = []
# async for page in loader.alazy_load():
#     pages.append(page)

docs = loader.load()

print(len(docs))
# with open("result.txt", "w") as f:
#     f.write(docs)