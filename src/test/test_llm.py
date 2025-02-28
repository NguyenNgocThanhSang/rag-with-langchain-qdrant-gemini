from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from google import generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model='gemini-2.0-flash-exp',
    api_key=GEMINI_API_KEY,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GEMINI_API_KEY
)

print(embeddings.embed_query("How does AI works?"))

