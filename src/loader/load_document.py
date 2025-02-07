from langchain_community.document_loaders import UnstructuredWordDocumentLoader, Docx2txtLoader
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# khởi tạo llm model
llm = GoogleGenerativeAI(
    model='gemini-2.0-flash-exp',
    api_key=GEMINI_API_KEY,
)

# khởi tạo llm embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GEMINI_API_KEY,
)

def is_file_exist(file_path):
    return os.path.isfile(file_path)

file_path = '../../documents/23_2008_QH12_82203.docx'

if is_file_exist(file_path): # kiểm tra file tồn tại
    # tạo loader tải docx
    loader = Docx2txtLoader(file_path=file_path)
    docs = loader.load()
    
    # Lưu documents vào file txt trong thư mục logs
    with open('../logs/documents.txt', 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(str(doc) + '\n')
        
    print("Đã lưu documents vào logs/documents.txt")
else:
    print('File not found')
    
# khởi tạo SemanticChunker text splitter
text_splitter = SemanticChunker(
    embeddings=embeddings,
    buffer_size=3,
    breakpoint_threshold_type='gradient',
    breakpoint_threshold_amount=95.0,
)

# chunk documents
documents = text_splitter.split_documents(docs)

# lưu chunked documents vào file txt trong thư mục logs
with open('../logs/chunked_documents.txt', 'w', encoding='utf-8') as f:
    for doc in documents:
        f.write(str(doc) + '\n')
print("Đã lưu chunked documents vào logs/chunked_documents.txt")


