from document_loader import DocumentLoader
from vector_database import QdrantDatabase
from uuid import uuid4
from rich import print
from rich import traceback
traceback.install()

file_path = "../../documents/23_2008_QH12_82203.docx"

loader = DocumentLoader(file_path=file_path)

vector_store = QdrantDatabase()

vector_store.delete_collection()

documents = loader.load_and_split()

vector_store.upload(documents=documents)
