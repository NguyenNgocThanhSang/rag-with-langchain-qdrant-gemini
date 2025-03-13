from src.processor.document_loader import DocumentLoader
from src.processor.vector_store import QdrantDatabase
from uuid import uuid4
from rich import print
from rich import traceback
traceback.install()

file_path = "documents/20_2017_TT-BTTTT_349633.docx"
# file_path = "documents/155_2024_ND-CP_320926.docx"
# file_path="documents/168_2024_ND-CP_619502.docx"


loader = DocumentLoader(file_path=file_path)

# vector_store = QdrantDatabase()

# vector_store.delete_collection()

documents = loader.load_and_split()

# vector_store.upload(documents=documents)

print(documents[0].metadata)