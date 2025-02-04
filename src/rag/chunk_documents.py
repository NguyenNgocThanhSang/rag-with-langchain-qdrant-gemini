from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader, PyPDFDirectoryLoader
import os

folder_path = "documents/"
docs = []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.pdf'):
        file_path = os.path.join(folder_path, file_name)
        
        # tạo loader cho file pdf
        loader = PDFMinerLoader(file_path=file_path)
        
        # load pdf và thêm vào danh sách
        doc = loader.load()
        docs.extend(doc)
        
print(f"Loaded {len(docs)} documents from {folder_path}")

for d in docs:
    print(f"- {d.page_content=}", sep='\n')