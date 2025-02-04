from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, PyMuPDFLoader, PDFMinerPDFasHTMLLoader

file_path = './../../documents/69_2024_ND-CP_597437.pdf'

loader = PDFMinerPDFasHTMLLoader(
    file_path=file_path,
)

doc = loader.load()
print(doc)