from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
import re
import os
import json
from dotenv import load_dotenv

class LegalDocumentProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.documents = []
        self.metadata = {}

    def load_document(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"File không tồn tại: {self.file_path}")

        loader = Docx2txtLoader(file_path=self.file_path)
        docs = loader.load()
        self.documents = [MetadataExtractor(page_content=doc.page_content, metadata={"source": os.path.basename(self.file_path)}) for doc in docs]

        # Trích xuất metadata ngay khi tải tài liệu
        for document in self.documents:
            document.extract_metadata()
            if not self.metadata:
                self.metadata = document.metadata
                self.metadata['map'] = {"title": self.metadata.get('title', '')}

    def chunk_by_chapter_and_article(self):
        chapter_pattern = r'(Chương\s+[IVXLCDM]+)\s*(.*?)(?=Chương\s+[IVXLCDM]+|$)'  # Chia theo chương
        article_pattern = r'(Điều\s+\d+)\.\s*(.*?)(?=Điều\s+\d+\.|Chương\s+[IVXLCDM]+|$)'  # Chia theo điều luật

        chunks = []
        for document in self.documents:
            chapters = re.findall(chapter_pattern, document.page_content, flags=re.DOTALL)

            for chapter in chapters:
                chapter_title = f"{chapter[0]} {chapter[1].strip()}"
                chapter_content = chapter[1]

                chapter_metadata = document.metadata.copy()
                chapter_metadata['map'].update({"chapter": chapter_title})

                articles = re.findall(article_pattern, chapter_content, flags=re.DOTALL)

                for article in articles:
                    first_line = article[1].strip().split('\n')[0]  # Tách dòng đầu tiên ra ngoài f-string
                    article_title = f"{article[0]}: {first_line}"
                    article_content = article[1].strip()

                    article_metadata = chapter_metadata.copy()
                    article_metadata['map'].update({"article": article_title})

                    # Tạo đối tượng MetadataExtractor mới và trích xuất metadata
                    article_doc = MetadataExtractor(page_content=article_content, metadata=article_metadata)
                    article_doc.extract_metadata()

                    chunks.append(article_doc)

        return chunks

    def save_to_files(self, output_dir="../logs"):
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "extracted_docs.txt"), 'w', encoding='utf-8') as f:
            for document in self.documents:
                f.write(f"{document.page_content}\n")

        with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=4)

        print("Xử lý xong và lưu trữ dữ liệu.")

class MetadataExtractor(Document):
    def __init__(self, page_content: str, metadata: dict = None):
        super().__init__(page_content=page_content, metadata=metadata or {})

    def extract_metadata(self):
        text = re.sub(r'\s+', ' ', self.page_content)

        type_title_match = re.search(r'\b(LUẬT|NGHỊ ĐỊNH|THÔNG TƯ|QUYẾT ĐỊNH|CHỈ THỊ|THÔNG BÁO|CÔNG VĂN|HƯỚNG DẪN)\b[\s\n]+([A-ZÀ-Ỵ\s]{5,100}?)(?=\s*(Căn cứ|Điều|Chương|Mục|Phần|\n|$))', text)
        if type_title_match:
            self.metadata['type'] = type_title_match.group(1).capitalize()
            self.metadata['title'] = f"{type_title_match.group(1).capitalize()} {type_title_match.group(2).strip()}"

        number_match = re.search(r'(Số|Số hiệu|Luật số):\s*([\d/\-A-Z]+)', text, re.IGNORECASE)
        if number_match:
            self.metadata['number'] = number_match.group(2)

        date_match = re.search(r'(Hà Nội|Tp\. HCM|[^,]+),\s*ngày\s*(\d{1,2})\s*(?:tháng)?\s*(\d{1,2})\s*(?:năm)?\s*(\d{4})', text)
        if date_match:
            self.metadata['issued_date'] = f"{date_match.group(2)}/{date_match.group(3)}/{date_match.group(4)}"

# Ví dụ sử dụng
if __name__ == "__main__":
    load_dotenv()
    file_path = "../../documents/23_2008_QH12_82203.docx"

    processor = LegalDocumentProcessor(file_path)
    processor.load_document()
    processor.chunk_by_chapter_and_article()
    processor.save_to_files()
