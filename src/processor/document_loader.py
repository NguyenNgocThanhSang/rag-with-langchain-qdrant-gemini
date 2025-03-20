from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
import re
import os
import json
from typing import List
from underthesea import word_tokenize

class DocumentLoader:
    '''Một lớp để xử lý các tài liệu, trích xuất metadata và chia văn bản thành các phần theo cấu trúc'''
    def __init__(self, file_path: str, original_file: str = None):
        self.file_path = file_path
        self.page_content = ""
        self.metadata = {}
        self.original_filename=""
        
    def load(self):
        '''Tải nội dung văn bản từ file docx'''
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Không tìm thấy file: {self.file_path}")
        
        loader = Docx2txtLoader(file_path=self.file_path)
        docs = loader.load()
        self.page_content = '\n'.join(doc.page_content for doc in docs)

    def extract_metadata(self):
        '''Trích xuất metadata từ nội dung văn bản'''
        # Dùng original_filename nếu có, nếu không thì dùng basename của file_path
        # self.metadata = {'source': self.original_filename or os.path.basename(self.file_path)}
        self.metadata = {'source': os.path.basename(self.file_path)}
        
        pattern =  r'\b(LUẬT|NGHỊ ĐỊNH|THÔNG TƯ|QUYẾT ĐỊNH|CHỈ THỊ)\b[\s\xa0\n]*([A-ZÀ-Ỵ\s,;]{5,300}?)(?=\s*(Căn cứ|Điều|Chương|Mục|Phần|$))'
        
        # Lấy loại và tên của văn bản
        type_title_match = re.search(
            pattern=pattern,
            string=self.page_content,
        )
        
        # print(type_title_match)
        
        # thêm trường type và title vào metadata
        if type_title_match:
            self.metadata.update({
                'type': type_title_match.group(1).lower(),
                'title': f"{type_title_match.group(1)} {type_title_match.group(2).strip()}".lower()
            })
            
        # Lấy số hiệu văn bản
        number_match = re.search(
            r'(Số|Số hiệu|Luật số|Nghị định số):\s*([\d/\-A-ZĐ]+)', 
            self.page_content,
            re.IGNORECASE
        )
        
        # Thêm số hiệu văn bản vào metadata
        if number_match:
            self.metadata['number'] = number_match.group(2).lower()
        
        # lấy ngày ban hành
        issued_date_match = re.search(
            r'(Hà Nội|Tp\.HCM|[^,]+),\s*ngày\s(\d{1,2})\s*(?:tháng)?\s*(\d{1,2})\s*(?:năm)?\s*(\d{4})',
            self.page_content
        )
        
        # Thêm ngày ban hành vào trong metadata
        if issued_date_match:
            self.metadata['issued_date'] = f"{issued_date_match.group(2)}/{issued_date_match.group(3)}/{issued_date_match.group(4)}"
        
    def remove_appendix(self):
        '''Xóa phần phụ lục'''
        # Tìm kiếm các phần phụ lục
        markers = ["Nơi nhận:", "PHỤ LỤC", "CHỦ TỊCH QUỐC HỘI"]
        
        # Tìm vị trí sớm nhất của markers
        earliest_pos = len(self.page_content)
        for marker in markers:
            pos =   self.page_content.find(marker)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
                
        # Nếu không tìm thấy markers nào, giữ nguyên văn bản
        if earliest_pos == len(self.page_content):
            return self.page_content.strip()
        
        # Cắt từ đầu văn bản đến marker sớm nhất
        return self.page_content[:earliest_pos].strip()
    
    def load_and_split(self) -> List[Document]:
        '''Load tài liệu, trích xuất metadata và chia nhỏ theo chương, mục, điều luật'''
        self.load()
        self.extract_metadata()
        self.page_content = self.remove_appendix()
        document = Document(page_content=self.page_content, metadata=self.metadata)
        return self._chunk_by_article(self._chunk_by_sections(self._chunk_by_chapter(document=document)))
    
    def _chunk_by_chapter(self, document: Document) -> list[Document]:
        '''Chia tài liệu theo chương'''
        chapter_docs = []
        # Biên dịch mẫu regex thành mẫu pattern chương
        chapter_pattern = re.compile(r'(Chương\s+[IVXLCDM]+\s+[^\n]+)\n+(.*?)(?=\nChương\s+[IVXLCDM]+|$)', re.DOTALL)
        # Tìm nội dung các đoạn theo chương dựa trên pattern 
        chapters = chapter_pattern.findall(document.page_content)
        
        for title, content in chapters:
            # Lowercase và tách tiêu đề chương
            title = title.lower().strip() # chuyển thành chữ thường
            
            # Sử dụng regex để tách số chương và tên chương
            title_pattern = re.compile(r'(chương\s+[ivxlcdm]+)(?:\s+(.*))?')
            title_match = title_pattern.match(title)
            
            if title_match:
                chapter_number = title_match.group(1) # vd: "chương i"
                chapter_title =title_match.group(2).strip() if title_match.group(2) else "" # vd: quy định chung
                chapter_list = [chapter_number, chapter_title]
            else:
                chapter_list =[]
                
            # Tạo Document cho từng chương
            chapter_docs.append(
                Document(
                    page_content=content.strip(),
                    metadata={**document.metadata, 'chapter': chapter_list}
                )
            )
                
        return chapter_docs
        
    def _chunk_by_sections(self, chapter_docs: list[Document]) -> list[Document]:
        '''Chia tài liệu chương theo mục (nếu có)'''
        section_pattern = re.compile(r'(?i)(Mục\s+\d+(?:\.\s+[^\n]*)?)\s*(.*?)(?=\nMục\s+\d+|$)', re.DOTALL)
        section_docs = []
        
        for chapter_doc in chapter_docs:
            matches = section_pattern.findall(chapter_doc.page_content)
            
            if matches:
                for title, content in matches:
                    # Lowercase và tách tiêu đề mục
                    title = title.lower().strip() # chuyển thành chữ thường
                    
                    # Sử dụng regex để tách số mục và tên mục
                    title_pattern = re.compile(r'(mục \d+)(?:\.\s(.*))?')
                    title_match = title_pattern.match(title)
                    
                    if title_match:
                        section_number = title_match.group(1)
                        section_title = title_match.group(2) if title_match.group(2) else ""
                        section_list = [section_number, section_title]
                    else:
                        section_list = []
                        
                    # Tạo Document cho từng mục
                    section_docs.append(
                        Document(
                            page_content=content.strip(),
                            metadata={**chapter_doc.metadata, 'section': section_list}
                        )
                    )
            else:
                section_docs.append(chapter_doc)
                
        return section_docs
    
    def _chunk_by_article(self, documents: list[Document]) -> list[Document]:
        '''Chia tài liệu theo điều luật'''
        article_pattern = re.compile(r'(Điều\s+\d{1,2}+\..*?)(?=\nĐiều\s+\d{1,2}\.|$)', re.DOTALL)
        article_docs = []
        
        for document in documents:
            matches = article_pattern.findall(document.page_content)
            
            for article in matches:
                # Tách theo tiêu đề điều luật
                title_pattern = re.compile(r'(Điều \d+)\.\s*(.+)')
                title_match = title_pattern.match(article.split('\n')[0].strip())
                
                if title_match:
                    article_number = title_match.group(1).lower()
                    article_title = title_match.group(2).lower()
                    article_list = [article_number, article_title]
                else:
                    article_list = []

                # Tạo document cho từng điều luật
                article_docs.append(
                    Document(
                        page_content=article.strip().lower(),
                        metadata={**document.metadata, 'article': article_list}
                    )
                )
                
        return article_docs
    
    def save_to_txt(self, documents: list[Document], output_path: str):
        """Lưu kết quả xử lý vào file txt"""
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(f"Metadata: {json.dumps(doc.metadata, ensure_ascii=False, indent=4)}\n")
                f.write(f"Content:\n{doc.page_content}\n\n")
        print(f"Đã lưu các Document vào {output_path}")

