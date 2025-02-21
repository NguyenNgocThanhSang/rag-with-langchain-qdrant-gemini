from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from typing import List
import os
import json
from rich import print
from rich import traceback
from dotenv import load_dotenv

traceback.install()
load_dotenv()

class EntityExtractor:
    '''Lớp trích xuất thực thể (keywords) từ truy vấn người dùng'''
    def __init__(self):
        self.llm = GoogleGenerativeAI(
            model=os.getenv('MODEL_NAME'),
            api_key=os.getenv('GEMINI_API_KEY'),
        )
        self.prompt_template = PromptTemplate(
            input_variables=['query'],
            template="""
            Hãy trích xuất các thực thể quan trọng trong câu sau:
            Câu: "{query}"
            Trả về danh sách các thực thể quan trọng, cách nhau bằng dấu phẩy.
            """
        )
        # self.schema=
        
    def extract_entities(self, query: str) -> List[str]:
        """
        Trích xuất thực thể (keywords) từ truy vấn người dùng. 
        """
        prompt_text = self.prompt_template.format(query=query)
        print(prompt_text)
        response = self.llm.invoke(prompt_text)
        print(response)
        
        # Kiểm tra nếu response có content
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)
        
        keywords = response_text.strip().split(",")
        return [keyword.strip() for keyword in keywords]

# Test    
if __name__ == "__main__":
    extractor = EntityExtractor()
    query = "Những điểm mới trong Quyết định 776/QĐ-UBND năm 2016 có ảnh hưởng như thế nào đến quản lý tài nguyên tại địa phương?"
    print(extractor.extract_entities(query.lower()))
    
    