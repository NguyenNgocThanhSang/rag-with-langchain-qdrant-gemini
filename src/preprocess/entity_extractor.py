from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
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
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv('MODEL_NAME'),
            api_key=os.getenv('GEMINI_API_KEY'),
        )
        self.prompt_template = PromptTemplate(
            input_variables=['query'],
            template="""
            Hãy trích xuất các thực thể quan trọng trong câu sau:
            
            Câu: "{query}"
            
            Trả về một Dictionary gồm các thực thể quan trọng theo cấu trúc sau đây:
            - type: loại văn bản (str). (ví dụ: "luật", "quyết định", "thông tư", "nghị quyết"...)
            - title: tiêu đề văn bản (str). (ví dụ: "luật giao thông đường bộ)
            - number: số hiệu văn bản (str). (ví dụ: "01/2021/qđ-ubnd")
            - issued_date: thời gian ban hành (str). (ví dụ: 01/01/2025, 2025, 3/2025...)
            - chapter: chương (str). (ví dụ: "chương i", "chương ii", "chương iii")
            - section: mục (str). (ví dụ: "mục 1", "mục 2", "mục 3")
            - article: điều luật (str). (ví dụ: "điều 1", "điều 2", "điều 3")
            - keywords: danh sách các thưc thể quan trọng (List[str]). (ví dụ: ["giao thông", "đường bộ", "danh tính", "quyền lợi"])
            
            
            Lưu ý: trường nào không có thì truyền vào chuỗi rỗng"".
            """
        )
        
    def extract_entities(self, query: str) -> Dict:
        """
        Trích xuất thực thể (keywords) từ truy vấn người dùng 
        """
        
        
        prompt_text = self.prompt_template.format(query=query)
        print(prompt_text)
        response = self.llm.invoke(prompt_text)
        print(response.content)
        
        # Trích xuất thông tin từ response
        try:
            content = response.content.strip()
            # Nếu có dấu code block, loại bỏ chúng:
            if content.startswith("```json"):
                content = content[len("```json"):].strip()
            if content.endswith("```"):
                content = content[:-3].strip()

            return json.loads(content)
            
        except Exception as e:
            raise Exception("Lỗi khi parse response: ", str(e))
        
    def dict_to_json(self, data: Dict) -> str:
        """
        Chuyển đổi một dictionary sang chuỗi JSON định dạng.
        
        Return:
            Chuỗi JSON đã được định dạng (với indent) và đảm bảo hiển thị tiếng Việt đúng (ensure_ascii=False).
        """
        try:
            return json.dumps(data, ensure_ascii=False, indent=4)
        except Exception as e:
            raise Exception("Lỗi khi chuyển đổi Dict sang JSON: " + str(e))

# Test    
if __name__ == "__main__":
    extractor = EntityExtractor()
    query = "Những điểm mới trong Quyết định 776/QĐ-UBND năm 2016 có ảnh hưởng như thế nào đến quản lý tài nguyên tại địa phương?"
    entities = extractor.extract_entities(query.lower())
    print("Parsed Entities:", entities)
    
    json_string = extractor.dict_to_json(entities)
    print("JSON String:\n", json_string)
    
    