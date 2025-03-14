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

# Định nghĩa Pydantic schema
class Entity(BaseModel):
    """Mô tả các keywords được trích xuất từ truy vấn"""
    type: str = Field(
        default="",
        description="Loại văn bản, ví dụ: 'luật', 'quyết định', 'thông tư'"
    )
    title: str = Field(
        default="",
        description="Tiêu đề văn bản, ví dụ 'luật giao thông đường bộ'"
    )
    number: str = Field(
        default="",
        description="Số hiệu văn bản, ví dụ: '01/2021/qđ-ubnd'"
    )
    issued_date: str = Field(
        default="",
        description="Thời gian ban hành, ví dụ: '01/01/2025', '2025', '01/2025'"
    )
    chapter: str = Field(
        default="",
        description="Chương, ví dụ: 'chương i', 'chương ii'"
    )
    section: str = Field(
        default="",
        description="Mục, ví dụ: 'mục 1', 'mục 2'"
    )
    article:str = Field(
        default="",
        description="Điều luật, ví dụ: 'điều 1', 'điều 2'"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Danh sách các từ khóa quan trọng, ví dụ: ['giao thông', 'đường bộ']"
    )

class KeywordsExtractor:
    '''Lớp trích xuất thực thể (keywords) từ truy vấn người dùng'''
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv('MODEL_NAME'),
            api_key=os.getenv('GEMINI_API_KEY'),
        )
        self.prompt_template = PromptTemplate(
            input_variables=['query'],
            template="""
            Bạn là một trọ lý AI chuyên phân tích văn bản pháp luật tiếng Việt. Nhiệm vụ của bạn là trích xuất các thực thể quan trọng từ câu truy vấn sau và trả về theo định dạng có cấu trúc.
            
            Câu truy vấn: "{query}"
            
            Hãy phân tích câu truy vấn và điền thông tin vào các trường sau:
            - type: Loại văn bản (ví dụ: 'luật', 'thông tư', 'nghị định', 'quyết định', ...). Nếu không rõ, để trống.
            - title: Tiêu đề đầy đủ của văn bản (ví dụ: 'luật giao thông đường bộ', 'thông tư giáo dục',...). Nếu không có, để trống.
            - number: 'Số hiệu văn bản (ví dụ: '85/2016/NĐ-CP', '168/2024/nđ-cp',...). Nếu không có, để trống.
            - issued_date: Ngày/tháng/năm ban hành (ví dụ: '01/01/2025', 2025',...). Nếu không có, để trống.
            - chapter: Chương trong văn bản (ví dụ: 'chương i', 'chương 2',...). Nếu không có, để trống.
            - section: Mục trong văn bản (ví dụ: 'mục 1', 'mục 2'). Nếu không có, để trống.
            - keywords: Danh sách các từ khóa quan trọng không thuộc các trường trên (ví dụ: 'đối tượng áp dụng', 'quy định', 'phạm vi điều chỉnh',...). Nếu không có, để trống.
            
            **Hướng dẫn**:
            - Giữ nguyên kiểu chữ hoa/thường từ câu truy vấn.
            - Chỉ trích xuất thông tin rõ ràng từ câu truy vấn, không suy đoán nếu như không có dữ liệu.
            - Nếu không có thông tin cho một trường, để giá trị mặc định là chuỗi rỗng ("") hoặc danh sách rỗng([]).
            
            ** Ví dụ**:
            1. Truy vấn: "điều 3 của luật giao thông đường bộ có những nội dung gì?"
                - type: "luật"
                - title: "luật giao thông đường bộ"
                - number: ""
                - issued_date: ""
                - chapter: ""
                - section: ""
                - article: "điều 3"
                - keywords: ["nội dung"]

            2. Truy vấn: "thông tư 01/2021 về giáo dục ban hành ngày 15/03/2021"
                - type: "thông tư"
                - title: "thông tư về giáo dục"
                - number: "01/2021"
                - issued_date: "15/03/2021"
                - chapter: ""
                - section: ""
                - article: ""
                - keywords: ["giáo dục"]
            3. Truy vấn: "chương ii mục 1 của nghị định 85/2016/NĐ-CP"
                - type: "nghị định"
                - title: "nghị định 85/2016/NĐ-CP"
                - number: "85/2016/NĐ-CP"
                - issued_date: ""
                - chapter: "chương ii"
                - section: "mục 1"
                - article: ""
                - keywords: []

            Hãy áp dụng cách phân tích tương tự cho câu truy vấn đã cho và trả về kết quả theo định dạng yêu cầu.
            """
        )
        
        self.structured_llm = self.llm.with_structured_output(Entity)
        
    def extract_entities(self, query: str) -> Dict:
        """
        Trích xuất thực thể (keywords) từ truy vấn người dùng 
        """
        if not query or not query.strip():
            raise ValueError("Truy vấn không được để trống")
        
        prompt_text = self.prompt_template.format(query=query)
        response = self.structured_llm.invoke(prompt_text)
        print(response)
        
        return response.model_dump()
        
        # # Trích xuất thông tin từ response
        # try:
        #     content = response.content.strip()
        #     # Nếu có dấu code block, loại bỏ chúng:
        #     if content.startswith("```json"):
        #         content = content[len("```json"):].strip()
        #     if content.endswith("```"):
        #         content = content[:-3].strip()

        #     return json.loads(content)
            
        # except Exception as e:
        #     raise Exception("Lỗi khi parse response: ", str(e))
        
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
    extractor = KeywordsExtractor()
    query = "Điều khiển xe chạy quá tốc độ quy định trên 35 km/h bị phạt bao nhiêu"
    entities = extractor.extract_entities(query.lower())
    print("Parsed Entities:", entities)
    
    json_string = extractor.dict_to_json(entities)
    print("JSON String:\n", json_string)
    
    