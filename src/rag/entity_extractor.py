from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

class EntityExtractor:
    '''Lớp trích xuất thực thể (keywords) từ truy vấn người dùng'''
    def __init__(self):
        self.llm = GoogleGenerativeAI(
            model=os.getenv('MODEL_NAME'),
            api_key=os.getenv('GEMINI_API_KEY'),
        ),
        self.prompt_template = PromptTemplate(
            input_variables=['query'],
            template="""
            Hãy trích xuất các thực thể quan trọng trong câu sau:
            Câu: "{query}"
            Trả về danh sách các thực thể quan trọng, cách nhau bằng dấu phẩy.
            """
        )
        
    def extract_entities(self, query: str) -> List[str]:
        """
        Trích xuất thực thể (keywords) từ truy vấn người dùng. 
        """
        prompt_text = self.prompt_template.format(query=query)
        response = self.llm.invoke(prompt_text)
        
        # Kiểm tra nếu response có content
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)
        
        keywords = response_text.strip().split(",")
        return [keyword.strip() for keyword in keywords]

    
if __name__ == "__main__":
    extractor = EntityExtractor()
    query = "Luật Giao thông đường bộ quy định thế nào về việc sử dụng làn đường?"
    print(extractor.extract_entities(query))