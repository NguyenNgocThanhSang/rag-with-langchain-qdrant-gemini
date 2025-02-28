import os
from dotenv import load_dotenv
from src.rag.retriever import Retriever
from rag.generator import Generator
from processor.keyword_extractor import KeywordsExtractor

load_dotenv()

retriever = Retriever()
generator = Generator(temperature=0.9)
extractor = KeywordsExtractor()

query = "hành vi sử dụng thông tin, dữ liệu khí tượng thủy văn không đúng mục đích bị phạt bao nhiêu tiền?"

keywords = extractor.extract_entities(query)
print(keywords)

retrieved_docs = retriever.hybrid_search(query=query, keywords=keywords)

answer = generator.generate_answer(question=query, retrieved_docs=retrieved_docs)

print(answer)