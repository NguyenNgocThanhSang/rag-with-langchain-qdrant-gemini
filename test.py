from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
result = client.models.embed_content(
    model=os.getenv('EMBEDDING_MODEL_NAME'),
    contents = "Chở người trên buồng lái quá số lượng quy định;"
)

embedding = result.embeddings[0].values

print(embedding)
print(type(embedding))


# Khởi tạo Qdrant Client cho keyword searchu thuần túy
qdrant_client = QdrantClient(
    url=os.getenv('QDRANT_API_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)

qdrant_client.query_points(
    collection_name='hpt_rag_pipeline',
    query_vector=embedding,
    limit=3
)