# import google.generativeai as gemini_client
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, PointStruct, VectorParams
import os
from dotenv import load_dotenv

load_dotenv()

# collection_name = "example_collection"

# # GEMINI_API_KEY = "YOUR GEMINI API KEY"  # add your key here

# client = QdrantClient(url="http://localhost:6333")
# gemini_client.configure(api_key=os.getenv('GEMINI_API_KEY'))
# texts = [
#     "Qdrant is a vector database that is compatible with Gemini.",
#     "Gemini is a new family of Google PaLM models, released in December 2023.",
# ]

# results = [
#     gemini_client.embed_content(
#         model="models/text-embedding-004",
#         content=sentence,
#         task_type="retrieval_document",
#         title="Qdrant x Gemini",
#     )
#     for sentence in texts
# ]

# # create qdrant points
# points = [
#     PointStruct(
#         id=idx,
#         vector=response['embedding'],
#         payload={"text": text}
#     )
#     for idx, (response, text) in enumerate(zip(results, texts))
# ]

# # create collection
# client.create_collection(
#     collection_name=collection_name,
#     vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
# )

# client.upsert(collection_name=collection_name, points=points)

from google import genai

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents="What is the meaning of life?")

print(result.embeddings)