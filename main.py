import pinecone
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from pinecone import Pinecone,PodSpec
import time
import openai
from openai import OpenAI
import os

# client = OpenAI(
#     # This is the default and can be omitted
#     # api_key=os.environ.get("OPENAI_API_KEY"),
# 	api_key="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"
# )
# # Embedding engine
# # model = SentenceTransformer('all-MiniLM-L6-v2')
# # def get_embedding(text):
# #     return model.encode(text).tolist()  # Ensure output is list

# def get_embedding(text_to_embed):
# 	# Embed a line of text
# 	response = openai.Embedding.create(
#     	model= "text-embedding-3-small",
#     	input=[text_to_embed]
# 	)
# 	# Extract the AI output embedding as a list of floats
# 	embedding = response["data"][0]["embedding"]
# 	return embedding

# print(get_embedding('Hi hello how are you'))

# initialize connection to pinecone (get API key at app.pc.io)
api_key = "8b8780c5-5220-4f23-892a-4654b9daa0ef"
environment = "gcp-starter"

# configure client
pc = Pinecone(api_key=api_key)
spec = PodSpec(environment=environment)
index_name = pc.Index("questions-index")


# index = pc.Index(index_name)
print(index_name.describe_index_stats())

example_questions = [
    {"id": "q1", "text": "What is the capital of France?"},
    {"id": "q2", "text": "Explain Newton's laws of motion"},
    # Add more questions as needed
]


for question in example_questions:
    embedding = get_embedding(question['text'])  # Embedding is already a list
    index_name.upsert(vectors=[(question['id'], embedding)])

query = "which city has the highest population in the world?"
xq = get_embedding(query)
xc = index_name.query(vector=xq, top_k=5, include_metadata=True)
print(xc)



