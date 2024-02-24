import pinecone
from pinecone import Pinecone,PodSpec,ServerlessSpec
from openai import OpenAI
import json
import os 
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment="gcp-starter")
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
spec = PodSpec(environment="gcp-starter")

def get_embeddings(text_to_embed):
  response = client.embeddings.create(
    input=[text_to_embed],
    model="text-embedding-3-small",
    dimensions=1536
  )
  embedding = response.data[0].embedding
  return embedding

def load_questions(file):
  with open(file, 'r') as file:
        questions = json.load(file)
  return questions

index_name = "questions-index"
index = pc.Index(index_name)
print(index.describe_index_stats())

questions=load_questions('question.json')

for question in questions:
    embedding = get_embeddings(question['text'])
    print(question['text'])   
