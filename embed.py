import pinecone
from pinecone import Pinecone,PodSpec
from openai import OpenAI

import os 
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_embeddings(text_to_embed):
  response = client.embeddings.create(
    input=[text_to_embed],
    model="text-embedding-3-small",
    dimensions=384
  )
  embedding = response.data[0].embedding
  return embedding

print(get_embeddings('hai hello'))