from openai import OpenAI
import os 
from dotenv import load_dotenv
import pinecone
from pinecone import PineconeClient

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
resp = client.embeddings.create(
  input=["feline friends say"],
  model="text-embedding-3-small",
  dimensions=384
)


print(resp)
print(resp)
print(resp)
print(resp)