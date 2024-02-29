import pinecone
from pinecone import Pinecone,PodSpec,ServerlessSpec
from langchain_keyword import generate_embeddings,keywordfinder
import os 
import json
from dotenv import load_dotenv
load_dotenv()


pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
spec = PodSpec(environment="gcp-starter")


def load_questions(file):
  with open(file, 'r') as file:
        questions = json.load(file)
  return questions

index_name = "questions-index"
index = pc.Index(index_name)
questions=load_questions('question.json')

for question in questions:
    embedding = get_embeddings(question['text'])
    index.upsert(vectors=[(question['id'], embedding, {"category": question['category']})])
    
# filter_criteria = {"category": "ProductFeatures"}

# question="How can I manage user permissions and access in Slack, and how can administrators regulate data access and sharing within Slack?"
# question="If we switch from Microsoft Teams to Slack, how will it affect our communication efficiency, and what are the price and costs involved for a medium-sized team?"
# question="We are considering Slack for our startup, but we are concerned about scalability and compliance issues as we grow. How does Slack handle these?"
query_result = index.query(vector=generate_embeddings(question), top_k=5,include_metadata=True)
for q in query_result['matches']:
   print(q['id'],q['score'],q['metadata'])

# question_keywords=keywordfinder(question)
# print(question_keywords)
   

