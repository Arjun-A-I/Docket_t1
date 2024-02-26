from langchain_openai import OpenAI,OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
 

import os
import json
from dotenv import load_dotenv
load_dotenv()
openapi_key=os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)

def generate_embeddings(text):
    return embeddings.embed_query(text)


def keywordfinder(user_question):
    prompt_template=PromptTemplate(
        input_variables=['user_question'],
        template=''' Given the question: {user_question}, analyze and extract the main 
        keywords that represent the core topics or subjects addressed. 
        Please list these keywords in a structured json format, emphasizing the 
        specific terms with no extra wordings.'''
    )
    llm = OpenAI(temperature=0.6)

    chain=LLMChain(llm=llm,prompt=prompt_template)

    res=chain.invoke(user_question)
    parsed_text = json.loads(res['text'])
    keywords=parsed_text['keywords']
    return keywords

# print(keywordfinder("What is the color of the sky?"))
# print(generate_embeddings("Hai hello namaskaram"))