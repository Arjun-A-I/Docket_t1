from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import os
from dotenv import load_dotenv
load_dotenv()
openapi_key=os.getenv('OPENAI_API_KEY')


prompt_template=PromptTemplate(
    input_variables=['user_question'],
    template=" Identify the keywords in the question {user_question}. "
)

llm = OpenAI(temperature=0.6)
# res=prompt_template.format(user_question="What is the color of the sky?")
chain=LLMChain(llm=llm,prompt=prompt_template)
print(chain.invoke("What is the color of the sky?"))
