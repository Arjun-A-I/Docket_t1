from openai import OpenAI
import os 
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def keyword_extract(user_question):
    response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
        "role": "user",
        "content": 
        f'''
            Given the user question: '{user_question}', please analyze the question to identify and extract the main keywords that represent the core topics or subjects addressed. Each identified keyword should reflect a significant element of the question, capturing both specific terms and broader concepts relevant to the user's inquiry. 
            After analyzing the question, format the output as a JSON object, where 'keywords' is a list containing the extracted keywords. Ensure the JSON object is correctly structured for easy parsing. For example, if the keywords extracted from the user question are 'security', 'features', and 'product', the output should be formatted as follows:
            {'keywords': ['security', 'features', 'product']}
            Please provide the structured output based on the given user question.
        '''
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response
# Assuming `response` is the variable holding the response from the API
question='Can Slack accommodate custom compliance requirements for large enterprises?'
content=keyword_extract(question)
joke_content = content.choices[0].message.content
print(joke_content)
