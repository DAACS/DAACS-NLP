from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_MODEL = "gpt-3.5-turbo-1106"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL)

result = llm.invoke("Pretend I am a freshman in college who has just taken a class on how to make better grades. Now please write me a 300 word essay on what I might to to get those better grades.")
print(result)