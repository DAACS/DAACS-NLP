import os
from dotenv import load_dotenv
from daacs.infrastructure.sample_data import SampleData
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class Estimates(BaseModel):
    age: str = Field(description="estimate age of the person writing this essay")
    gender: str = Field(description="estimate the gender of this person")
    grade_level: str = Field(description="estimate the grade level of the person writing this essay")
    hs_gpa: str = Field(description="estimate the high school GPA of this student")
    positivity: str = Field(description="estimate how positive this person is on a scale of one to ten")

load_dotenv()

OPENAI_MODEL = "gpt-4"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

essay = SampleData.good_student  # Ensure this is the actual essay text

PROMPT_ESSAY_ANALYSIS = """
    Given this essay:
    {essay}
    Estimate the following about the author:
    {format_instructions}
"""

def main():
    parser = PydanticOutputParser(pydantic_object=Estimates)

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL)

    message = HumanMessagePromptTemplate.from_template(template=PROMPT_ESSAY_ANALYSIS)
    chat_prompt = ChatPromptTemplate.from_messages([message])

    print("Generating response...")
    chat_prompt_with_values = chat_prompt.format_prompt(
        essay=essay, format_instructions=parser.get_format_instructions()
    )

    output = llm.invoke(chat_prompt_with_values.to_messages())

    results = parser.parse(output.content)
    
    print(results)

if __name__ == "__main__":
    main()