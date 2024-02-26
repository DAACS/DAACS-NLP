import os
from dotenv import load_dotenv
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# Define your Estimates class
class Estimates(BaseModel):
    age: str = Field(description="estimate age of the person writing this essay")
    gender: str = Field(description="estimate the gender of this person as either male, female, or unknown")
    positivity: str = Field(description="estimate how positive this person is on a scale of one to ten")
    state_of_residence: str = Field(description="estimate what state they are from")
    grade_level: str = Field(description="estimate the grade level or college year level of the person writing this essay")
    gpa: str = Field(description="estimate the current grade point average on a four point scale")
    awareness: str = Field(description= "estimate the awareness on a scale of one to 10, of the person writing this essay. Awareness phrases include but are not limited to:  should try/ I should, I could/can work on/use, It would be best if I can, I will be consider doing/ , possibly implementing, I need to/ I will need to , I can see the benefit/advantage of , (strategies) may also help to improve/some ways to improve , I have really learned how to,, I would like to work upon and heavily improve.. , It would be very beneficial to,  will help , , I think/ I would commit to , e.g., I think I would be committed to using all of these strategies that I have listed. ")


# Load environment variables
load_dotenv()
OPENAI_MODEL = "gpt-4"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

PROMPT_ESSAY_ANALYSIS ="""
Provide information about {essay}.
{format_instructions}
"""

# Define your EstimateService class
class EstimateService:
    def __init__(self):
        self.parser = PydanticOutputParser(pydantic_object=Estimates)
        self.llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL)

    def run(self, essay: str) -> dict:
        message = HumanMessagePromptTemplate.from_template(template=PROMPT_ESSAY_ANALYSIS)
        chat_prompt = ChatPromptTemplate.from_messages([message])

        chat_prompt_with_values = chat_prompt.format_prompt(
            essay=essay, format_instructions=self.parser.get_format_instructions()
        )

        output = self.llm.invoke(chat_prompt_with_values.to_messages())
        results = self.parser.parse(output.content)
        return results.dict()

