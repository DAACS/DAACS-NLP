import os
from dotenv import load_dotenv


from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class Country(BaseModel):
    capital: str = Field(description="capital of the country")
    name: str = Field(description="name of the country, in uppercase")
    dancing: str = Field(description="a one paragraph overview of the most famous folk dancing from that country")
    description: str = Field(description="a one paragraph overview of nightlight within the capital city")

load_dotenv()

OPENAI_MODEL = "gpt-4"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


PROMPT_COUNTRY_INFO = """
    Provide information about {country}.
    {format_instructions}
    """


def main():
    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=Country)

    # setup the chat model
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL)
    
    message = HumanMessagePromptTemplate.from_template(template=PROMPT_COUNTRY_INFO)
    chat_prompt = ChatPromptTemplate.from_messages([message])

    # get user input
    country_name = input("Enter the name of a country: ")

    print("Generating response...")
    chat_prompt_with_values = chat_prompt.format_prompt(
        country=country_name, format_instructions=parser.get_format_instructions()
    )

    output = llm.invoke(chat_prompt_with_values.to_messages())

    country = parser.parse(output.content)
    
    print(f"Country: {country.name} Capital: {country.capital}\n\nNight Life Overview: {country.description}\n\nFolk Dance History: {country.dancing}")


if __name__ == "__main__":
    main()