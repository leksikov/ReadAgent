
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pathlib import Path
import logging
import asyncio
import yaml
logging.basicConfig(level=logging.ERROR)




CHAT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.6
MAX_TOKENS = 1024

ENTITY = "Assistant AI"
TEMPLATE = """You are an {entity} tasked with finding a logical breakpoints in a given text.
The following text is what you remember from reading an
article and a multiple choice question related to it.
You may read 1 to 3 page(s) of the article again to refresh
your memory to prepare yourself for the question.
Please respond with which page(s) you would like to read in yaml format.
DO NOT select more pages if you don't need to.
You don't need to answer the question yet.
Only answer if you are confident in your answer.

Please answer in Yaml format only for the question from user. With the 'pages'  and 'rationale' fields.
For example, if you want to read a single page, answer with YAML like this:
pages: [8]
rationale: "Your rationale for page selection here."
If you want to read page 2,7 answer with YAML like this:
pages: [2,7]
rationale: "Your rationale for pages selection here."

If there is no relevant pages which can help with answering the question, you can answer with YAML like this:
pages: []
rationale: "No relevant pages to answer the question."
"""

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(TEMPLATE).format(entity=ENTITY),
        HumanMessagePromptTemplate.from_template("Text Memory: ```\n{gist_memory}\n```.\nPlease select the pages you would like to read to answer for my question.\nQuestion: {question}"),
    ]
)

class PageSelector:
    """
    Select pages to answer given user question.
    """

    def __init__(self):
        self.model = ChatOpenAI(model=CHAT_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        self.prompt = chat_template

    async def generate(self, gist_memory:str, question: str) -> str:
        """
        Generate an answer given text pages.

        Returns:
            str: The response generated by the ChatOpenAI model.
        """
        # Generate a random word
        max_retries = 3
        for attempt in range(max_retries):

            try:
                chain = self.prompt | self.model | StrOutputParser()

                response = await chain.ainvoke({"gist_memory": gist_memory, "question": question})

                yaml_content = response.strip('`')  # Removes the backticks around the yaml block
                yaml_content = yaml_content.replace('yaml', '')  # Removes the 'yaml' tag if present
                # Parse the YAML content

                parsed_yaml = yaml.safe_load(yaml_content)

                pages = parsed_yaml.get('pages')
                rationale = parsed_yaml.get('rationale')
                # load yaml
                logging.info(f"PageSelector response: {response}, {gist_memory}, {question}")
                return pages, rationale
            except Exception as e:
                logging.error(f"Error in PageSelector chain {e}")
                logging.error(f"Attempt {attempt + 1} failed in PageSelector chain {e}")
                if attempt < max_retries - 1:  # Check if not the last attempt
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    logging.error("Max retry attempts reached, giving up on PageSelector chain.")
        return ""
