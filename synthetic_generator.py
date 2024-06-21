import json
import os

from dotenv import load_dotenv
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_experimental.tabular_synthetic_data.openai import (
    OPENAI_TEMPLATE,
    create_openai_data_generator,
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from langchain_openai import ChatOpenAI

from data import examples

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

for e in examples:
    if not e["prompt"] or not e["completion"]:
        print("UGH")


class TestingData(BaseModel):
    prompt: str
    completion: str


OPENAI_TEMPLATE = PromptTemplate(
    input_variables=["prompt", "completion"], template="Generate unit tests for the following code: {prompt}\nUnit tests: {completion}")

prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["prompt", "completion"],
    example_prompt=OPENAI_TEMPLATE,
)

synthetic_data_generator = create_openai_data_generator(
    output_schema=TestingData,
    llm=ChatOpenAI(
        model_name="gpt-3.5-turbo",
        api_key=OPENAI_API_KEY,
        temperature=0,
    ),
    prompt=prompt_template,
)

synthetic_results = synthetic_data_generator.generate(
    subject="Synthetic Test Case Data",
    extra="Generate synthetic data test cases for Python code with a prompt of the code to be tested and then a completion of the unit test. Don't forget to wrap code in ``` backticks. Return the exact ouput schema. Don't generate any examples that are already in the example set.",
    runs=1,
)

print(synthetic_results)

output_list = [{"prompt": f"{result.prompt}", "completion": f"{result.completion}"}
               for result in synthetic_results]

with open('additional_examples.json', 'w') as file:
    json.dump(output_list, file, indent=4)
