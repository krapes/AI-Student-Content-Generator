import os
import time
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain, SequentialChain
from langchain.vectorstores import Chroma


# Load environment variables from .env file
_ = load_dotenv(find_dotenv())

# Constants
DEFAULT_MODEL = "gpt-3.5-turbo"
PERSIST_DIRECTORY = '../data/chroma'

# Initialize model
llm = ChatOpenAI(temperature=0.9, model=DEFAULT_MODEL)
start_time = time.time()

def open_template(filename):
    with open(os.path.join('templates', filename), 'r') as file:
        data = file.read()
    return data

# Templates
TEMPLATES = {
    "generate_context": open_template("generate_context_template.txt"),
    "generate_question": open_template("generate_question_template.txt"),
    "generate_rubric": open_template("generate_rubric_template.txt"),
    "format_output": open_template("format_output_instructions_template.txt"),
}


# Inputs
topic = "Baseball"
ccss_input = 'CCSS.ELA-LITERACY.W.4.9 '

vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=OpenAIEmbeddings())

nearest_ccss = vectordb.similarity_search(ccss_input, k=1)[0]
common_core = nearest_ccss.metadata.get('concat_text')


generate_context = LLMChain(llm=llm,
                            prompt=ChatPromptTemplate.from_template(TEMPLATES["generate_context"]),
                            output_key="student_reading")
generate_question = LLMChain(llm=llm,
                             prompt=ChatPromptTemplate.from_template(TEMPLATES["generate_question"]),
                             output_key='student_question')
generate_rubric = LLMChain(llm=llm,
                           prompt=ChatPromptTemplate.from_template(TEMPLATES["generate_rubric"]),
                           output_key='student_rubric'
                           )
format_output = LLMChain(llm=llm,
                         prompt=ChatPromptTemplate.from_template(TEMPLATES["format_output"]),
                         output_key='formatted_output')

overall_chain = SequentialChain(chains=[generate_context, generate_question, generate_rubric, format_output],
                                input_variables=["topic", "common_core"],
                                output_variables=["student_reading",
                                                  "student_question",
                                                  "student_rubric",
                                                  "formatted_output"],
                                verbose=True
                                )
response = overall_chain({'topic': topic, 'common_core': common_core})

for key in response.keys():
    print(f"KEY: {key}")
    print(response.get(key))
print("--- %s seconds ---" % (time.time() - start_time))