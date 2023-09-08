import os
import time
import argparse
import ast
import util
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain, SequentialChain
from langchain.vectorstores import Chroma


class CCSSContentGenerator:
    DEFAULT_MODEL = "gpt-3.5-turbo"
    PERSIST_DIRECTORY = '../data/chroma'
    TEMPLATES_PATH = 'templates'

    def __init__(self, topic, ccss_input):
        """
        Initialize the content generator with a topic and a given ccss_input.

        :param topic: str, topic for the content generation.
        :param ccss_input: str, input related to the Common Core State Standards.
        """
        self._load_environment()
        self.llm = ChatOpenAI(temperature=0.9, model=self.DEFAULT_MODEL)
        self.templates = self._load_templates()
        self.topic = topic
        self.ccss_input = ccss_input
        self.vectordb = Chroma(persist_directory=self.PERSIST_DIRECTORY, embedding_function=OpenAIEmbeddings())
        self.common_core = self._get_nearest_ccss()

    def _load_environment(self):
        """Load environment variables from the .env file."""
        load_dotenv(find_dotenv())

    def _load_template(self, filename):
        """
        Load a specific template based on the given filename.

        :param filename: str, name of the template file to be loaded.
        :return: str, content of the template file.
        """
        with open(os.path.join(self.TEMPLATES_PATH, filename), 'r') as file:
            return file.read()

    def _load_templates(self):
        """
        Load all the necessary templates for content generation.

        :return: dict, mapping of template names to their content.
        """
        return {
            "generate_context": self._load_template("generate_context_template.txt"),
            "generate_question": self._load_template("generate_question_template.txt"),
            "generate_rubric": self._load_template("generate_rubric_template.txt"),
            "format_output": self._load_template("format_output_instructions_template.txt"),
        }

    def _get_nearest_ccss(self):
        """
        Retrieve the nearest CCSS based on the provided ccss_input.

        :return: str, nearest CCSS text.
        """
        nearest_ccss = self.vectordb.similarity_search(self.ccss_input, k=1)[0]
        return nearest_ccss.metadata.get('concat_text')

    def generate_content(self):
        """
        Generate content using the configured chains.

        :return: dict, generated content.
        """
        generate_context = LLMChain(llm=self.llm,
                                    prompt=ChatPromptTemplate.from_template(self.templates["generate_context"]),
                                    output_key="student_reading")
        generate_question = LLMChain(llm=self.llm,
                                     prompt=ChatPromptTemplate.from_template(self.templates["generate_question"]),
                                     output_key='student_question')
        generate_rubric = LLMChain(llm=self.llm,
                                   prompt=ChatPromptTemplate.from_template(self.templates["generate_rubric"]),
                                   output_key='student_rubric')
        format_output = LLMChain(llm=self.llm,
                                 prompt=ChatPromptTemplate.from_template(self.templates["format_output"]),
                                 output_key='formatted_output')

        overall_chain = SequentialChain(chains=[generate_context, generate_question, generate_rubric, format_output],
                                        input_variables=["topic", "common_core"],
                                        output_variables=["student_reading",
                                                          "student_question",
                                                          "student_rubric",
                                                          "formatted_output"],
                                        verbose=True)
        response = overall_chain({'topic': self.topic, 'common_core': self.common_core})
        response = util.parse_llm_output_to_json(response["formatted_output"])
        return response

    def print_results(self, response):
        """
        Print the results of the content generation.

        :param response: dict, generated content to be printed.
        """
        for key in response.keys():
            print(f"KEY: {key}")
            print(response.get(key))


if __name__ == '__main__':
    # Initialize the argparse parser
    parser = argparse.ArgumentParser(description='Generate content based on a given topic and CCSS input.')

    # Add the arguments for topic and ccss_input, specifying the default values and help text
    parser.add_argument('--topic', default='Baseball', type=str,
                        help='The topic to generate content for (default: Baseball).')
    parser.add_argument('--ccss_input', default='CCSS.ELA-LITERACY.W.4.9', type=str,
                        help='The CCSS input value (default: CCSS.ELA-LITERACY.W.4.9).')

    # Parse the arguments
    args = parser.parse_args()

    generator = CCSSContentGenerator(args.topic, args.ccss_input)
    start_time = time.time()
    response = generator.generate_content()
    generator.print_results(response)
    print("--- %s seconds ---" % (time.time() - start_time))
