import os
import pandas as pd
import random
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain, SequentialChain


class RandInputGenerator:
    def __init__(self, model="gpt-3.5-turbo"):
        """
            Initializes the random input generator.

            Parameters:
            - model: str, Name of the LLM model to use for topic generation.
            """
        # Load environment configurations
        self._load_environment()
        self.llm = ChatOpenAI(temperature=0.9, model=model)
        self.df_path = '../data/CCSS_standards_dataframe.pkl'
        self.df = pd.read_pickle(self.df_path)

    def _load_environment(self):
        """Load environment variables from the .env file."""
        load_dotenv(find_dotenv())

    def _get_random_topic(self):
        """
            Generate a random topic using an LLM model that's appropriate for grade school children.

            Returns:
            - topic: str, Randomly generated topic.
            """

        prompt = """Generate an essay topic suitable for grade school children. RANDOM NUMBER {randomness}
        None is NOT an option. You must give me some topic that you believe school children study.
        Please keep your answer to 5 words or less.
        """
        generate_topic = LLMChain(llm=self.llm,
                                  prompt=ChatPromptTemplate.from_template(prompt),
                                  output_key="topic")
        randomness = random.random()
        overall_chain = SequentialChain(
            chains=[generate_topic],
            input_variables=['randomness'],
            output_variables=["topic"],
            verbose=True)
        response = overall_chain(randomness)
        return response['topic']

    def _get_random_ccss_input(self):
        """
            Get a random `ccss_input` value from the "Standard" column of the dataframe.

            Returns:
            - ccss_input: str, Randomly selected standard.
            """
        return random.choice(self.df["Standard"].tolist())

    def rand_input(self):
        """
            Generate a random topic and `ccss_input`.

            Returns:
            - topic: str, Randomly generated topic.
            - ccss_input: str, Randomly selected standard.
            """
        topic = self._get_random_topic()
        ccss_input = self._get_random_ccss_input()

        return topic, ccss_input


if __name__ == '__main__':
    generator = RandInputGenerator()

    topic, ccss_input = generator.rand_input()

    print("Randomly generated topic:", topic)
    print("Randomly selected CCSS standard:", ccss_input)
