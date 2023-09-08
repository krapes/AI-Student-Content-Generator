from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
import ast
import os
import argparse
import argparse
import json
import util
# Assuming you have a module to interact with your LLM, import it here.
# from your_llm_module import LLM
DUMMY_PATH = '../testing/dummy_responses'
FILENAME = 'StudentSimulator_take_exam_output.txt'

class StudentResponseEvaluator:
    DEFAULT_MODEL = "gpt-3.5-turbo"
    TEMPLATES_PATH = 'templates/StudentResponseEvaluator'

    def __init__(self, model=DEFAULT_MODEL):
        # Load environment variables from the .env file.
        self._load_environment()
        self.templates = self._load_templates()
        self.llm = ChatOpenAI(temperature=0.9, model=model)  # Initialize LLM mode


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
            "generate_student_response": self._load_template("generate_student_response_template.txt"),
            "format_output": self._load_template("format_output_template.txt")
        }

    def evaluate(self, data):
        # Validate required fields
        required_fields = ["student_reading", "student_question", "student_rubric", "student_response"]
        for field in required_fields:
            if field not in data:
                return f"Error: {field} is missing."

        # Interact with LLM to evaluate student response
        score = self._get_student_score(data["student_reading"],
                                        data["student_question"],
                                        data["student_response"],
                                        data["student_rubric"])

        return score

    def _get_student_score(self, student_reading, student_question, student_response, student_rubric):
        # Here, you'll use the LLM to compare the student_response with the examples and definitions in the student_rubric

        print("Generating Student Evaluation")
        generate_student_response = LLMChain(llm=self.llm,
                                             prompt=ChatPromptTemplate.from_template(self.templates["generate_student_response"]),
                                             output_key="student_evaluation")
        format_output = LLMChain(llm=self.llm,
                                 prompt=ChatPromptTemplate.from_template(self.templates["format_output"]),
                                 output_key='formatted_output')

        overall_chain = SequentialChain(
            chains=[generate_student_response, format_output],
            input_variables=["student_reading", "student_question", "student_response", "student_rubric"],
            output_variables=["student_evaluation", "formatted_output"],
            verbose=True)
        response = overall_chain({"student_reading":student_reading,
                                  "student_question": student_question,
                                  "student_response": student_response,
                                 "student_rubric": student_rubric})

        response = util.parse_llm_output_to_json(response["formatted_output"])

        return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a student's response.")

    # Arguments for each expected parameter. Adjust as per your needs.
    parser.add_argument('--student_reading', type=str, help='The student reading text.')
    parser.add_argument('--student_question', type=str, help='The student question.')
    parser.add_argument('--student_rubric', type=str, help='The rubric for evaluating the student\'s response.')
    parser.add_argument('--student_response', type=str, help='The student\'s response.')

    args = parser.parse_args()

    # Check if any of the parameters are passed, otherwise read from file
    if args.student_reading and args.student_question and args.student_rubric and args.student_response:
        data = {
            "student_reading": args.student_reading,
            "student_question": args.student_question,
            "student_rubric": args.student_rubric,
            "student_response": args.student_response
        }
    else:
        with open(os.path.join(DUMMY_PATH, FILENAME), 'r') as f:
            data = ast.literal_eval(f.read())  # Assuming your file contains JSON data. Adjust as needed.

    # Now, you can create an instance of your class and evaluate
    evaluator = StudentResponseEvaluator()
    result = evaluator.evaluate(data)
    print(result)
