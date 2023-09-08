from langchain.chat_models import ChatOpenAI
from RandInputGenerator import RandInputGenerator
from CCSSContentGenerator import CCSSContentGenerator
from StudentResponseEvaluator import StudentResponseEvaluator
from Student import Student
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
import ast
import os
import argparse
import util


class StudentSimulator:

    DUMMY_PATH = '../testing/dummy_responses'
    DEFAULT_MODEL = "gpt-3.5-turbo"

    def __init__(self, student, topic=None, ccss_input=None):

        if student == None:
            raise ValueError("Must inclued a Student Object")
        self.student = student
        self._load_environment()
        # If topic and ccss_input are not provided, get them using RandInputGenerator
        if not topic or not ccss_input:
            rand_input_generator = RandInputGenerator()
            topic, ccss_input = rand_input_generator.rand_input()
            print(f"Random Inputs Created: topic: {topic}  ccss_input: {ccss_input}")

        self.topic = topic
        self.ccss_input = ccss_input
        self.llm = ChatOpenAI(temperature=0.9, model=self.DEFAULT_MODEL)  # Initialize LLM model
        self.exam = None
        self.exam_content = None

    def _load_environment(self):
        """Load environment variables from the .env file."""
        load_dotenv(find_dotenv())
    def _load_dummy(self, filename):
        """
        Load a specific template based on the given filename.

        :param filename: str, name of the template file to be loaded.
        :return: str, content of the template file.
        """
        with open(os.path.join(self.DUMMY_PATH, filename), 'r') as file:
            return file.read()


    def receive_exam(self, test=False):
        if test:
            print("Collecting Dummy CCSSContentGenerator_good_response")
            self.exam_content = self._load_dummy('CCSSContentGenerator_good_response.txt')
        else:
            # 1. Create an object CCSSContentGenerator with topic and ccss_input
            content_generator = CCSSContentGenerator(self.topic, self.ccss_input)
            print("Generating Content for Exam")
            # 2. Run generate_content on the object
            self.exam_content = content_generator.generate_content()
            print(f"Generated Content: {self.exam_content}")

        return self.exam_content

    def take_exam(self, test=False):

        if self.exam_content == None:
            self.receive_exam(test)
        # 4. Using LLM, read the "student_reading" and then respond to the "student_question"
        prompt_template = """
        You are a student in grade {student_grade} with a skill level of {student_skill_level}.
        Read the text provided in INPUTS['student_reading'] then read 
        and answer the question in INPUTS['student_question']
        
        
        INPUTS
        ###
        {content}
        ###
        
        
        Add your response to the INPUTS json in a key name "student_response"
        Please make sure the output is a single, VALID json
        
        Output Example
        {{{{
            "student_reading": 
            "student_question": 
            "student_rubric": 
            "student_response": 
        }}}}
        
        """

        format_output_template = """
        Please format the Input below into a single, valid, json. An example of the output is provide. Do not include
        any text in your output that is not part of this json. The json MUST BE VALID. Do not include any line breaks
        in your response.
        
        Output Example
        {{{{
            "student_reading": 
            "student_question": 
            "student_rubric": 
            "student_response": 
        }}}}
        
        
        INPUT
        {student_response}
        """

        print("Generating Student Response")
        generate_student_response = LLMChain(llm=self.llm,
                                  prompt=ChatPromptTemplate.from_template(prompt_template),
                                  output_key="student_response")
        format_output = LLMChain(llm=self.llm,
                                 prompt=ChatPromptTemplate.from_template(format_output_template),
                                 output_key='formatted_output')

        overall_chain = SequentialChain(
            chains=[generate_student_response, format_output],
            input_variables=["content", "student_grade", "student_skill_level"],
            output_variables=["student_response", 'formatted_output'],
            verbose=True)
        response = overall_chain({"content": self.exam_content,
                                  "student_grade": self.student.grade_level,
                                  "student_skill_level": self.student.skill_level})

        response = util.parse_llm_output_to_json(response['formatted_output'])

        # 5. Output the result as a dictionary
        self.exam = {
            "topic": self.topic,
            "ccss_input": self.ccss_input,
            "student_reading": response["student_reading"],
            "student_question": response["student_question"],
            "student_rubric": response["student_rubric"],
            "student_response": response["student_response"]
        }

        print(self.exam)

        return self.exam

    def evaluate_student_response(self):
        if self.exam == None:
            return Exception("You must first .take_exam() before you can evaluate it")
        evaluator = StudentResponseEvaluator()
        results = evaluator.evaluate(self.exam)
        print(f"Student Received Score: {results}")

        self.exam['student_evaluation'] = results
        return results

    def show_exam(self):
        return self.exam


# For testing:
if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Simulate a student taking an exam.')
    parser.add_argument('--topic', type=str, help='Topic for the exam.', default="Baseball")
    parser.add_argument('--ccss_input', type=str, help='CCSS input for the exam.', default="CCSS.ELA-LITERACY.W.4.9")
    parser.add_argument('--student_grade', type=str, help='student grade that will be taking the exam .',
                        default="4")
    parser.add_argument('--student_skill_level', type=str,
                        help='student skill level that will be taking the exam.',
                        default="Meets or exceeds standard")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Use the arguments to create a StudentSimulator object and take the exam
    student = Student(grade_level=args.student_grade, skill_level=args.student_skill_level)
    simulator = StudentSimulator(student=student, topic=args.topic, ccss_input=args.ccss_input)
    result = simulator.take_exam(test=True)

    # Display the results
    for key, value in result.items():
        print(f"{key}: {value}\n")
