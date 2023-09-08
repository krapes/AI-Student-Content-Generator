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

    def __init__(self):
        # Load environment variables from the .env file.
        load_dotenv(find_dotenv())
        self.llm = ChatOpenAI(temperature=0.9, model=self.DEFAULT_MODEL)  # Initialize LLM mode

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

        prompt_template = """
        You are a teacher evaluating an answer a student gave in your class. 
        
        The student was shown STUDENT READING and asked to respond to STUDENT QUESTION.
        
        Now compare their response in STUDENT RESPONSE to the rubric provided in STUDENT RUBRIC and given the student 
        a valuation.
        
        STUDENT READING
        ###
        {student_reading}
        ###
        
        STUDENT QUESTION
        ###
        {student_question}
        ###
        
        STUDENT RESPONSE
        ###
        {student_response}
        ###
        
        STUDENT RUBRIC
        ###
        {student_rubric}
        ###
        
        Use the following procedure to complete the task:
        1. Read STUDENT READING
        2. Read STUDENT QUESTION
        3. Read STUDENT RESPONSE
        4. The valuation option will be keys in the Values dict in the STUDENT RUBRIC. 
        Example: "Meets or exceeds standard", "Partially meets standard", "Doesn't meet standard" 
        What are the valuation options in this rubric? 
        5. Read the STUDENT RUBRIC ["Definitions"] for each of the valuations
        6. Now compare the STUDENT RESPONSE to the STUDENT RUBRIC ["Examples"]. Keeping the definitions you read in
        step 5 in mind is the STUDENT RESPONSE better, worse, or the same as the example for "Doesn't meet standard" 
        7. Now compare the STUDENT RESPONSE to the STUDENT RUBRIC ["Examples"]. Keeping the definitions you read in
        step 5 in mind is the STUDENT RESPONSE better, worse, or the same as the example for "Partially meets standard"
        8.  Now compare the STUDENT RESPONSE to the STUDENT RUBRIC ["Examples"]. Keeping the definitions you read in
        step 5 in mind is the STUDENT RESPONSE better, worse, or the same as the example for "Meets or exceeds standard"
        9. Given your answers for steps 6, 7 and 8 what do you think is the best evaluation of this student response?
        10. Given the evaluation in step 9 what is the score the student should receive? Scores are found in 
        STUDENT RUBRIC ["Values"]. 
        11. Output a single VALID json example provided below. Do not return any other text that is not this json
        Example output
        {{{{ 'student_evaluation': "Doesn't meet standard" , 'student_score': 0 }}}}
        
        """

        format_output_template = """Please format the Input below into a single, valid, json. An example of the 
        output is provide. Do not include any text in your output that is not part of this json. The json MUST BE 
        VALID. Do not include any line breaks in your response.

               Output Example
               {{{{"student_evaluation": "Meets or exceeds standard", "student_score": 1}}}}


               INPUT
               {student_response}
               """

        print("Generating Student Evaluation")
        generate_student_response = LLMChain(llm=self.llm,
                                             prompt=ChatPromptTemplate.from_template(prompt_template),
                                             output_key="student_evaluation")
        format_output = LLMChain(llm=self.llm,
                                 prompt=ChatPromptTemplate.from_template(format_output_template),
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
