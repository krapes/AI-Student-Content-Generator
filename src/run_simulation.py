from StudentSimulator import StudentSimulator
from Student import Student
from RandInputGenerator import RandInputGenerator
from datetime import datetime
import random
import re
import pickle

start_time = datetime.now()
generator = RandInputGenerator()

N = 5
TARGET_SAMPLE_SIZE = 200

DEFAULT_RESULT = {
    "error": None,
    "correctly_formatted_student_content_output": None,
    "correctly_formatted_final_output": None,
    "standard_rubric_mismatch": None,
    "difficulty": None,
    "iteration_identifier": None,
    "total_runtime": None
}


def extract_standard_grade(ccss: str) -> str:
    """Extracts and returns the standard grade from a given CCSS string."""
    pattern = r'.*\.(.*?)\.[^\.]*$'
    matches = re.search(pattern, ccss)
    return matches.group(1) if matches else None


def get_student(standard_grade: str) -> tuple:
    """Generates a student profile and expected score based on the given standard grade."""
    random_value = random.random()
    if random_value < 0.5:
        random_value = random.random()
        skill_level = ["Meets or exceeds standard", "Partially meets standard", "Doesn't meet standard"]
        expected_scores = [1, 0.5, 0]
        thresholds = [0.5, 0.75, 1]
        for i in range(3):
            if random_value < thresholds[i]:
                student = Student(grade_level=standard_grade, skill_level=skill_level[i])
                expected_score = expected_scores[i]
                break
    else:
        grade_levels = ['K', '11-12']
        skill_levels = ["Doesn't meet standard", "Meets or exceeds standard"]
        expected_scores = [0, 1]
        thresholds = [0.75, 1]
        for i in range(2):
            if random_value < thresholds[i]:
                student = Student(grade_level=grade_levels[i], skill_level=skill_levels[i])
                expected_score = expected_scores[i]
                break
    return student, expected_score

def save_pickle(filename: str, file: object) -> None:
    """Saves the given file as a pickle file with the specified filename."""
    with open(filename, 'wb') as f:
        pickle.dump(file, f)


def test_correctly_formatted_student_content_output(exam: dict) -> bool:
    """Tests if the student content output is correctly formatted."""
    try:
        assert all(key in exam for key in ['student_reading', 'student_question', 'student_rubric'])
        return True
    except Exception as e:
        print(f"Breaking for incorrect student content output: {str(e)} \nKeys present: {exam.keys()}")
        return False


def test_standard_rubric_mismatch(standard: str, ccss_input: str):
    result = True
    if standard.lower() == ccss_input.lower():
        return False
    try:
        pattern = r'(CCSS\.(ELA-Literacy|Math|ELA-LITERACY|MATH)\.(\D+)\.(K|[1-8]|9-10|11-12)\.(\d+))(.*)'
        match = re.findall(pattern, ccss_input)
        standard, subject, ccss_skill, ccss_grade, ccss_numeration, description = match[0]

        # Regex pattern to capture the standard and its description
        pattern = r'(CCSS\.(ELA-Literacy|Math|ELA-LITERACY|MATH)\.(\D+)\.(K|[1-8]|9-10|11-12)\.(\d+))(.*)'
        matches = re.findall(pattern, standard)
        for match in matches:
            standard, subject, skill, grade, numeration, description = match
            if ccss_skill.lower() == skill.lower() and ccss_grade == grade and ccss_numeration == numeration:
                return False

        else:
            print(f"standard {type(standard)}: {standard} ccss_input ({type(ccss_input)}): {ccss_input}")
    except Exception as e:
        print(f"Error in determining standard mismatch \n {e}")
        print(f"standard {type(standard)}: {standard} ccss_input ({type(ccss_input)}): {ccss_input}")
    return result


def test_student_rubric_completeness(student_rubric: dict):
    result = False
    try:
        if type(student_rubric['Definitions']["Doesn't meet standard"]) == str \
                and type(student_rubric['Definitions']["Partially meets standard"]) == str \
                and type(student_rubric['Definitions']["Meets or exceeds standard"]) == str \
                and type(student_rubric["Examples"]["Doesn't meet standard"]) == str \
                and type(student_rubric["Examples"]["Partially meets standard"]) == str \
                and type(student_rubric["Examples"]["Doesn't meet standard"]) == str \
                and type(student_rubric["Standard"]) == str:
            result = True
    except Exception as e:
        print(f"Breaking for rubric_completeness error \n {str(e)} \n {student_rubric.keys()}")
    return result

def test_correctly_formatted_final_output(exam_scores: dict) -> bool:
    """Tests if the final output is correctly formatted."""
    try:
        assert all(key in exam_scores for key in ['student_evaluation', 'student_score'])
        return True
    except Exception as e:
        print(f"Exception on exam score: {e}")
        return False

def main():
    print("Start...")
    results = []

    while len(results) < TARGET_SAMPLE_SIZE:
        save_pickle(f'../data/simulation_results_{start_time.strftime("%H:%M:%S")}', results)
        topic, ccss_input = generator.rand_input()
        print("Randomly generated topic:", topic)
        print("Randomly selected CCSS standard:", ccss_input)
        standard_grade = extract_standard_grade(ccss_input)
        for i in range(N):
            print(f"Running Simulation {i} of {N} on this student content")
            results.append(DEFAULT_RESULT.copy())
            result = results[-1]
            result["iteration_identifier"] = i
            try:
                student, expected_score = get_student(standard_grade)
                print(f"Student: {student}, Expected Score: {expected_score}")
                simulation = StudentSimulator(student=student, topic=topic, ccss_input=ccss_input)
                exam = simulation.receive_exam(test=False)
                result['error'] = False
                result['total_runtime'] = (datetime.now() - start_time)
                result["correctly_formatted_student_content_output"] = test_correctly_formatted_student_content_output(
                    exam)
                if not result["correctly_formatted_student_content_output"]:
                    break
                student_rubric = exam['student_rubric']
                result['rubric_complete'] = test_student_rubric_completeness(student_rubric)
                if not result['rubric_complete']:
                    break
                result["standard_rubric_mismatch"] = test_standard_rubric_mismatch(student_rubric['Standard'],
                                                                                   ccss_input)
                if result["standard_rubric_mismatch"]:
                    break
                simulation.take_exam(test=False)
                exam_scores = simulation.evaluate_student_response()
                result["correctly_formatted_final_output"] = test_correctly_formatted_final_output(exam_scores)
                if not result["correctly_formatted_final_output"]:
                    break

                exam_scores = exam_scores["student_evaluation"]
                student_score = exam_scores["student_score"]
                result["difficulty"] = expected_score - student_score


            except Exception as e:
                result['error'] = True
                print(f"Breaking for error \n {e}")

        try:
            print(f"exam scores: \n {exam_scores}")
        except:
            pass
        print(result)


if __name__ == '__main__':
    main()
