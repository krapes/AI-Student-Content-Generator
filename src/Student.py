class Student:
    VALID_GRADE_LEVELS = ["K", "1", "2", "3", "4", "5", "6", "7", "8", "9-10", "11-12"]
    VALID_SKILL_LEVELS = [
        "Doesn't meet standard",
        "Partially meets standard",
        "Meets or exceeds standard",
    ]

    def __init__(self, grade_level: str, skill_level: str):
        if grade_level not in self.VALID_GRADE_LEVELS:
            raise ValueError(f"Invalid grade level. Must be one of {self.VALID_GRADE_LEVELS}")

        if skill_level not in self.VALID_SKILL_LEVELS:
            raise ValueError(f"Invalid skill level. Must be one of {self.VALID_SKILL_LEVELS}")

        self.grade_level = grade_level
        self.skill_level = skill_level

    def __str__(self):
        return f'Student(grade_level: {self.grade_level}, skill_level: "{self.skill_level}")'

    def __repr__(self):
        return self.__str__()


# Usage example
if __name__ == "__main__":
    try:
        # Create a Student object
        student = Student(grade_level="9-10", skill_level="Advanced")
    except ValueError as e:
        print(e)

    # Create a valid Student object
    valid_student = Student(grade_level="9-10", skill_level="Meets or exceeds standard")
    print(valid_student)
