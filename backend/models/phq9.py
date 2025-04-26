from datetime import datetime
import uuid

class PHQ9Response:
    # PHQ-9 questions for reference and validation
    QUESTIONS = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself - or that you are a failure or have let yourself or your family down",
        "Trouble concentrating on things, such as reading the newspaper or watching television",
        "Moving or speaking so slowly that other people could have noticed. Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual",
        "Thoughts that you would be better off dead or of hurting yourself in some way"
    ]

    SEVERITY_LEVELS = {
        "minimal": (0, 4),
        "mild": (5, 9),
        "moderate": (10, 14),
        "moderately_severe": (15, 19),
        "severe": (20, 27)
    }

    def __init__(self, user_id, answers):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        # Convert simple array to structured answers
        self.answers = [
            {
                "question_idx": idx,
                "answer": score,
                "question_text": self.QUESTIONS[idx]
            }
            for idx, score in enumerate(answers)
        ]
        self.total_score = sum(answers)
        self.severity = self._calculate_severity()
        self.created_at = datetime.utcnow()

    def _calculate_severity(self):
        score = self.total_score
        for level, (min_score, max_score) in self.SEVERITY_LEVELS.items():
            if min_score <= score <= max_score:
                return level
        return "unknown"

    @staticmethod
    def validate_answers(answers):
        errors = []
        
        if not isinstance(answers, list):
            return ["Answers must be provided as a list of numbers"]
        
        if len(answers) != len(PHQ9Response.QUESTIONS):
            return [f"Must provide exactly {len(PHQ9Response.QUESTIONS)} answers"]

        for idx, answer in enumerate(answers):
            if not isinstance(answer, int):
                errors.append(f"Answer {idx + 1} must be a number")
                continue

            if not 0 <= answer <= 3:
                errors.append(f"Answer {idx + 1} must be between 0 and 3")

        return errors

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "answers": self.answers,
            "total_score": self.total_score,
            "severity": self.severity,
            "created_at": self.created_at
        }

    @staticmethod
    def from_dict(data):
        # Extract simple array of answers from stored format
        answers = [answer["answer"] for answer in sorted(
            data.get("answers", []), 
            key=lambda x: x["question_idx"]
        )]
        
        response = PHQ9Response(
            user_id=data.get("user_id"),
            answers=answers
        )
        if "id" in data:
            response.id = data["id"]
        if "created_at" in data:
            response.created_at = data["created_at"]
        return response 