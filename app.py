import os
import json
import gradio as gr
from groq import Groq

# ============================================
# 1. API & MODEL SETUP
# ============================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "gsk_yahan_apni_groq_key_bhi_daal_sakti_hain")

client = Groq(api_key=GROQ_API_KEY)
MODEL = "openai/gpt-oss-120b"

quiz_data = None

# ============================================
# 2. LEARNFLOW Q&A AGENT
# ============================================
def ask_learnflow(question):
    if not question or not question.strip():
        return "Please enter a question."

    system_prompt = """
You are LearnFlow AI, a friendly personal AI learning companion.
Your job is to help students learn.

Rules:
- Explain concepts in easy language.
- Adjust explanations according to the student's level.
- Use examples when useful.
- Break difficult topics into small steps.
- Do not invent facts or sources.
- If you are uncertain, clearly say so.
- Encourage learning and understanding rather than simply giving answers.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================
# 3. QUIZ AGENT LOGIC
# ============================================
def generate_quiz(
    education_level,
    class_degree,
    subject,
    topic,
    student_level,
    difficulty,
    number_of_questions
):
    global quiz_data

    if not education_level:
        return "❌ Please select an education level.", gr.update(visible=False)
    if not class_degree or not class_degree.strip():
        return "❌ Please enter your class or degree.", gr.update(visible=False)
    if not subject or not subject.strip():
        return "❌ Please enter a subject.", gr.update(visible=False)
    if not topic or not topic.strip():
        return "❌ Please enter a topic.", gr.update(visible=False)

    number_of_questions = int(number_of_questions)

    prompt = f"""
You are an expert educational assessment designer.
Create exactly {number_of_questions} multiple-choice questions.

STUDENT INFORMATION:
Education Level: {education_level}
Class / Degree: {class_degree}
Subject: {subject}
Topic: {topic}
Student Level: {student_level}
Difficulty: {difficulty}

QUESTION RULES:
1. Create exactly {number_of_questions} questions.
2. Every question must be directly related to the selected subject and topic.
3. Every question must have exactly 4 options.
4. All 4 options must be different.
5. There must be exactly ONE correct option.
6. The answer field must contain an integer: 0=A, 1=B, 2=C, 3=D.
7. Use ONLY single dollar signs for inline math LaTeX: $x^2$.
8. Every question MUST have a clear explanation.

Return ONLY valid JSON in exactly this structure:
{{
  "questions": [
    {{
      "question": "question text",
      "options": ["option A", "option B", "option C", "option D"],
      "answer": 0,
      "explanation": "clear explanation"
    }}
  ]
}}
Do not include markdown or ```json.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a highly accurate educational quiz generator. Use single-dollar LaTeX for math. Return only valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        raw_response = response.choices[0].message.content.strip()
        if raw_response.startswith("```"):
            raw_response = raw_response.replace("```json", "").replace("```", "").strip()

        quiz_data = json.loads(raw_response)
        questions = quiz_data.get("questions", [])

        if len(questions) != number_of_questions:
            quiz_data = None
            return "❌ Incorrect number of questions generated.", gr.update(visible=False)

        for q in questions:
            if not all(field in q for field in ["question", "options", "answer", "explanation"]):
                quiz_data = None
                return "❌ Invalid question structure generated.", gr.update(visible=False)
            if len(q["options"]) != 4:
                quiz_data = None
                return "❌ Every question must contain exactly 4 options.", gr.update(visible=False)
            if q["answer"] not in [0, 1, 2, 3]:
                quiz_data = None
                return "❌ Invalid answer index generated.", gr.update(visible=False)

        quiz_data = {"questions": questions}
        return "", gr.update(visible=True)

    except Exception as e:
        quiz_data = None
        return f"❌ Error: {str(e)}", gr.update(visible=False)


def start_quiz(
    education_level,
    class_degree,
    subject,
    topic,
    student_level,
    difficulty,
    number_of_questions
):
    result, visibility = generate_quiz(
        education_level,
        class_degree,
        subject,
        topic,
        student_level,
        difficulty,
        number_of_questions
    )

    outputs = []

    if quiz_data is None:
        for i in range(10):
            outputs.append(gr.update(value="", visible=False))
            outputs.append(gr.update(value="", visible=False))
            outputs.append(gr.update(choices=["A", "B", "C", "D"], value=None, visible=False))
        return tuple(outputs)

    for i in range(10):
        if i < len(quiz_data["questions"]):
            q = quiz_data["questions"][i]
            question_text = f"### Question {i + 1}\n\n{q['question']}"
            options_text = (
                f"**A.** {q['options'][0]}\n\n"
                f"**B.** {q['options'][1]}\n\n"
                f"**C.** {q['options'][2]}\n\n"
                f"**D.** {q['options'][3]}"
            )
            outputs.append(gr.update(value=question_text, visible=True))
            outputs.append(gr.update(value=options_text, visible=True))
            outputs.append(gr.update(choices=["A", "B", "C", "D"], value=None, visible=True))
        else:
            outputs.append(gr.update(value="", visible=False))
            outputs.append(gr.update(value="", visible=False))
            outputs.append(gr.update(choices=["A", "B", "C", "D"], value=None, visible=False))

    return tuple(outputs)


def evaluate_quiz(*answers):
    global quiz_data
    if quiz_data is None:
        return "❌ Please generate a quiz first."

    questions = quiz_data["questions"]
    score = 0
    total = len(questions)
    feedback = []

    for i, question in enumerate(questions):
        student_answer = answers[i]
        correct_answer = question["answer"]

        if student_answer is None:
            feedback.append(
                f"### Question {i+1} ❌\n\n"
                f"**Your answer:** Not attempted\n\n"
                f"**Correct answer:** {chr(65 + correct_answer)}. {question['options'][correct_answer]}\n\n"
                f"**Explanation:**\n{question.get('explanation', 'No explanation available.')}"
            )
        elif student_answer == correct_answer:
            score += 1
            feedback.append(
                f"### Question {i+1} ✅\n\n"
                f"**Your answer:** {chr(65 + student_answer)}. {question['options'][student_answer]}\n\n"
                f"**Explanation:**\n{question.get('explanation', 'No explanation available.')}"
            )
        else:
            feedback.append(
                f"### Question {i+1} ❌\n\n"
                f"**Your answer:** {chr(65 + student_answer)}. {question['options'][student_answer]}\n\n"
                f"**Correct answer:** {chr(65 + correct_answer)}. {question['options'][correct_answer]}\n\n"
                f"**Explanation:**\n{question.get('explanation', 'No explanation available.')}"
            )

    percentage = (score / total) * 100
    result = f"# 📊 Quiz Result\n\n**Score:** {score}/{total}\n\n**Percentage:** {percentage:.0f}%\n\n---\n\n# 🧑‍🏫 Learn From Your Answers\n\n"
    result += "\n\n".join(feedback)
    return result


def submit_quiz(*answers):
    converted_answers = [None if a is None else int(a) for a in answers]
    return evaluate_quiz(*converted_answers)

# ============================================
# 4. STUDY PLANNER LOGIC
# ============================================
def generate_study_plan(
    planner_goal,
    planner_subjects,
    planner_topics,
    planner_hours,
    planner_days,
    planner_difficulty,
    planner_language
):
    if not planner_goal or not planner_goal.strip():
        return "❌ Please enter your study goal."
    if not planner_subjects or not planner_subjects.strip():
        return "❌ Please enter your subjects."
    if not planner_topics or not planner_topics.strip():
        return "❌ Please enter your topics."
    if not planner_hours:
        return "❌ Please enter study hours per day."
    if not planner_days:
        return "❌ Please enter the number of available days."

    prompt = f"""
You are an expert AI study planner. Create a personalized study plan in {planner_language}.

Study Goal: {planner_goal}
Subjects: {planner_subjects}
Topics: {planner_topics}
Study Hours Per Day: {planner_hours}
Days Available: {planner_days}
Difficulty Level: {planner_difficulty}

RULES:
1. Create a realistic plan for exactly {planner_days} days.
2. Consider available study time of {planner_hours} hours per day.
3. Format in Markdown. Include overview, day-by-day plan with tasks/breaks, and study tips.
4. Generate ENTIRE plan in {planner_language}.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful and realistic AI study planner."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================
# 5. GRADIO DASHBOARD
# ============================================
with gr.Blocks(title="LearnFlow AI") as app:

    # ------------------ TAB 1: LEARNFLOW Q&A ------------------
    with gr.Tab("💡 LearnFlow Companion"):
        gr.Markdown("""
# 🧠 LearnFlow AI
### Your Personal AI Learning Companion
**Plan → Learn → Practice → Evaluate → Adapt**
""")
        question_input = gr.Textbox(
            label="What do you want to learn?",
            placeholder="For example: Explain recursion in very simple language.",
            lines=4
        )
        ask_button = gr.Button("Ask LearnFlow AI", variant="primary")
        answer_output = gr.Markdown(label="AI Answer")

        ask_button.click(
            fn=ask_learnflow,
            inputs=question_input,
            outputs=answer_output
        )

    # ------------------ TAB 2: QUIZ AGENT ------------------
    with gr.Tab("📝 AI Quiz Agent"):
        gr.Markdown("### 🎓 Education Information")
        with gr.Row():
            education_level = gr.Dropdown(
                choices=["Primary", "Middle", "Secondary", "Higher Secondary", "Undergraduate", "Graduate", "Doctoral"],
                label="Education Level",
                value="Undergraduate"
            )
            class_degree = gr.Textbox(
                label="Class / Degree",
                placeholder="Example: BS Computer Science, Class 10"
            )

        gr.Markdown("### 📚 Quiz Information")
        with gr.Row():
            subject = gr.Textbox(label="Subject", placeholder="e.g. Mathematics")
            topic = gr.Textbox(label="Topic", placeholder="e.g. Calculus")

        with gr.Row():
            student_level = gr.Dropdown(choices=["Beginner", "Intermediate", "Advanced"], label="Student Level", value="Beginner")
            difficulty = gr.Dropdown(choices=["Easy", "Medium", "Hard"], label="Difficulty", value="Medium")
            number_of_questions = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Questions")

        start_button = gr.Button("🚀 Generate Quiz", variant="primary")

        question_displays = []
        option_displays = []
        answer_boxes = []

        for i in range(10):
            with gr.Group():
                q_disp = gr.Markdown(value="", visible=False, latex_delimiters=[{"left": "$", "right": "$", "display": False}])
                o_disp = gr.Markdown(value="", visible=False, latex_delimiters=[{"left": "$", "right": "$", "display": False}])
                a_box = gr.Radio(choices=["A", "B", "C", "D"], label="Select your answer", type="index", visible=False)
                question_displays.append(q_disp)
                option_displays.append(o_disp)
                answer_boxes.append(a_box)

        submit_button = gr.Button("✅ Submit Quiz", variant="primary")
        result_display = gr.Markdown(latex_delimiters=[{"left": "$", "right": "$", "display": False}])

        start_button.click(
            start_quiz,
            inputs=[education_level, class_degree, subject, topic, student_level, difficulty, number_of_questions],
            outputs=[item for triple in zip(question_displays, option_displays, answer_boxes) for item in triple]
        )

        submit_button.click(
            submit_quiz,
            inputs=answer_boxes,
            outputs=result_display
        )

    # ------------------ TAB 3: STUDY PLANNER ------------------
    with gr.Tab("📅 AI Study Planner"):
        gr.Markdown("### Create your personalized study plan")
        planner_goal = gr.Textbox(label="🎯 Study Goal", placeholder="Example: Prepare for Calculus final exam")
        planner_subjects = gr.Textbox(label="📚 Subjects", placeholder="Example: Calculus, Programming")
        planner_topics = gr.Textbox(label="📝 Topics", placeholder="Example: Integration, Arrays")

        with gr.Row():
            planner_hours = gr.Number(label="⏰ Study Hours Per Day", value=2, minimum=1, maximum=12)
            planner_days = gr.Number(label="📅 Days Available", value=7, minimum=1, maximum=60)
            planner_difficulty = gr.Dropdown(choices=["Easy", "Medium", "Hard"], label="📊 Difficulty Level", value="Medium")
            planner_language = gr.Dropdown(choices=["English", "Urdu", "Roman Urdu"], label="🌐 Preferred Language", value="English")

        generate_plan_button = gr.Button("🤖 Generate Study Plan", variant="primary")
        planner_output = gr.Markdown()

        generate_plan_button.click(
            generate_study_plan,
            inputs=[planner_goal, planner_subjects, planner_topics, planner_hours, planner_days, planner_difficulty, planner_language],
            outputs=planner_output
        )

# ============================================
# 6. LAUNCH
# ============================================
if __name__ == "__main__":
    app.launch()