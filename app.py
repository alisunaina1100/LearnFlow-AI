import os
import json
import streamlit as st
from groq import Groq

# ============================================
# 1. API & MODEL SETUP
# ============================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is not configured.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
MODEL = "openai/gpt-oss-120b"


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
# 3. QUIZ AGENT
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

    if not education_level:
        return None, "❌ Please select an education level."

    if not class_degree or not class_degree.strip():
        return None, "❌ Please enter your class or degree."

    if not subject or not subject.strip():
        return None, "❌ Please enter a subject."

    if not topic or not topic.strip():
        return None, "❌ Please enter a topic."

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
6. The answer field must contain an integer:
   0=A, 1=B, 2=C, 3=D.
7. Use ONLY single dollar signs for inline math LaTeX: $x^2$.
8. Every question MUST have a clear explanation.

Return ONLY valid JSON in exactly this structure:

{{
  "questions": [
    {{
      "question": "question text",
      "options": [
        "option A",
        "option B",
        "option C",
        "option D"
      ],
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
                    "content": (
                        "You are a highly accurate educational quiz generator. "
                        "Use single-dollar LaTeX for math. "
                        "Return only valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1
        )

        raw_response = response.choices[0].message.content.strip()

        if raw_response.startswith("```"):
            raw_response = (
                raw_response
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        quiz_data = json.loads(raw_response)

        questions = quiz_data.get("questions", [])

        if len(questions) != number_of_questions:
            return None, "❌ Incorrect number of questions generated."

        for q in questions:

            if not all(
                field in q
                for field in [
                    "question",
                    "options",
                    "answer",
                    "explanation"
                ]
            ):
                return None, "❌ Invalid question structure generated."

            if len(q["options"]) != 4:
                return None, "❌ Every question must contain exactly 4 options."

            if q["answer"] not in [0, 1, 2, 3]:
                return None, "❌ Invalid answer index generated."

        return quiz_data, ""

    except Exception as e:

        return None, f"❌ Error: {str(e)}"


# ============================================
# 4. QUIZ EVALUATION
# ============================================

def evaluate_quiz(quiz_data, student_answers):

    if quiz_data is None:
        return "❌ Please generate a quiz first."

    questions = quiz_data["questions"]

    score = 0
    total = len(questions)

    feedback = []

    for i, question in enumerate(questions):

        student_answer = student_answers[i]
        correct_answer = question["answer"]

        if student_answer is None:

            feedback.append(
                f"""
### Question {i + 1} ❌

**Your answer:** Not attempted

**Correct answer:** {chr(65 + correct_answer)}. {question["options"][correct_answer]}

**Explanation:**

{question.get("explanation", "No explanation available.")}
"""
            )

        elif student_answer == correct_answer:

            score += 1

            feedback.append(
                f"""
### Question {i + 1} ✅

**Your answer:** {chr(65 + student_answer)}. {question["options"][student_answer]}

**Explanation:**

{question.get("explanation", "No explanation available.")}
"""
            )

        else:

            feedback.append(
                f"""
### Question {i + 1} ❌

**Your answer:** {chr(65 + student_answer)}. {question["options"][student_answer]}

**Correct answer:** {chr(65 + correct_answer)}. {question["options"][correct_answer]}

**Explanation:**

{question.get("explanation", "No explanation available.")}
"""
            )

    percentage = (score / total) * 100

    result = f"""
# 📊 Quiz Result

**Score:** {score}/{total}

**Percentage:** {percentage:.0f}%

---

# 🧑‍🏫 Learn From Your Answers

"""

    result += "\n\n".join(feedback)

    return result


# ============================================
# 5. STUDY PLANNER
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
You are an expert AI study planner.

Create a personalized study plan in {planner_language}.

Study Goal: {planner_goal}
Subjects: {planner_subjects}
Topics: {planner_topics}
Study Hours Per Day: {planner_hours}
Days Available: {planner_days}
Difficulty Level: {planner_difficulty}

RULES:

1. Create a realistic plan for exactly {planner_days} days.
2. Consider available study time of {planner_hours} hours per day.
3. Format in Markdown.
4. Include overview.
5. Include a day-by-day plan.
6. Include tasks and breaks.
7. Include useful study tips.
8. Generate the ENTIRE plan in {planner_language}.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and realistic AI study planner."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except Exception as e:

        return f"❌ Error: {str(e)}"


# ============================================
# 6. STREAMLIT PAGE SETUP
# ============================================

st.set_page_config(
    page_title="LearnFlow AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 LearnFlow AI")
st.subheader("Your Personal AI Learning Companion")

st.markdown(
    "**Plan → Learn → Practice → Evaluate → Adapt**"
)


# ============================================
# 7. TABS
# ============================================

tab1, tab2, tab3 = st.tabs(
    [
        "💡 LearnFlow Companion",
        "📝 AI Quiz Agent",
        "📅 AI Study Planner"
    ]
)


# ============================================
# TAB 1 — Q&A
# ============================================

with tab1:

    st.header("💡 LearnFlow Companion")

    question = st.text_area(
        "What do you want to learn?",
        placeholder=(
            "For example: Explain recursion "
            "in very simple language."
        ),
        height=150
    )

    if st.button(
        "🤖 Ask LearnFlow AI",
        key="ask_button"
    ):

        with st.spinner("LearnFlow AI is thinking..."):

            answer = ask_learnflow(question)

        st.markdown("### 🧑‍🏫 AI Answer")
        st.markdown(answer)


# ============================================
# TAB 2 — QUIZ
# ============================================

with tab2:

    st.header("📝 AI Quiz Agent")

    st.markdown("### 🎓 Education Information")

    education_level = st.selectbox(
        "Education Level",
        [
            "Primary",
            "Middle",
            "Secondary",
            "Higher Secondary",
            "Undergraduate",
            "Graduate",
            "Doctoral"
        ],
        index=4
    )

    class_degree = st.text_input(
        "Class / Degree",
        placeholder="Example: BS Computer Science, Class 10"
    )

    st.markdown("### 📚 Quiz Information")

    col1, col2 = st.columns(2)

    with col1:

        subject = st.text_input(
            "Subject",
            placeholder="e.g. Mathematics"
        )

    with col2:

        topic = st.text_input(
            "Topic",
            placeholder="e.g. Calculus"
        )

    col3, col4, col5 = st.columns(3)

    with col3:

        student_level = st.selectbox(
            "Student Level",
            [
                "Beginner",
                "Intermediate",
                "Advanced"
            ]
        )

    with col4:

        difficulty = st.selectbox(
            "Difficulty",
            [
                "Easy",
                "Medium",
                "Hard"
            ],
            index=1
        )

    with col5:

        number_of_questions = st.slider(
            "Number of Questions",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )

    if st.button(
        "🚀 Generate Quiz",
        key="generate_quiz_button"
    ):

        with st.spinner("Generating your quiz..."):

            quiz_data, error = generate_quiz(
                education_level,
                class_degree,
                subject,
                topic,
                student_level,
                difficulty,
                number_of_questions
            )

        if error:

            st.error(error)

        else:

            st.session_state.quiz_data = quiz_data
            st.session_state.quiz_submitted = False

            # Reset previous answers
            st.session_state.quiz_answers = [
                None for _ in quiz_data["questions"]
            ]

            st.success("✅ Quiz generated! Attempt the questions below.")


    # ----------------------------------------
    # DISPLAY QUIZ
    # ----------------------------------------

    if "quiz_data" in st.session_state:

        quiz_data = st.session_state.quiz_data

        st.markdown("---")
        st.markdown("## 📝 Attempt Your Quiz")

        for i, q in enumerate(quiz_data["questions"]):

            st.markdown(
                f"### Question {i + 1}"
            )

            st.markdown(
                q["question"]
            )

            selected = st.radio(
                "Select your answer:",
                ["A", "B", "C", "D"],
                index=None,
                key=f"quiz_answer_{i}"
            )

            if selected is not None:

                st.session_state.quiz_answers[i] = (
                    ord(selected) - ord("A")
                )

        if st.button(
            "✅ Submit Quiz",
            key="submit_quiz_button"
        ):

            st.session_state.quiz_submitted = True

        # ------------------------------------
        # SHOW RESULT ONLY AFTER SUBMIT
        # ------------------------------------

        if st.session_state.get(
            "quiz_submitted",
            False
        ):

            result = evaluate_quiz(
                quiz_data,
                st.session_state.quiz_answers
            )

            st.markdown("---")
            st.markdown(result)


# ============================================
# TAB 3 — STUDY PLANNER
# ============================================

with tab3:

    st.header("📅 AI Study Planner")

    st.markdown(
        "### Create your personalized study plan"
    )

    planner_goal = st.text_input(
        "🎯 Study Goal",
        placeholder="Example: Prepare for Calculus final exam"
    )

    planner_subjects = st.text_input(
        "📚 Subjects",
        placeholder="Example: Calculus, Programming"
    )

    planner_topics = st.text_input(
        "📝 Topics",
        placeholder="Example: Integration, Arrays"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        planner_hours = st.number_input(
            "⏰ Study Hours Per Day",
            min_value=1,
            max_value=12,
            value=2
        )

    with col2:

        planner_days = st.number_input(
            "📅 Days Available",
            min_value=1,
            max_value=60,
            value=7
        )

    with col3:

        planner_difficulty = st.selectbox(
            "📊 Difficulty Level",
            [
                "Easy",
                "Medium",
                "Hard"
            ]
        )

    with col4:

        planner_language = st.selectbox(
            "🌐 Preferred Language",
            [
                "English",
                "Urdu",
                "Roman Urdu"
            ]
        )

    if st.button(
        "🤖 Generate Study Plan",
        key="generate_plan_button"
    ):

        with st.spinner(
            "Creating your personalized study plan..."
        ):

            plan = generate_study_plan(
                planner_goal,
                planner_subjects,
                planner_topics,
                planner_hours,
                planner_days,
                planner_difficulty,
                planner_language
            )

        st.markdown("---")
        st.markdown("## 📅 Your Study Plan")
        st.markdown(plan)
