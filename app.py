import os
import json
import sqlite3
from datetime import datetime

import streamlit as st
from groq import Groq


# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="LearnFlow AI",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# 2. PROFESSIONAL UI
# =========================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        background: linear-gradient(
            135deg,
            rgba(99,102,241,0.15),
            rgba(59,130,246,0.10)
        );
        border: 1px solid rgba(99,102,241,0.25);
    }

    .hero h1 {
        margin-bottom: 0.3rem;
    }

    .metric-card {
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        background: rgba(128,128,128,0.05);
    }

    .metric-number {
        font-size: 2rem;
        font-weight: 700;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.75;
    }

    .weak-topic {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.6rem;
        border-left: 5px solid #ef4444;
        background: rgba(239,68,68,0.08);
    }

    .recommendation {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.6rem;
        border-left: 5px solid #3b82f6;
        background: rgba(59,130,246,0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. API & MODEL SETUP
# =========================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is not configured.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

MODEL = "openai/gpt-oss-120b"


# =========================================================
# 4. DATABASE / MEMORY SYSTEM
# =========================================================

DB_NAME = "learnflow_memory.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def init_database():

    conn = get_connection()
    cursor = conn.cursor()

    # Chat memory
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student TEXT,
            question TEXT,
            answer TEXT,
            created_at TEXT
        )
        """
    )

    # Quiz history
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS quiz_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student TEXT,
            subject TEXT,
            topic TEXT,
            difficulty TEXT,
            score INTEGER,
            total INTEGER,
            percentage REAL,
            wrong_topics TEXT,
            created_at TEXT
        )
        """
    )

    # Study plans
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS study_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student TEXT,
            goal TEXT,
            subjects TEXT,
            topics TEXT,
            plan TEXT,
            created_at TEXT
        )
        """
    )

    conn.commit()
    conn.close()


init_database()


# =========================================================
# 5. STUDENT MEMORY
# =========================================================

if "student_name" not in st.session_state:
    st.session_state.student_name = "Student"


with st.sidebar:

    st.markdown("## 👤 Student Profile")

    student_name = st.text_input(
        "Your Name",
        value=st.session_state.student_name
    )

    if student_name.strip():
        st.session_state.student_name = student_name.strip()

    st.markdown("---")

    st.markdown("### 🧠 Your Learning Memory")

    st.caption(
        "LearnFlow saves your quiz results and learning conversations "
        "to build personalized recommendations."
    )


# =========================================================
# 6. MEMORY FUNCTIONS
# =========================================================

def save_chat(student, question, answer):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO chat_history
        (student, question, answer, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            student,
            question,
            answer,
            datetime.now().isoformat()
        )
    )

    conn.commit()
    conn.close()


def get_chat_history(student, limit=20):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT question, answer, created_at
        FROM chat_history
        WHERE student = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (student, limit)
    )

    rows = cursor.fetchall()

    conn.close()

    return rows


def save_quiz_result(
    student,
    subject,
    topic,
    difficulty,
    score,
    total,
    wrong_topics
):

    percentage = (score / total) * 100 if total else 0

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO quiz_history
        (
            student,
            subject,
            topic,
            difficulty,
            score,
            total,
            percentage,
            wrong_topics,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            student,
            subject,
            topic,
            difficulty,
            score,
            total,
            percentage,
            json.dumps(wrong_topics),
            datetime.now().isoformat()
        )
    )

    conn.commit()
    conn.close()


def get_quiz_history(student):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            subject,
            topic,
            difficulty,
            score,
            total,
            percentage,
            wrong_topics,
            created_at
        FROM quiz_history
        WHERE student = ?
        ORDER BY id DESC
        """,
        (student,)
    )

    rows = cursor.fetchall()

    conn.close()

    return rows


def save_study_plan(
    student,
    goal,
    subjects,
    topics,
    plan
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO study_plans
        (
            student,
            goal,
            subjects,
            topics,
            plan,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            student,
            goal,
            subjects,
            topics,
            plan,
            datetime.now().isoformat()
        )
    )

    conn.commit()
    conn.close()


# =========================================================
# 7. LEARNFLOW Q&A AGENT
# =========================================================

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
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"❌ Error: {str(e)}"


# =========================================================
# 8. QUIZ GENERATOR
# =========================================================

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

    if not class_degree.strip():
        return None, "❌ Please enter your class or degree."

    if not subject.strip():
        return None, "❌ Please enter a subject."

    if not topic.strip():
        return None, "❌ Please enter a topic."

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
2. Every question must be related to the selected subject and topic.
3. Every question must have exactly 4 options.
4. All 4 options must be different.
5. There must be exactly ONE correct option.
6. The answer field must contain an integer:
   0=A, 1=B, 2=C, 3=D.
7. Use ONLY single dollar signs for inline math LaTeX.
8. Every question must have a clear explanation.

Return ONLY valid JSON:

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

Do not include markdown.
Do not include ```json.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a highly accurate educational quiz generator. "
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

            required_fields = [
                "question",
                "options",
                "answer",
                "explanation"
            ]

            if not all(field in q for field in required_fields):

                return None, "❌ Invalid question structure."

            if len(q["options"]) != 4:

                return None, "❌ Every question needs 4 options."

            if q["answer"] not in [0, 1, 2, 3]:

                return None, "❌ Invalid answer index."

        return quiz_data, ""

    except Exception as e:

        return None, f"❌ Error: {str(e)}"


# =========================================================
# 9. QUIZ EVALUATION
# =========================================================

def evaluate_quiz(quiz_data, student_answers):

    questions = quiz_data["questions"]

    score = 0
    total = len(questions)

    feedback = []
    wrong_topics = []

    for i, question in enumerate(questions):

        student_answer = student_answers[i]

        correct_answer = question["answer"]

        if student_answer == correct_answer:

            score += 1

            feedback.append(
                f"""
### Question {i + 1} ✅

**Your answer:** {chr(65 + student_answer)}.
{question["options"][student_answer]}

**Explanation:**

{question.get("explanation", "No explanation available.")}
"""
            )

        elif student_answer is None:

            wrong_topics.append(
                f"Question {i + 1}"
            )

            feedback.append(
                f"""
### Question {i + 1} ❌

**Your answer:** Not attempted

**Correct answer:** {chr(65 + correct_answer)}.
{question["options"][correct_answer]}

**Explanation:**

{question.get("explanation", "No explanation available.")}
"""
            )

        else:

            wrong_topics.append(
                f"Question {i + 1}"
            )

            feedback.append(
                f"""
### Question {i + 1} ❌

**Your answer:** {chr(65 + student_answer)}.
{question["options"][student_answer]}

**Correct answer:** {chr(65 + correct_answer)}.
{question["options"][correct_answer]}

**Explanation:**

{question.get("explanation", "No explanation available.")}
"""
            )

    percentage = (score / total) * 100 if total else 0

    return score, total, percentage, wrong_topics, feedback


# =========================================================
# 10. WEAK TOPIC ANALYSIS
# =========================================================

def get_weak_topics(student):

    history = get_quiz_history(student)

    topic_stats = {}

    for row in history:

        subject = row[0]
        topic = row[1]
        score = row[3]
        total = row[4]

        key = f"{subject} — {topic}"

        if key not in topic_stats:

            topic_stats[key] = {
                "score": 0,
                "total": 0,
                "attempts": 0
            }

        topic_stats[key]["score"] += score
        topic_stats[key]["total"] += total
        topic_stats[key]["attempts"] += 1

    weak_topics = []

    for topic, data in topic_stats.items():

        percentage = (
            data["score"] / data["total"] * 100
            if data["total"]
            else 0
        )

        if percentage < 70:

            weak_topics.append(
                {
                    "topic": topic,
                    "percentage": percentage,
                    "attempts": data["attempts"]
                }
            )

    weak_topics.sort(
        key=lambda x: x["percentage"]
    )

    return weak_topics


# =========================================================
# 11. PERSONALIZED RECOMMENDATIONS
# =========================================================

def generate_recommendations(student):

    weak_topics = get_weak_topics(student)

    if not weak_topics:

        return [
            "Keep practicing different topics to build a strong foundation.",
            "Try medium or hard difficulty questions when you feel ready.",
            "Review your quiz explanations after every attempt."
        ]

    recommendations = []

    for item in weak_topics[:5]:

        topic = item["topic"]
        percentage = item["percentage"]

        recommendations.append(
            f"""
**{topic}**

Current performance: {percentage:.0f}%

➡️ Review the basic concepts first.

➡️ Practice 5–10 easy questions.

➡️ Then move to medium difficulty.

➡️ Review every incorrect answer carefully.
"""
        )

    return recommendations


# =========================================================
# 12. STUDY PLANNER
# =========================================================

def generate_study_plan(
    planner_goal,
    planner_subjects,
    planner_topics,
    planner_hours,
    planner_days,
    planner_difficulty,
    planner_language
):

    if not planner_goal.strip():
        return "❌ Please enter your study goal."

    if not planner_subjects.strip():
        return "❌ Please enter your subjects."

    if not planner_topics.strip():
        return "❌ Please enter your topics."

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
2. Consider {planner_hours} hours per day.
3. Format using Markdown.
4. Include an overview.
5. Include a day-by-day plan.
6. Include tasks and breaks.
7. Include useful study tips.
8. Generate the entire plan in {planner_language}.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI study planner."
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


# =========================================================
# 13. HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">

    <h1>🧠 LearnFlow AI</h1>

    <p style="font-size:1.2rem;">
    Your Personal AI Learning Companion
    </p>

    <p>
    <b>Plan → Learn → Practice → Evaluate → Adapt</b>
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 14. DASHBOARD
# =========================================================

quiz_history = get_quiz_history(
    st.session_state.student_name
)

weak_topics = get_weak_topics(
    st.session_state.student_name
)

total_quizzes = len(quiz_history)

if total_quizzes:

    total_score = sum(row[3] for row in quiz_history)
    total_questions = sum(row[4] for row in quiz_history)

    overall_percentage = (
        total_score / total_questions * 100
        if total_questions
        else 0
    )

else:

    overall_percentage = 0


m1, m2, m3, m4 = st.columns(4)

with m1:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-number">{total_quizzes}</div>
        <div class="metric-label">Quizzes Completed</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-number">{overall_percentage:.0f}%</div>
        <div class="metric-label">Overall Score</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m3:

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-number">{len(weak_topics)}</div>
        <div class="metric-label">Weak Topics</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m4:

    if overall_percentage >= 80:
        level = "Excellent"
    elif overall_percentage >= 60:
        level = "Good"
    elif overall_percentage > 0:
        level = "Needs Practice"
    else:
        level = "New Learner"

    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-number">{level}</div>
        <div class="metric-label">Learning Status</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("")


# =========================================================
# 15. TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "💡 AI Tutor",
        "📝 AI Quiz",
        "📅 Study Planner",
        "📈 Progress",
        "🧠 My Learning Memory"
    ]
)


# =========================================================
# TAB 1 — AI TUTOR
# =========================================================

with tab1:

    st.header("💡 LearnFlow Companion")

    question = st.text_area(
        "What do you want to learn?",
        placeholder=(
            "Example: Explain recursion in very simple language."
        ),
        height=150
    )

    if st.button(
        "🤖 Ask LearnFlow AI",
        key="ask_button"
    ):

        with st.spinner(
            "LearnFlow AI is thinking..."
        ):

            answer = ask_learnflow(question)

        save_chat(
            st.session_state.student_name,
            question,
            answer
        )

        st.markdown("### 🧑‍🏫 AI Answer")

        st.markdown(answer)

        st.success(
            "💾 This conversation has been saved to your learning memory."
        )


# =========================================================
# TAB 2 — QUIZ
# =========================================================

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
        placeholder="Example: BS Computer Science"
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
            value=5
        )

    if st.button(
        "🚀 Generate Quiz",
        key="generate_quiz_button"
    ):

        with st.spinner(
            "Generating your quiz..."
        ):

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

            st.session_state.quiz_answers = [
                None
                for _ in quiz_data["questions"]
            ]

            st.session_state.quiz_saved = False

            st.success(
                "✅ Quiz generated! Attempt all questions."
            )


    # ---------------------------------------------
    # DISPLAY QUIZ
    # ---------------------------------------------

    if "quiz_data" in st.session_state:

        quiz_data = st.session_state.quiz_data

        st.markdown("---")

        st.markdown("## 📝 Attempt Your Quiz")

        for i, q in enumerate(
            quiz_data["questions"]
        ):

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


        # -----------------------------------------
        # RESULT
        # -----------------------------------------

        if st.session_state.get(
            "quiz_submitted",
            False
        ):

            score, total, percentage, wrong_topics, feedback = (
                evaluate_quiz(
                    quiz_data,
                    st.session_state.quiz_answers
                )
            )

            # Save only once
            if not st.session_state.get(
                "quiz_saved",
                False
            ):

                save_quiz_result(
                    st.session_state.student_name,
                    subject,
                    topic,
                    difficulty,
                    score,
                    total,
                    wrong_topics
                )

                st.session_state.quiz_saved = True


            st.markdown("---")

            st.markdown("## 📊 Quiz Result")

            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric(
                    "Score",
                    f"{score}/{total}"
                )

            with r2:
                st.metric(
                    "Percentage",
                    f"{percentage:.0f}%"
                )

            with r3:

                if percentage >= 80:
                    status = "Excellent 🎉"
                elif percentage >= 60:
                    status = "Good 👍"
                else:
                    status = "Needs Practice 📚"

                st.metric(
                    "Status",
                    status
                )


            if percentage < 70:

                st.warning(
                    "🧠 LearnFlow detected that this topic "
                    "may need more practice."
                )

            else:

                st.success(
                    "🎉 Great work! Keep practicing to strengthen "
                    "your knowledge."
                )


            st.markdown(
                "## 🧑‍🏫 Learn From Your Answers"
            )

            st.markdown(
                "\n\n".join(feedback)
            )

            # -------------------------------------
            # PERSONALIZED RECOMMENDATIONS
            # -------------------------------------

            st.markdown("---")

            st.markdown(
                "## 🔄 Personalized Recommendations"
            )

            recommendations = generate_recommendations(
                st.session_state.student_name
            )

            for recommendation in recommendations:

                st.markdown(
                    f"""
                    <div class="recommendation">
                    {recommendation}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =========================================================
# TAB 3 — STUDY PLANNER
# =========================================================

with tab3:

    st.header("📅 AI Study Planner")

    st.markdown(
        "Create your personalized study plan."
    )

    planner_goal = st.text_input(
        "🎯 Study Goal",
        placeholder="Prepare for Calculus final exam"
    )

    planner_subjects = st.text_input(
        "📚 Subjects",
        placeholder="Calculus, Programming"
    )

    planner_topics = st.text_input(
        "📝 Topics",
        placeholder="Integration, Arrays"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        planner_hours = st.number_input(
            "⏰ Hours Per Day",
            min_value=1,
            max_value=12,
            value=2
        )

    with col2:

        planner_days = st.number_input(
            "📅 Days",
            min_value=1,
            max_value=60,
            value=7
        )

    with col3:

        planner_difficulty = st.selectbox(
            "📊 Difficulty",
            [
                "Easy",
                "Medium",
                "Hard"
            ]
        )

    with col4:

        planner_language = st.selectbox(
            "🌐 Language",
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

        save_study_plan(
            st.session_state.student_name,
            planner_goal,
            planner_subjects,
            planner_topics,
            plan
        )

        st.markdown("---")

        st.markdown(
            "## 📅 Your Study Plan"
        )

        st.markdown(plan)


# =========================================================
# TAB 4 — PROGRESS
# =========================================================

with tab4:

    st.header("📈 Student Progress")

    history = get_quiz_history(
        st.session_state.student_name
    )

    if not history:

        st.info(
            "📝 Complete your first quiz to start "
            "tracking your progress."
        )

    else:

        st.markdown(
            "### 📊 Quiz History"
        )

        for row in history:

            subject = row[0]
            topic = row[1]
            difficulty = row[2]
            score = row[3]
            total = row[4]
            percentage = row[5]
            date = row[7]

            with st.expander(
                f"{subject} — {topic} | {percentage:.0f}%"
            ):

                st.write(
                    f"**Difficulty:** {difficulty}"
                )

                st.write(
                    f"**Score:** {score}/{total}"
                )

                st.write(
                    f"**Percentage:** {percentage:.0f}%"
                )

                st.write(
                    f"**Date:** {date[:19]}"
                )


        st.markdown("---")

        st.markdown(
            "### 🧠 Automatically Identified Weak Topics"
        )

        weak_topics = get_weak_topics(
            st.session_state.student_name
        )

        if weak_topics:

            for item in weak_topics:

                st.markdown(
                    f"""
                    <div class="weak-topic">

                    <b>{item["topic"]}</b>

                    <br>

                    Performance:
                    {item["percentage"]:.0f}%

                    <br>

                    Attempts:
                    {item["attempts"]}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.success(
                "🎉 No major weak topics detected yet!"
            )


        st.markdown("---")

        st.markdown(
            "### 🔄 What Should You Study Next?"
        )

        recommendations = generate_recommendations(
            st.session_state.student_name
        )

        for recommendation in recommendations:

            st.markdown(
                f"""
                <div class="recommendation">
                {recommendation}
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# TAB 5 — MEMORY
# =========================================================

with tab5:

    st.header("🧠 My Learning Memory")

    st.markdown(
        "Your previous learning activity is stored here."
    )


    # -----------------------------------------
    # CHAT MEMORY
    # -----------------------------------------

    st.markdown(
        "## 💬 Previous AI Conversations"
    )

    chats = get_chat_history(
        st.session_state.student_name
    )

    if chats:

        for question, answer, created_at in chats:

            with st.expander(
                f"💬 {question[:80]}"
            ):

                st.markdown(
                    f"**You asked:**\n\n{question}"
                )

                st.markdown("---")

                st.markdown(
                    f"**LearnFlow AI:**\n\n{answer}"
                )

                st.caption(
                    created_at[:19]
                )

    else:

        st.info(
            "No saved conversations yet."
        )


    # -----------------------------------------
    # QUIZ MEMORY
    # -----------------------------------------

    st.markdown("---")

    st.markdown(
        "## 📝 Quiz Memory"
    )

    if quiz_history:

        for row in quiz_history:

            st.write(
                f"📚 **{row[0]} — {row[1]}** | "
                f"Score: {row[3]}/{row[4]} | "
                f"{row[5]:.0f}%"
            )

    else:

        st.info(
            "No quiz history yet."
        )


    # -----------------------------------------
    # WEAK TOPICS
    # -----------------------------------------

    st.markdown("---")

    st.markdown(
        "## 🧠 Current Weak Topics"
    )

    weak_topics = get_weak_topics(
        st.session_state.student_name
    )

    if weak_topics:

        for item in weak_topics:

            st.write(
                f"🔴 {item['topic']} — "
                f"{item['percentage']:.0f}%"
            )

    else:

        st.success(
            "No weak topics detected."
        )
