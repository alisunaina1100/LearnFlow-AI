import os
import json
import sqlite3
import hashlib
import secrets
from datetime import datetime
from urllib.parse import urlparse

import streamlit as st
from groq import Groq
from tavily import TavilyClient


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

    /* ---------- HERO ---------- */

    .hero {
        padding: 2.2rem;
        border-radius: 22px;
        margin-bottom: 1.5rem;
        background:
            linear-gradient(
                135deg,
                rgba(79,70,229,0.18),
                rgba(14,165,233,0.12),
                rgba(16,185,129,0.10)
            );
        border: 1px solid rgba(99,102,241,0.28);
        box-shadow: 0 10px 35px rgba(0,0,0,0.08);
    }

    .hero h1 {
        margin-bottom: 0.3rem;
        font-size: 2.6rem;
    }

    .hero p {
        margin-top: 0.4rem;
    }

    /* ---------- METRIC CARDS ---------- */

    .metric-card {
        padding: 1.25rem;
        border-radius: 18px;
        border: 1px solid rgba(99,102,241,0.22);
        text-align: center;
        background: linear-gradient(
            145deg,
            rgba(99,102,241,0.10),
            rgba(59,130,246,0.05)
        );
        box-shadow: 0 6px 20px rgba(0,0,0,0.06);
    }

    .metric-number {
        font-size: 2rem;
        font-weight: 700;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.75;
    }

    /* ---------- WEAK TOPIC ---------- */

    .weak-topic {
        padding: 1rem;
        border-radius: 14px;
        margin-bottom: 0.7rem;
        border-left: 5px solid #ef4444;
        background: rgba(239,68,68,0.08);
    }

    /* ---------- SOURCE CARD ---------- */

    .source-card {
        padding: 0.9rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(59,130,246,0.22);
        background: rgba(59,130,246,0.06);
    }

    .source-number {
        font-weight: 700;
        color: #3b82f6;
    }

    /* ---------- INFO CARD ---------- */

    .info-card {
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid rgba(16,185,129,0.25);
        background: rgba(16,185,129,0.07);
        margin-bottom: 1rem;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        min-height: 2.8rem;
    }

    /* ---------- HEADINGS ---------- */

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.15);
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
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is not configured.")
    st.stop()


TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    try:
        TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    except Exception:
        TAVILY_API_KEY = None


if not TAVILY_API_KEY:
    st.warning(
        "⚠️ TAVILY_API_KEY is not configured. "
        "AI Tutor web search will not be available."
    )


client = Groq(api_key=GROQ_API_KEY)

tavily_client = (
    TavilyClient(api_key=TAVILY_API_KEY)
    if TAVILY_API_KEY
    else None
)

MODEL = "openai/gpt-oss-120b"


# =========================================================
# 4. DATABASE / MEMORY SYSTEM
# =========================================================

DB_NAME = "learnflow_memory.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def ensure_column(cursor, table_name, column_name, column_type):

    cursor.execute(
        f"PRAGMA table_info({table_name})"
    )

    columns = [
        row[1]
        for row in cursor.fetchall()
    ]

    if column_name not in columns:

        cursor.execute(
            f"""
            ALTER TABLE {table_name}
            ADD COLUMN {column_name} {column_type}
            """
        )


def init_database():

    conn = get_connection()
    cursor = conn.cursor()

    # -----------------------------------------------------
    # USERS
    # -----------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT
        )
        """
    )

    # -----------------------------------------------------
    # CHAT MEMORY
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # QUIZ HISTORY
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # STUDY PLANS
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # MIGRATION FOR OLD DATABASE
    # -----------------------------------------------------

    # Existing databases will receive user_id columns.
    # Old records remain NULL and are NOT shown to new users.

    ensure_column(
        cursor,
        "chat_history",
        "user_id",
        "INTEGER"
    )

    ensure_column(
        cursor,
        "quiz_history",
        "user_id",
        "INTEGER"
    )

    ensure_column(
        cursor,
        "study_plans",
        "user_id",
        "INTEGER"
    )

    conn.commit()
    conn.close()


init_database()


# =========================================================
# 5. PASSWORD / AUTHENTICATION FUNCTIONS
# =========================================================

def hash_password(password):

    salt = secrets.token_bytes(16)

    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        200_000
    )

    return (
        salt.hex()
        + ":"
        + pwd_hash.hex()
    )


def verify_password(password, stored_hash):

    try:

        salt_hex, hash_hex = stored_hash.split(":")

        salt = bytes.fromhex(
            salt_hex
        )

        pwd_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            200_000
        )

        return secrets.compare_digest(
            pwd_hash.hex(),
            hash_hex
        )

    except Exception:

        return False


def create_user(name, email, password):

    name = name.strip()
    email = email.strip().lower()

    if not name:

        return False, "❌ Please enter your name."

    if not email:

        return False, "❌ Please enter your email."

    if not password:

        return False, "❌ Please enter a password."

    if len(password) < 8:

        return False, (
            "❌ Password must be at least 8 characters."
        )

    password_hash = hash_password(
        password
    )

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO users
            (name, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                name,
                email,
                password_hash,
                datetime.now().isoformat()
            )
        )

        conn.commit()

        user_id = cursor.lastrowid

        return (
            True,
            {
                "id": user_id,
                "name": name,
                "email": email
            }
        )

    except sqlite3.IntegrityError:

        return (
            False,
            "❌ An account with this email already exists."
        )

    finally:

        conn.close()


def authenticate_user(email, password):

    email = email.strip().lower()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, name, email, password_hash
        FROM users
        WHERE email = ?
        """,
        (email,)
    )

    user = cursor.fetchone()

    conn.close()

    if not user:

        return None

    user_id = user[0]
    name = user[1]
    user_email = user[2]
    password_hash = user[3]

    if verify_password(
        password,
        password_hash
    ):

        return (
            user_id,
            name,
            user_email
        )

    return None


# =========================================================
# 6. SESSION STATE
# =========================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "student_name" not in st.session_state:
    st.session_state.student_name = ""

if "student_email" not in st.session_state:
    st.session_state.student_email = ""

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = []

if "quiz_saved" not in st.session_state:
    st.session_state.quiz_saved = False

if "quiz_subject" not in st.session_state:
    st.session_state.quiz_subject = ""

if "quiz_topic" not in st.session_state:
    st.session_state.quiz_topic = ""

if "quiz_difficulty" not in st.session_state:
    st.session_state.quiz_difficulty = ""


# =========================================================
# 7. LOGIN / CREATE ACCOUNT
# =========================================================

if not st.session_state.authenticated:

    st.markdown(
        """
        <div class="hero">

        <h1>🧠 LearnFlow AI</h1>

        <p style="font-size:1.2rem;">
        Your Personal AI Learning Companion
        </p>

        <p>
        🔐 Create an account to keep your learning memory private.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    login_tab, signup_tab = st.tabs(
        [
            "🔐 Login",
            "📝 Create Account"
        ]
    )

    # -----------------------------------------------------
    # LOGIN
    # -----------------------------------------------------

    with login_tab:

        st.subheader("Welcome Back 👋")

        login_email = st.text_input(
            "Email",
            key="login_email",
            placeholder="Enter your email"
        )

        login_password = st.text_input(
            "Password",
            type="password",
            key="login_password",
            placeholder="Enter your password"
        )

        if st.button(
            "🔐 Login",
            key="login_button",
            use_container_width=True
        ):

            if not login_email.strip():

                st.error(
                    "❌ Please enter your email."
                )

            elif not login_password:

                st.error(
                    "❌ Please enter your password."
                )

            else:

                user = authenticate_user(
                    login_email,
                    login_password
                )

                if user:

                    st.session_state.authenticated = True
                    st.session_state.user_id = user[0]
                    st.session_state.student_name = user[1]
                    st.session_state.student_email = user[2]

                    st.session_state.quiz_data = None
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = []
                    st.session_state.quiz_saved = False

                    st.session_state.quiz_subject = ""
                    st.session_state.quiz_topic = ""
                    st.session_state.quiz_difficulty = ""

                    st.success(
                        f"Welcome back, {user[1]}! 🎉"
                    )

                    st.rerun()

                else:

                    st.error(
                        "❌ Invalid email or password."
                    )

    # -----------------------------------------------------
    # CREATE ACCOUNT
    # -----------------------------------------------------

    with signup_tab:

        st.subheader("Create Your LearnFlow Account 🚀")

        signup_name = st.text_input(
            "Your Name",
            key="signup_name",
            placeholder="Enter your name"
        )

        signup_email = st.text_input(
            "Email",
            key="signup_email",
            placeholder="Enter your email"
        )

        signup_password = st.text_input(
            "Password",
            type="password",
            key="signup_password",
            placeholder="At least 8 characters"
        )

        signup_confirm = st.text_input(
            "Confirm Password",
            type="password",
            key="signup_confirm",
            placeholder="Enter password again"
        )

        if st.button(
            "🚀 Create Account",
            key="signup_button",
            use_container_width=True
        ):

            if signup_password != signup_confirm:

                st.error(
                    "❌ Passwords do not match."
                )

            else:

                success, result = create_user(
                    signup_name,
                    signup_email,
                    signup_password
                )

                if success:

                    st.session_state.authenticated = True
                    st.session_state.user_id = result["id"]
                    st.session_state.student_name = result["name"]
                    st.session_state.student_email = result["email"]

                    st.session_state.quiz_data = None
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = []
                    st.session_state.quiz_saved = False

                    st.session_state.quiz_subject = ""
                    st.session_state.quiz_topic = ""
                    st.session_state.quiz_difficulty = ""

                    st.success(
                        "🎉 Account created successfully!"
                    )

                    st.rerun()

                else:

                    st.error(result)

    st.stop()


# =========================================================
# 8. STUDENT SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 👤 Student Profile")

    st.markdown(
        f"**{st.session_state.student_name}**"
    )

    st.caption(
        st.session_state.student_email
    )

    if st.button(
        "🚪 Logout",
        key="logout_button",
        use_container_width=True
    ):

        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.student_name = ""
        st.session_state.student_email = ""

        st.session_state.quiz_data = None
        st.session_state.quiz_submitted = False
        st.session_state.quiz_answers = []
        st.session_state.quiz_saved = False

        st.session_state.quiz_subject = ""
        st.session_state.quiz_topic = ""
        st.session_state.quiz_difficulty = ""

        st.rerun()

    st.markdown("---")

    st.markdown("### 🧠 Your Learning Memory")

    st.caption(
        "LearnFlow saves your quiz results and learning conversations "
        "to build personalized recommendations."
    )

    st.markdown("---")

    if tavily_client:
        st.success("🔎 Web Search: Connected")
    else:
        st.warning("🔎 Web Search: Not Connected")


# =========================================================
# 9. MEMORY FUNCTIONS
# =========================================================

def save_chat(
    user_id,
    student,
    question,
    answer
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO chat_history
        (
            user_id,
            student,
            question,
            answer,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            user_id,
            student,
            question,
            answer,
            datetime.now().isoformat()
        )
    )

    conn.commit()
    conn.close()


def get_chat_history(
    user_id,
    limit=20
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT question, answer, created_at
        FROM chat_history
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (
            user_id,
            limit
        )
    )

    rows = cursor.fetchall()

    conn.close()

    return rows


def save_quiz_result(
    user_id,
    student,
    subject,
    topic,
    difficulty,
    score,
    total,
    wrong_topics
):

    percentage = (
        (score / total) * 100
        if total
        else 0
    )

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO quiz_history
        (
            user_id,
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
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
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


def get_quiz_history(user_id):

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
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (user_id,)
    )

    rows = cursor.fetchall()

    conn.close()

    return rows


def save_study_plan(
    user_id,
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
            user_id,
            student,
            goal,
            subjects,
            topics,
            plan,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
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
# 10. WEB SEARCH / SOURCE VERIFICATION
# =========================================================

TRUSTED_DOMAIN_KEYWORDS = [

    # Official / academic / documentation
    ".gov",
    ".edu",
    ".ac.uk",

    "ibm.com",
    "oracle.com",
    "microsoft.com",
    "learn.microsoft.com",
    "postgresql.org",
    "python.org",
    "docs.python.org",
    "developer.mozilla.org",
    "w3.org",

    # Reputable educational resources
    "geeksforgeeks.org",
    "w3schools.com",
    "tutorialspoint.com",
    "javatpoint.com",
    "khanacademy.org",
    "britannica.com",

    # Universities
    "mit.edu",
    "stanford.edu",
    "harvard.edu",
    "ox.ac.uk",
    "cam.ac.uk"
]


def get_domain(url):

    try:

        domain = urlparse(
            url
        ).netloc.lower()

        return domain.replace(
            "www.",
            ""
        )

    except Exception:

        return ""


def source_quality_score(result):

    url = result.get(
        "url",
        ""
    )

    domain = get_domain(
        url
    )

    relevance = float(
        result.get(
            "score",
            0
        )
        or 0
    )

    quality_bonus = 0.0

    # Government
    if (
        domain.endswith(".gov")
        or ".gov." in domain
    ):

        quality_bonus += 0.45

    # Universities
    elif (
        domain.endswith(".edu")
        or ".edu." in domain
    ):

        quality_bonus += 0.45

    # UK academic
    elif domain.endswith(".ac.uk"):

        quality_bonus += 0.45

    # Official technical sources
    elif any(
        trusted in domain
        for trusted in [
            "ibm.com",
            "oracle.com",
            "microsoft.com",
            "learn.microsoft.com",
            "postgresql.org",
            "python.org",
            "docs.python.org",
            "developer.mozilla.org",
            "w3.org"
        ]
    ):

        quality_bonus += 0.35

    # Reputable educational sources
    elif any(
        trusted in domain
        for trusted in [
            "geeksforgeeks.org",
            "w3schools.com",
            "tutorialspoint.com",
            "javatpoint.com",
            "khanacademy.org",
            "britannica.com"
        ]
    ):

        quality_bonus += 0.25

    # Named universities
    elif any(
        trusted in domain
        for trusted in [
            "mit.edu",
            "stanford.edu",
            "harvard.edu",
            "ox.ac.uk",
            "cam.ac.uk"
        ]
    ):

        quality_bonus += 0.45

    return relevance + quality_bonus


def search_reliable_sources(question):

    if not tavily_client:

        return [], (
            "Web search is not configured. "
            "Please configure TAVILY_API_KEY."
        )

    try:

        response = tavily_client.search(
            query=question,
            search_depth="advanced",
            topic="general",
            max_results=10,
            include_answer=False
        )

        results = response.get(
            "results",
            []
        )

        processed_results = []

        for result in results:

            title = result.get(
                "title",
                ""
            ).strip()

            url = result.get(
                "url",
                ""
            ).strip()

            content = result.get(
                "content",
                ""
            ).strip()

            if (
                not title
                or not url
                or not content
            ):

                continue

            domain = get_domain(
                url
            )

            quality = source_quality_score(
                result
            )

            is_trusted = any(
                trusted in domain
                for trusted in TRUSTED_DOMAIN_KEYWORDS
            )

            processed_results.append(
                {
                    "title": title,
                    "url": url,
                    "content": content,
                    "score": quality,
                    "trusted": is_trusted
                }
            )

        reliable_results = [
            result
            for result in processed_results
            if result["trusted"]
            and result["score"] >= 0.35
        ]

        reliable_results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        if reliable_results:

            return (
                reliable_results[:5],
                ""
            )

        # -------------------------------------------------
        # SECOND SEARCH WITH IMPROVED QUERY
        # -------------------------------------------------

        improved_query = (
            f"{question} "
            "definition explanation educational"
        )

        response = tavily_client.search(
            query=improved_query,
            search_depth="advanced",
            topic="general",
            max_results=10,
            include_answer=False
        )

        results = response.get(
            "results",
            []
        )

        processed_results = []

        for result in results:

            title = result.get(
                "title",
                ""
            ).strip()

            url = result.get(
                "url",
                ""
            ).strip()

            content = result.get(
                "content",
                ""
            ).strip()

            if (
                not title
                or not url
                or not content
            ):

                continue

            domain = get_domain(
                url
            )

            quality = source_quality_score(
                result
            )

            is_trusted = any(
                trusted in domain
                for trusted in TRUSTED_DOMAIN_KEYWORDS
            )

            if (
                is_trusted
                and quality >= 0.30
            ):

                processed_results.append(
                    {
                        "title": title,
                        "url": url,
                        "content": content,
                        "score": quality,
                        "trusted": True
                    }
                )

        processed_results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        if processed_results:

            return (
                processed_results[:5],
                ""
            )

        return [], (
            "No sufficiently reliable source "
            "was found."
        )

    except Exception as e:

        return [], (
            f"Web search error: {str(e)}"
        )


# =========================================================
# 11. SOURCE-GROUNDED LEARNFLOW Q&A AGENT
# =========================================================

def ask_learnflow(question):

    if (
        not question
        or not question.strip()
    ):

        return (
            "Please enter a question.",
            []
        )

    sources, search_error = search_reliable_sources(
        question
    )

    # -----------------------------------------------------
    # NO RELIABLE SOURCE
    # -----------------------------------------------------

    if not sources:

        return (
            "⚠️ I couldn't verify this from a reliable source.",
            []
        )

    # -----------------------------------------------------
    # BUILD SOURCE CONTEXT
    # -----------------------------------------------------

    source_context_parts = []

    for i, source in enumerate(
        sources,
        start=1
    ):

        source_context_parts.append(
            f"""
SOURCE {i}

Title:
{source["title"]}

URL:
{source["url"]}

Content:
{source["content"]}
"""
        )

    source_context = "\n".join(
        source_context_parts
    )

    # -----------------------------------------------------
    # STRICT SOURCE-GROUNDED SYSTEM PROMPT
    # -----------------------------------------------------

    system_prompt = """
You are LearnFlow AI, a reliable personal AI learning companion.

Your job is to explain educational information clearly,
accurately, and in student-friendly language.

The user question has been searched on the web.

SOURCE RELIABILITY RULES:

1. Use the retrieved sources provided in the context.
2. Give priority to official documentation, government sources,
   universities, academic institutions, reputable textbooks,
   and reputable educational resources.
3. Do NOT invent sources.
4. Do NOT invent citations.
5. Do NOT invent URLs.
6. Do NOT invent books, papers, authors, statistics, or references.
7. Do NOT claim that you checked a source that is not provided.
8. Do NOT use unsupported information as if it came from the sources.
9. If the retrieved sources do not adequately support the answer,
   say exactly:

   "I couldn't verify this from a reliable source."

10. Do not fill missing information with guesses.
11. Do not present uncertain information as a verified fact.

ANSWER RULES:

12. Answer the student's actual question.
13. Use simple language.
14. Break difficult concepts into small steps.
15. Use examples when useful.
16. Keep the explanation educational rather than overly complicated.
17. If the sources disagree, explain the disagreement instead
    of silently choosing one.
18. For current or changing information, rely on the retrieved
    sources rather than your internal knowledge.

CITATION RULE:

19. Every important factual claim based on retrieved web information
    should have an inline citation such as [1], [2], or [1][2].
20. The citation numbers must correspond exactly to the SOURCE
    numbers provided in the context.
21. Never create a citation number for a source that does not exist.

IMPORTANT:

The retrieved source content is the evidence for your answer.
Do not pretend to have searched anything beyond the provided sources.
"""

    user_prompt = f"""
STUDENT QUESTION:

{question}

RETRIEVED SOURCES:

{source_context}

Now answer the student's question using the retrieved
sources as the primary evidence.

The answer should be directly supported by the retrieved
sources. You may explain the information in simpler words,
but do not introduce unsupported facts.

If the retrieved sources clearly explain the concept,
answer the question normally with inline citations.

Only respond with:

"I couldn't verify this from a reliable source."

when the retrieved sources genuinely do not contain enough
information to answer the student's question.

Remember:
- Use inline citations like [1] and [2].
- Do not invent citations.
- Do not invent URLs.
- Do not claim information came from a source unless that
  source actually supports it.
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
                    "content": user_prompt
                }
            ],
            temperature=0.1
        )

        answer = response.choices[0].message.content.strip()

        if not answer:

            return (
                "⚠️ I couldn't verify this from a reliable source.",
                []
            )

        return (
            answer,
            sources
        )

    except Exception as e:

        return (
            f"❌ Error: {str(e)}",
            []
        )


# =========================================================
# 12. QUIZ GENERATOR
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
9. Prefer standard textbook concepts and established educational knowledge.
10. Do not invent facts, fake citations, fake sources, or fake references.
11. Make the difficulty appropriate for the selected Student Level and Difficulty.
12. If Student Level is Beginner and Difficulty is Easy, avoid unnecessarily
    advanced or tricky questions.

13. IMPORTANT CONCEPT TRACKING RULE:

For every question, identify the MAIN educational concept or subtopic
being tested.

The "concept" must be a short, meaningful topic name, NOT a question number.

GOOD examples:
"Arrays"
"Loops"
"Functions"
"Pointers"
"Variables"
"Integration"
"Derivatives"
"Normalization"

BAD examples:
"Question 1"
"Question 2"
"Q1"
"Q2"

If the selected topic is "Arrays", concepts may be things such as:
"Array Indexing"
"Array Traversal"
"Array Declaration"

The concept must accurately describe the knowledge being tested
by that specific question.

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
            "explanation": "clear explanation",
            "concept": "actual concept or subtopic"
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
                        "Use established educational knowledge. "
                        "Every question must include an accurate concept field. "
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

        quiz_data = json.loads(
            raw_response
        )

        questions = quiz_data.get(
            "questions",
            []
        )

        if len(questions) != number_of_questions:

            return None, (
                "❌ Incorrect number of questions generated."
            )

        for q in questions:

            required_fields = [
                "question",
                "options",
                "answer",
                "explanation",
                "concept"
            ]

            if not all(
                field in q
                for field in required_fields
            ):

                return None, (
                    "❌ Invalid question structure."
                )

            if len(q["options"]) != 4:

                return None, (
                    "❌ Every question needs 4 options."
                )

            if q["answer"] not in [
                0,
                1,
                2,
                3
            ]:

                return None, (
                    "❌ Invalid answer index."
                )

            if (
                not isinstance(
                    q["concept"],
                    str
                )
                or not q["concept"].strip()
            ):

                return None, (
                    "❌ Invalid concept field."
                )

        return (
            quiz_data,
            ""
        )

    except Exception as e:

        return None, (
            f"❌ Error: {str(e)}"
        )


# =========================================================
# 13. QUIZ EVALUATION
# =========================================================

def evaluate_quiz(
    quiz_data,
    student_answers
):

    questions = quiz_data["questions"]

    score = 0
    total = len(questions)

    feedback = []
    wrong_topics = []

    for i, question in enumerate(
        questions
    ):

        student_answer = student_answers[i]

        correct_answer = question["answer"]

        concept = question.get(
            "concept",
            "General Topic"
        ).strip()

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
                concept
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
                concept
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

    wrong_topics = list(
        dict.fromkeys(
            wrong_topics
        )
    )

    percentage = (
        (score / total) * 100
        if total
        else 0
    )

    return (
        score,
        total,
        percentage,
        wrong_topics,
        feedback
    )


# =========================================================
# 14. WEAK TOPIC ANALYSIS
# =========================================================

def get_weak_topics(user_id):

    history = get_quiz_history(
        user_id
    )

    topic_stats = {}

    for row in history:

        subject = row[0]
        selected_topic = row[1]
        score = row[3]
        total = row[4]
        saved_wrong_topics = row[6]

        try:

            wrong_concepts = json.loads(
                saved_wrong_topics
            )

        except (
            json.JSONDecodeError,
            TypeError
        ):

            wrong_concepts = []

        wrong_concepts = [
            concept
            for concept in wrong_concepts
            if not concept.lower().startswith(
                "question "
            )
        ]

        if wrong_concepts:

            for concept in wrong_concepts:

                key = (
                    f"{subject} — {concept}"
                )

                if key not in topic_stats:

                    topic_stats[key] = {
                        "score": 0,
                        "total": 0,
                        "attempts": 0
                    }

                topic_stats[key]["score"] += 0
                topic_stats[key]["total"] += 1
                topic_stats[key]["attempts"] += 1

        selected_key = (
            f"{subject} — {selected_topic}"
        )

        if selected_key not in topic_stats:

            topic_stats[selected_key] = {
                "score": 0,
                "total": 0,
                "attempts": 0
            }

        topic_stats[selected_key]["score"] += score
        topic_stats[selected_key]["total"] += total
        topic_stats[selected_key]["attempts"] += 1

    weak_topics = []

    for topic, data in topic_stats.items():

        percentage = (
            data["score"]
            / data["total"]
            * 100
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
# 15. PERSONALIZED RECOMMENDATIONS
# =========================================================

def generate_recommendations(user_id):

    weak_topics = get_weak_topics(
        user_id
    )

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
# 16. STUDY PLANNER
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
9. Do not invent academic sources or fake references.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful and accurate AI study planner. "
                        "Do not invent sources, citations, or references."
                    )
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
# 17. HEADER
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
# 18. DASHBOARD
# =========================================================

quiz_history = get_quiz_history(
    st.session_state.user_id
)

weak_topics = get_weak_topics(
    st.session_state.user_id
)

total_quizzes = len(
    quiz_history
)

if total_quizzes:

    total_score = sum(
        row[3]
        for row in quiz_history
    )

    total_questions = sum(
        row[4]
        for row in quiz_history
    )

    overall_percentage = (
        total_score
        / total_questions
        * 100
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
# 19. TABS
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

    st.markdown(
        """
        <div class="info-card">
        🔎 <b>Source-Backed Learning</b><br>
        LearnFlow searches the web for relevant reliable sources
        before generating an educational explanation.
        </div>
        """,
        unsafe_allow_html=True
    )

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

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        elif not tavily_client:

            st.error(
                "❌ Web Search is not configured. "
                "Please add TAVILY_API_KEY."
            )

        else:

            with st.spinner(
                "🔎 Searching reliable sources and preparing your answer..."
            ):

                answer, sources = ask_learnflow(
                    question
                )

            st.markdown(
                "### 🧑‍🏫 AI Answer"
            )

            st.markdown(
                answer
            )

            # ---------------------------------------------
            # SOURCES
            # ---------------------------------------------

            if sources:

                st.markdown("---")

                st.markdown(
                    "### 📚 Sources Used"
                )

                for i, source in enumerate(
                    sources,
                    start=1
                ):

                    st.markdown(
                        f"""
                        <div class="source-card">

                        <span class="source-number">
                        [{i}]
                        </span>

                        <b>{source["title"]}</b>

                        <br><br>

                        🔗
                        <a href="{source["url"]}"
                           target="_blank">
                           {source["url"]}
                        </a>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # ---------------------------------------------
            # SAVE CHAT
            # ---------------------------------------------

            if (
                not answer.startswith("❌")
                and not answer.startswith("⚠️")
            ):

                save_chat(
                    st.session_state.user_id,
                    st.session_state.student_name,
                    question,
                    answer
                )

                st.success(
                    "💾 This conversation has been saved "
                    "to your learning memory."
                )

            elif answer.startswith("⚠️"):

                st.warning(
                    "This answer was not saved because "
                    "reliable source verification was not available."
                )

            else:

                st.error(
                    "The answer could not be generated, "
                    "so it was not saved."
                )


# =========================================================
# TAB 2 — QUIZ
# =========================================================

with tab2:

    st.header("📝 AI Quiz Agent")

    st.markdown(
        "### 🎓 Education Information"
    )

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

    st.markdown(
        "### 📚 Quiz Information"
    )

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
            index=0
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

            # Store quiz metadata so it remains tied
            # to the generated quiz.
            st.session_state.quiz_subject = subject
            st.session_state.quiz_topic = topic
            st.session_state.quiz_difficulty = difficulty

            st.success(
                "✅ Quiz generated! Attempt all questions."
            )


    # ---------------------------------------------
    # DISPLAY QUIZ
    # ---------------------------------------------

    if st.session_state.quiz_data is not None:

        quiz_data = st.session_state.quiz_data

        st.markdown("---")

        st.markdown(
            "## 📝 Attempt Your Quiz"
        )

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
                [
                    "A",
                    "B",
                    "C",
                    "D"
                ],
                index=None,
                key=f"quiz_answer_{i}"
            )

            if selected is not None:

                st.session_state.quiz_answers[i] = (
                    ord(selected)
                    - ord("A")
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

            if not st.session_state.get(
                "quiz_saved",
                False
            ):

                save_quiz_result(
                    st.session_state.user_id,
                    st.session_state.student_name,
                    st.session_state.quiz_subject,
                    st.session_state.quiz_topic,
                    st.session_state.quiz_difficulty,
                    score,
                    total,
                    wrong_topics
                )

                st.session_state.quiz_saved = True


            st.markdown("---")

            st.markdown(
                "## 📊 Quiz Result"
            )

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


            st.markdown("---")

            st.markdown(
                "## 🔄 Personalized Recommendations"
            )

            recommendations = generate_recommendations(
                st.session_state.user_id
            )

            for recommendation in recommendations:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        recommendation
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

        st.markdown("---")

        st.markdown(
            "## 📅 Your Study Plan"
        )

        st.markdown(
            plan
        )

        if not plan.startswith("❌"):

            save_study_plan(
                st.session_state.user_id,
                st.session_state.student_name,
                planner_goal,
                planner_subjects,
                planner_topics,
                plan
            )

            st.success(
                "💾 This study plan has been saved "
                "to your learning memory."
            )


# =========================================================
# TAB 4 — PROGRESS
# =========================================================

with tab4:

    st.header("📈 Student Progress")

    history = get_quiz_history(
        st.session_state.user_id
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
            st.session_state.user_id
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
            st.session_state.user_id
        )

        for recommendation in recommendations:

            with st.container(
                border=True
            ):

                st.markdown(
                    recommendation
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
        st.session_state.user_id
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
        st.session_state.user_id
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
