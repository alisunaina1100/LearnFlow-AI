import os
import re
import json
import sqlite3
import hashlib
import secrets
import logging
from datetime import datetime, timezone

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

st.markdown("""
<style>

.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #eef4ff, #f8fbff);
    border: 1px solid #dbe7ff;
    margin-bottom: 25px;
}

.hero h1 {
    margin-bottom: 5px;
}

.hero p {
    color: #555;
    font-size: 17px;
}

.metric-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
    background: white;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.metric-card h2 {
    margin: 0;
}

.metric-card p {
    margin: 5px 0 0 0;
    color: #666;
}

.weak-topic {
    padding: 14px;
    border-radius: 12px;
    background: #fff7ed;
    border: 1px solid #fed7aa;
    margin-bottom: 10px;
}

.source-card {
    padding: 15px;
    border-radius: 12px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    margin-bottom: 10px;
}

.source-number {
    display: inline-block;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    text-align: center;
    padding-top: 4px;
    font-weight: bold;
    margin-right: 8px;
    background: #e2e8f0;
}

.info-card {
    padding: 18px;
    border-radius: 14px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    margin: 10px 0;
}

button {
    border-radius: 10px !important;
}

h1, h2, h3 {
    font-weight: 700;
}

section[data-testid="stSidebar"] {
    border-right: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LearnFlowAI")


# =========================================================
# 4. API CONFIGURATION
# =========================================================

def get_secret_or_env(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name, default)


GROQ_API_KEY = get_secret_or_env("GROQ_API_KEY")
TAVILY_API_KEY = get_secret_or_env("TAVILY_API_KEY")

MODEL = "openai/gpt-oss-120b"

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is missing. Please add it to Streamlit Secrets."
    )
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

tavily_client = None

if TAVILY_API_KEY:
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    except Exception as e:
        logger.error("Tavily initialization failed: %s", e)
        tavily_client = None


# =========================================================
# 5. SECURITY LIMITS
# =========================================================

MAX_NAME_LENGTH = 100
MAX_EMAIL_LENGTH = 254
MAX_TEXT_LENGTH = 10000
MAX_QUESTION_LENGTH = 5000
MAX_SOURCE_CONTENT_LENGTH = 8000

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128

CURRENT_PASSWORD_ITERATIONS = 300_000
LEGACY_PASSWORD_ITERATIONS = 200_000


# =========================================================
# 6. GENERAL HELPERS
# =========================================================

def clean_text(value, max_length=MAX_TEXT_LENGTH):
    if value is None:
        return ""

    value = str(value)
    value = value.replace("\x00", "")
    value = re.sub(r"\s+", " ", value).strip()

    return value[:max_length]


def is_valid_email(email):
    email = clean_text(email, MAX_EMAIL_LENGTH)

    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

    return bool(re.fullmatch(pattern, email))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# =========================================================
# 7. DATABASE
# =========================================================

DB_PATH = "learnflow_memory.db"


def get_connection():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=10,
        check_same_thread=False
    )

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")

    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass

    return conn


def secure_database_file():
    try:
        if os.path.exists(DB_PATH):
            os.chmod(DB_PATH, 0o600)
    except Exception:
        pass


def ensure_column(conn, table, column, definition):
    allowed_tables = {
        "chat_history",
        "quiz_history",
        "study_plans"
    }

    allowed_columns = {
        "user_id",
        "concept_stats"
    }

    if table not in allowed_tables or column not in allowed_columns:
        return

    existing = conn.execute(
        f"PRAGMA table_info({table})"
    ).fetchall()

    names = {row["name"] for row in existing}

    if column not in names:
        conn.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
        )


def init_database():
    conn = get_connection()

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                password_iterations INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                student TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS quiz_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                score INTEGER NOT NULL,
                total INTEGER NOT NULL,
                percentage REAL NOT NULL,
                wrong_topics TEXT,
                concept_stats TEXT,
                feedback TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS study_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                plan TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            )
        """)

        ensure_column(
            conn,
            "chat_history",
            "user_id",
            "INTEGER"
        )

        ensure_column(
            conn,
            "quiz_history",
            "user_id",
            "INTEGER"
        )

        ensure_column(
            conn,
            "quiz_history",
            "concept_stats",
            "TEXT"
        )

        ensure_column(
            conn,
            "study_plans",
            "user_id",
            "INTEGER"
        )

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_user
            ON chat_history(user_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_quiz_user
            ON quiz_history(user_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_plan_user
            ON study_plans(user_id)
        """)

        conn.commit()

    finally:
        conn.close()

    secure_database_file()


init_database()


# =========================================================
# 8. PASSWORD SECURITY
# =========================================================

def hash_password(password):
    salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        CURRENT_PASSWORD_ITERATIONS
    )

    return (
        password_hash.hex(),
        salt.hex(),
        CURRENT_PASSWORD_ITERATIONS
    )


def verify_password(password, stored_hash, stored_salt, iterations):
    try:
        salt = bytes.fromhex(stored_salt)

        calculated = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            int(iterations)
        ).hex()

        return secrets.compare_digest(
            calculated,
            stored_hash
        )

    except Exception:
        return False


# =========================================================
# 9. USER AUTHENTICATION
# =========================================================

def create_user(name, email, password):
    name = clean_text(name, MAX_NAME_LENGTH)
    email = clean_text(email, MAX_EMAIL_LENGTH).lower()

    if not name:
        return False, "Please enter your name."

    if not is_valid_email(email):
        return False, "Please enter a valid email address."

    if len(password) < MIN_PASSWORD_LENGTH:
        return False, (
            f"Password must be at least "
            f"{MIN_PASSWORD_LENGTH} characters."
        )

    if len(password) > MAX_PASSWORD_LENGTH:
        return False, "Password is too long."

    password_hash, salt, iterations = hash_password(password)

    conn = get_connection()

    try:
        conn.execute("""
            INSERT INTO users
            (
                name,
                email,
                password_hash,
                password_salt,
                password_iterations,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            name,
            email,
            password_hash,
            salt,
            iterations,
            utc_now()
        ))

        conn.commit()

        return True, "Account created successfully."

    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."

    except Exception as e:
        logger.exception("Signup failed")
        return False, "Unable to create account right now."

    finally:
        conn.close()


def authenticate_user(email, password):
    email = clean_text(email, MAX_EMAIL_LENGTH).lower()

    conn = get_connection()

    try:
        row = conn.execute("""
            SELECT *
            FROM users
            WHERE email = ?
        """, (email,)).fetchone()

        if not row:
            return None

        if not verify_password(
            password,
            row["password_hash"],
            row["password_salt"],
            row["password_iterations"]
        ):
            return None

        return {
            "id": row["id"],
            "name": row["name"],
            "email": row["email"]
        }

    except Exception:
        logger.exception("Authentication failed")
        return None

    finally:
        conn.close()


# =========================================================
# 10. SESSION STATE
# =========================================================

defaults = {
    "authenticated": False,
    "user_id": None,
    "student_name": "",
    "student_email": "",

    "quiz_data": None,
    "quiz_submitted": False,
    "quiz_answers": [],
    "quiz_saved": False,
    "quiz_subject": "",
    "quiz_topic": "",
    "quiz_difficulty": "",

    # FIX:
    # Every generated quiz gets a new widget generation ID.
    "quiz_generation_id": 0,

    "login_attempts": 0,
    "login_blocked_until": 0.0,

    "tutor_messages": []
}


for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 11. LOGIN RATE LIMITING
# =========================================================

MAX_LOGIN_ATTEMPTS = 5
LOGIN_COOLDOWN_SECONDS = 60


def reset_login_attempts():
    st.session_state.login_attempts = 0
    st.session_state.login_blocked_until = 0.0


# =========================================================
# 12. LOGIN / SIGNUP
# =========================================================

if not st.session_state.authenticated:

    st.markdown("""
    <div class="hero">
        <h1>🧠 LearnFlow AI</h1>
        <p>Your Personal AI Learning Companion</p>
    </div>
    """, unsafe_allow_html=True)

    login_tab, signup_tab = st.tabs(
        ["🔐 Login", "📝 Create Account"]
    )

    with login_tab:

        st.subheader("Welcome Back")

        email = st.text_input(
            "Email",
            key="login_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.session_state.login_blocked_until > 0:
            remaining = (
                st.session_state.login_blocked_until
                - datetime.now().timestamp()
            )

            if remaining > 0:
                st.warning(
                    f"Too many attempts. "
                    f"Please wait {int(remaining)} seconds."
                )
            else:
                reset_login_attempts()

        if st.button(
            "Login",
            type="primary",
            use_container_width=True
        ):

            if st.session_state.login_blocked_until > 0:
                remaining = (
                    st.session_state.login_blocked_until
                    - datetime.now().timestamp()
                )

                if remaining > 0:
                    st.error(
                        f"Please wait {int(remaining)} seconds."
                    )
                    st.stop()

            if not email or not password:
                st.error("Please enter email and password.")

            else:

                user = authenticate_user(email, password)

                if user:

                    st.session_state.authenticated = True
                    st.session_state.user_id = user["id"]
                    st.session_state.student_name = user["name"]
                    st.session_state.student_email = user["email"]

                    st.session_state.quiz_data = None
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = []
                    st.session_state.quiz_saved = False
                    st.session_state.quiz_generation_id = 0
                    st.session_state.tutor_messages = []

                    reset_login_attempts()

                    st.success("Login successful!")
                    st.rerun()

                else:

                    st.session_state.login_attempts += 1

                    if (
                        st.session_state.login_attempts
                        >= MAX_LOGIN_ATTEMPTS
                    ):
                        st.session_state.login_blocked_until = (
                            datetime.now().timestamp()
                            + LOGIN_COOLDOWN_SECONDS
                        )

                    st.error(
                        "Invalid email or password."
                    )

    with signup_tab:

        st.subheader("Create Your LearnFlow Account")

        name = st.text_input(
            "Full Name",
            key="signup_name"
        )

        email = st.text_input(
            "Email Address",
            key="signup_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="signup_password"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            key="signup_confirm_password"
        )

        if st.button(
            "Create Account",
            type="primary",
            use_container_width=True
        ):

            if password != confirm_password:
                st.error("Passwords do not match.")

            else:

                success, message = create_user(
                    name,
                    email,
                    password
                )

                if success:

                    user = authenticate_user(
                        email,
                        password
                    )

                    if user:

                        st.session_state.authenticated = True
                        st.session_state.user_id = user["id"]
                        st.session_state.student_name = user["name"]
                        st.session_state.student_email = user["email"]

                        st.session_state.quiz_data = None
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_answers = []
                        st.session_state.quiz_saved = False
                        st.session_state.quiz_generation_id = 0
                        st.session_state.tutor_messages = []

                        st.success(message)
                        st.rerun()

                else:
                    st.error(message)

    st.stop()


# =========================================================
# 13. MEMORY FUNCTIONS
# =========================================================

def save_chat(user_id, student, question, answer):

    conn = get_connection()

    try:
        conn.execute("""
            INSERT INTO chat_history
            (
                user_id,
                student,
                question,
                answer,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            user_id,
            student,
            clean_text(question, MAX_QUESTION_LENGTH),
            clean_text(answer, MAX_TEXT_LENGTH),
            utc_now()
        ))

        conn.commit()

    except Exception:
        logger.exception("Could not save chat.")

    finally:
        conn.close()


def get_chat_history(user_id, limit=20):

    conn = get_connection()

    try:
        rows = conn.execute("""
            SELECT *
            FROM chat_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (
            user_id,
            limit
        )).fetchall()

        return [dict(row) for row in rows]

    except Exception:
        return []

    finally:
        conn.close()


def save_quiz_result(
    user_id,
    subject,
    topic,
    difficulty,
    score,
    total,
    percentage,
    wrong_topics,
    concept_stats,
    feedback
):

    conn = get_connection()

    try:
        conn.execute("""
            INSERT INTO quiz_history
            (
                user_id,
                subject,
                topic,
                difficulty,
                score,
                total,
                percentage,
                wrong_topics,
                concept_stats,
                feedback,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            clean_text(subject, 200),
            clean_text(topic, 300),
            clean_text(difficulty, 50),
            score,
            total,
            percentage,
            json.dumps(wrong_topics, ensure_ascii=False),
            json.dumps(concept_stats, ensure_ascii=False),
            clean_text(feedback, 3000),
            utc_now()
        ))

        conn.commit()

    except Exception:
        logger.exception("Could not save quiz.")

    finally:
        conn.close()


def get_quiz_history(user_id):

    conn = get_connection()

    try:
        rows = conn.execute("""
            SELECT *
            FROM quiz_history
            WHERE user_id = ?
            ORDER BY id DESC
        """, (user_id,)).fetchall()

        return [dict(row) for row in rows]

    except Exception:
        return []

    finally:
        conn.close()


def save_study_plan(user_id, subject, topic, plan):

    conn = get_connection()

    try:
        conn.execute("""
            INSERT INTO study_plans
            (
                user_id,
                subject,
                topic,
                plan,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            user_id,
            clean_text(subject, 200),
            clean_text(topic, 300),
            clean_text(plan, MAX_TEXT_LENGTH),
            utc_now()
        ))

        conn.commit()

    except Exception:
        logger.exception("Could not save study plan.")

    finally:
        conn.close()


# =========================================================
# 14. TRUSTED WEB SOURCES
# =========================================================

TRUSTED_EXACT_DOMAINS = {
    "ibm.com",
    "oracle.com",
    "microsoft.com",
    "learn.microsoft.com",
    "postgresql.org",
    "python.org",
    "docs.python.org",
    "developer.mozilla.org",
    "w3.org",
    "geeksforgeeks.org",
    "w3schools.com",
    "tutorialspoint.com",
    "javatpoint.com",
    "khanacademy.org",
    "britannica.com",
    "mit.edu",
    "stanford.edu",
    "harvard.edu",
    "ox.ac.uk",
    "cam.ac.uk",
    "wolfram.com"
}


def get_domain(url):

    try:

        parsed = __import__("urllib.parse", fromlist=["urlparse"]).urlparse(url)

        if parsed.scheme.lower() != "https":
            return None

        if parsed.username or parsed.password:
            return None

        domain = parsed.hostname

        if not domain:
            return None

        return domain.lower().lstrip("www.")

    except Exception:
        return None


def domain_matches(domain, trusted):

    return (
        domain == trusted
        or domain.endswith("." + trusted)
    )


def is_trusted_domain(domain):

    if not domain:
        return False

    return any(
        domain_matches(domain, trusted)
        for trusted in TRUSTED_EXACT_DOMAINS
    )


def source_quality_score(result):

    url = result.get("url", "")
    domain = get_domain(url)

    score = 0

    if is_trusted_domain(domain):
        score += 100

    title = result.get("title", "").lower()

    educational_words = [
        "documentation",
        "guide",
        "tutorial",
        "reference",
        "university",
        "official",
        "learn"
    ]

    for word in educational_words:
        if word in title:
            score += 5

    return score


def process_search_results(results):

    processed = []

    for item in results or []:

        url = item.get("url", "")
        title = clean_text(item.get("title", ""), 300)
        content = clean_text(
            item.get("content", ""),
            MAX_SOURCE_CONTENT_LENGTH
        )

        domain = get_domain(url)

        if not domain or not url or not title:
            continue

        processed.append({
            "title": title,
            "url": url,
            "content": content,
            "domain": domain,
            "trusted": is_trusted_domain(domain),
            "score": source_quality_score(item)
        })

    return processed


def search_reliable_sources(question):

    if not tavily_client:
        return []

    question = clean_text(
        question,
        MAX_QUESTION_LENGTH
    )

    try:

        response = tavily_client.search(
            query=question,
            search_depth="advanced",
            topic="general",
            max_results=10,
            include_answer=False
        )

        results = process_search_results(
            response.get("results", [])
        )

        if len(results) < 3:

            response2 = tavily_client.search(
                query=(
                    question
                    + " official documentation university "
                    + "educational explanation"
                ),
                search_depth="advanced",
                topic="general",
                max_results=10,
                include_answer=False
            )

            results2 = process_search_results(
                response2.get("results", [])
            )

            results.extend(results2)

        unique = {}

        for result in results:
            unique[result["url"]] = result

        final_results = list(unique.values())

        final_results.sort(
            key=lambda x: (
                x["trusted"],
                x["score"]
            ),
            reverse=True
        )

        return final_results[:5]

    except Exception as e:

        logger.exception(
            "Web search failed: %s",
            e
        )

        return []


# =========================================================
# 15. TUTOR CONVERSATION CONTEXT
# =========================================================

def build_conversation_context(
    user_id,
    limit=8
):

    history = get_chat_history(
        user_id,
        limit
    )

    if not history:
        return ""

    history = list(reversed(history))

    lines = []

    for item in history:

        lines.append(
            "STUDENT: "
            + item["question"]
        )

        lines.append(
            "LEARNFLOW AI: "
            + item["answer"]
        )

    return "\n".join(lines)


# =========================================================
# 16. AI TUTOR
# =========================================================

def ask_learnflow(
    question,
    tutor_level="Beginner"
):

    question = clean_text(
        question,
        MAX_QUESTION_LENGTH
    )

    tutor_level = clean_text(
        tutor_level,
        50
    )

    if not question:
        return (
            "Please enter a question.",
            [],
            False
        )

    previous_context = build_conversation_context(
        st.session_state.user_id,
        limit=8
    )

    sources = search_reliable_sources(
        question
    )

    source_context = ""

    if sources:

        source_parts = []

        for i, source in enumerate(sources, 1):

            source_parts.append(
                f"""
SOURCE {i}
Title: {source["title"]}
Domain: {source["domain"]}
URL: {source["url"]}
Content:
{source["content"]}
"""
            )

        source_context = "\n".join(
            source_parts
        )

    else:

        source_context = (
            "No reliable web sources were available. "
            "Use established knowledge and do not invent "
            "citations or references."
        )

    system_prompt = f"""
You are LearnFlow AI, a helpful educational tutor.

Student level:
{tutor_level}

Previous conversation:
{previous_context if previous_context else "No previous conversation."}

Web research:
{source_context}

Rules:

1. Answer educational questions accurately.
2. Match the student's requested language.
3. You may answer in English, Urdu, Roman Urdu, or mixed language.
4. Keep explanations clear and student-friendly.
5. For Beginner level, explain concepts from basics.
6. For Intermediate, give moderate detail and examples.
7. For Advanced, give deeper technical explanation.
8. If the user asks a follow-up question, use the previous conversation.
9. Do not pretend that an unavailable source was checked.
10. If web sources are available, cite relevant claims using [1], [2], etc.
11. Do not create fake sources.
12. Do not invent URLs.
13. For coding questions, give correct code and explain it simply.
14. Use examples when helpful.
15. If the student is confused, simplify rather than making the answer longer.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        answer = response.choices[0].message.content

        if not answer:
            answer = (
                "I couldn't generate an answer right now."
            )

        return (
            answer.strip(),
            sources,
            bool(sources)
        )

    except Exception as e:

        logger.exception(
            "Tutor request failed: %s",
            e
        )

        return (
            "Sorry, I couldn't process your question right now.",
            [],
            False
        )


# =========================================================
# 17. QUIZ GENERATOR
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

    education_level = clean_text(
        education_level,
        100
    )

    class_degree = clean_text(
        class_degree,
        150
    )

    subject = clean_text(
        subject,
        200
    )

    topic = clean_text(
        topic,
        300
    )

    student_level = clean_text(
        student_level,
        50
    )

    difficulty = clean_text(
        difficulty,
        50
    )

    try:
        number_of_questions = int(
            number_of_questions
        )
    except Exception:
        return None, "Invalid number of questions."

    if not education_level:
        return None, "Please select education level."

    if not class_degree:
        return None, "Please enter class/degree."

    if not subject:
        return None, "Please enter subject."

    if not topic:
        return None, "Please enter topic."

    if number_of_questions < 1 or number_of_questions > 10:
        return None, "Questions must be between 1 and 10."

    difficulty_rules = {
        "Easy": """
- Basic recall or simple application.
- Mostly one-step reasoning.
- No tricky or confusing wording.
- Avoid advanced terminology.
- Suitable for genuine beginner practice.
""",

        "Medium": """
- Moderate conceptual understanding.
- May require two-step reasoning or application.
- Distractors should be plausible.
- More than simple memorization.
""",

        "Hard": """
- Challenging conceptual application.
- May require multi-step reasoning.
- Include edge cases or deeper understanding.
- Distractors should be close and plausible.
"""
    }

    selected_rules = difficulty_rules.get(
        difficulty,
        difficulty_rules["Easy"]
    )

    prompt = f"""
Create exactly {number_of_questions} multiple-choice questions.

Education Level: {education_level}
Class/Degree: {class_degree}
Subject: {subject}
Topic: {topic}
Student Level: {student_level}
Difficulty: {difficulty}

IMPORTANT DIFFICULTY RULES:

{selected_rules}

Never ignore the selected difficulty.

If Difficulty is Easy:
- Questions must genuinely be easy.
- Do not secretly create Medium or Hard questions.
- Do not use advanced concepts unless required by the topic.
- Keep reasoning simple.

If Student Level is Beginner AND Difficulty is Easy:
- Make questions especially beginner-friendly.
- Explain concepts using simple wording.

Each question must:

1. Be directly related to the selected subject and topic.
2. Have exactly 4 different options.
3. Have exactly one correct answer.
4. Use options A, B, C, D.
5. Have a clear explanation.
6. Have a meaningful concept/subtopic.
7. Avoid fake references.
8. Avoid ambiguous questions.
9. Avoid duplicate options.
10. Avoid questions where two answers could be correct.

Concept examples:
- Power Rule
- Constant Rule
- Array Indexing
- Loops
- Variables
- Functions
- Normalization
- SQL SELECT
- etc.

For mathematics:
- Verify calculations carefully.
- Keep units correct.
- Do not make arithmetic mistakes.
- Put the correct result in exactly one option.

Return ONLY valid JSON.

Use this exact structure:

{{
  "questions": [
    {{
      "question": "Question text",
      "options": [
        "Option A",
        "Option B",
        "Option C",
        "Option D"
      ],
      "answer": 0,
      "explanation": "Clear explanation",
      "concept": "Concept name"
    }}
  ]
}}

The answer index must be:
0 = A
1 = B
2 = C
3 = D
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict educational quiz "
                        "generator. Return valid JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        raw = response.choices[0].message.content

        if not raw:
            return None, "AI returned an empty quiz."

        raw = raw.strip()

        # Remove Markdown code fences if AI adds them
        raw = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw,
            flags=re.IGNORECASE
        )

        raw = re.sub(
            r"\s*```$",
            "",
            raw
        )

        data = json.loads(raw)

        if not isinstance(data, dict):
            return None, "Invalid quiz format."

        questions = data.get("questions")

        if not isinstance(questions, list):
            return None, "Quiz questions are missing."

        if len(questions) != number_of_questions:
            return None, (
                f"AI generated {len(questions)} questions "
                f"instead of {number_of_questions}."
            )

        validated_questions = []

        for index, q in enumerate(questions):

            if not isinstance(q, dict):
                return None, (
                    f"Question {index + 1} has invalid format."
                )

            question_text = clean_text(
                q.get("question", ""),
                MAX_QUESTION_LENGTH
            )

            options = q.get("options")

            answer = q.get("answer")

            explanation = clean_text(
                q.get("explanation", ""),
                3000
            )

            concept = clean_text(
                q.get("concept", ""),
                200
            )

            if not question_text:
                return None, (
                    f"Question {index + 1} is empty."
                )

            if not isinstance(options, list):
                return None, (
                    f"Question {index + 1} has invalid options."
                )

            if len(options) != 4:
                return None, (
                    f"Question {index + 1} must have exactly "
                    "4 options."
                )

            cleaned_options = [
                clean_text(option, 1000)
                for option in options
            ]

            if any(
                not option
                for option in cleaned_options
            ):
                return None, (
                    f"Question {index + 1} contains an empty option."
                )

            if len(
                set(
                    option.lower()
                    for option in cleaned_options
                )
            ) != 4:
                return None, (
                    f"Question {index + 1} contains duplicate options."
                )

            if not isinstance(answer, int):
                return None, (
                    f"Question {index + 1} has invalid answer index."
                )

            if answer not in [0, 1, 2, 3]:
                return None, (
                    f"Question {index + 1} has invalid answer."
                )

            if not explanation:
                return None, (
                    f"Question {index + 1} has no explanation."
                )

            if not concept:
                concept = topic

            validated_questions.append({
                "question": question_text,
                "options": cleaned_options,
                "answer": answer,
                "explanation": explanation,
                "concept": concept
            })

        return {
            "questions": validated_questions
        }, None

    except json.JSONDecodeError:

        logger.exception("Quiz JSON parsing failed")

        return None, (
            "The AI returned an invalid quiz format. "
            "Please generate the quiz again."
        )

    except Exception:

        logger.exception("Quiz generation failed")

        return None, (
            "Unable to generate the quiz right now. "
            "Please try again."
        )


# =========================================================
# 18. QUIZ EVALUATION
# =========================================================

def evaluate_quiz(
    quiz_data,
    student_answers
):

    questions = quiz_data.get(
        "questions",
        []
    )

    score = 0
    total = len(questions)

    wrong_topics = []
    concept_stats = {}

    for i, question in enumerate(questions):

        correct_answer = question.get(
            "answer"
        )

        concept = clean_text(
            question.get(
                "concept",
                "General"
            ),
            200
        )

        if not concept:
            concept = "General"

        if concept not in concept_stats:
            concept_stats[concept] = {
                "correct": 0,
                "total": 0
            }

        concept_stats[concept]["total"] += 1

        student_answer = (
            student_answers[i]
            if i < len(student_answers)
            else None
        )

        if student_answer == correct_answer:

            score += 1

            concept_stats[concept]["correct"] += 1

        else:

            wrong_topics.append(
                concept
            )

    percentage = (
        (score / total) * 100
        if total
        else 0
    )

    if percentage >= 90:
        feedback = (
            "Excellent performance! You have a strong "
            "understanding of this topic."
        )

    elif percentage >= 75:
        feedback = (
            "Good performance! You understand most of "
            "the concepts, but a little more practice "
            "will make you stronger."
        )

    elif percentage >= 60:
        feedback = (
            "Fair performance. Review the weak concepts "
            "and practice a few more questions."
        )

    else:
        feedback = (
            "You should review the topic from the basics "
            "and practice more questions before moving "
            "to harder concepts."
        )

    return (
        score,
        total,
        percentage,
        wrong_topics,
        concept_stats,
        feedback
    )


# =========================================================
# 19. WEAK TOPICS
# =========================================================

def get_weak_topics(history):

    aggregated = {}

    for item in history:

        raw_stats = item.get(
            "concept_stats"
        )

        if raw_stats:

            try:
                stats = json.loads(
                    raw_stats
                )

                if isinstance(stats, dict):

                    for concept, values in stats.items():

                        if concept not in aggregated:
                            aggregated[concept] = {
                                "correct": 0,
                                "total": 0
                            }

                        aggregated[concept]["correct"] += int(
                            values.get("correct", 0)
                        )

                        aggregated[concept]["total"] += int(
                            values.get("total", 0)
                        )

            except Exception:
                pass

    weak = []

    for concept, values in aggregated.items():

        total = values["total"]

        if total <= 0:
            continue

        percentage = (
            values["correct"]
            / total
            * 100
        )

        if percentage < 70:

            weak.append({
                "concept": concept,
                "percentage": percentage,
                "correct": values["correct"],
                "total": total
            })

    weak.sort(
        key=lambda x: x["percentage"]
    )

    return weak


def recommendation_for_score(
    percentage,
    weak_topics
):

    if not weak_topics:

        return (
            "🎉 No major weak concepts were detected. "
            "Keep practicing to maintain your performance."
        )

    topics = ", ".join(
        item["concept"]
        for item in weak_topics[:5]
    )

    if percentage >= 80:

        return (
            f"Your overall performance is good. "
            f"Focus on these concepts for improvement: "
            f"{topics}."
        )

    if percentage >= 60:

        return (
            f"Review these concepts carefully: "
            f"{topics}. Start with basic examples and "
            f"then attempt practice questions."
        )

    return (
        f"Start your revision with: {topics}. "
        f"Review the basics first, then practice "
        f"easy questions before moving to harder ones."
    )


# =========================================================
# 20. STUDY PLANNER
# =========================================================

def generate_study_plan(
    subject,
    topic,
    days,
    hours_per_day
):

    subject = clean_text(
        subject,
        200
    )

    topic = clean_text(
        topic,
        300
    )

    prompt = f"""
Create a practical study plan.

Subject: {subject}
Topic: {topic}
Days: {days}
Hours per day: {hours_per_day}

Requirements:
- Use simple language.
- Divide study into manageable sessions.
- Include revision.
- Include practice questions.
- Include short breaks.
- Make it realistic for a student.
- Give a day-by-day plan.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an educational study planner."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content.strip()

    except Exception:

        logger.exception(
            "Study planner failed"
        )

        return (
            "Unable to generate a study plan right now."
        )


# =========================================================
# 21. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 👤 Student Profile")

    st.write(
        f"**{st.session_state.student_name}**"
    )

    st.write(
        st.session_state.student_email
    )

    st.divider()

    st.markdown("### 🧠 Your Learning Memory")

    st.caption(
        "LearnFlow saves your quiz results and "
        "learning conversations to build "
        "personalized learning recommendations."
    )

    st.divider()

    if tavily_client:

        st.success(
            "🌐 Web Search: Connected"
        )

    else:

        st.warning(
            "🌐 Web Search: Not Connected"
        )

    st.divider()

    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):

        for key, value in defaults.items():
            st.session_state[key] = value

        st.rerun()


# =========================================================
# 22. MAIN HEADER
# =========================================================

st.markdown("""
<div class="hero">

<h1>🧠 LearnFlow AI</h1>

<p>
Your Personal AI Learning Companion
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# 23. DASHBOARD METRICS
# =========================================================

quiz_history = get_quiz_history(
    st.session_state.user_id
)

chat_history = get_chat_history(
    st.session_state.user_id,
    limit=100
)

total_quizzes = len(
    quiz_history
)

total_questions = sum(
    item["total"]
    for item in quiz_history
)

total_correct = sum(
    item["score"]
    for item in quiz_history
)

average_score = (
    (total_correct / total_questions) * 100
    if total_questions
    else 0
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <h2>{total_quizzes}</h2>
            <p>Quizzes</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class="metric-card">
            <h2>{total_questions}</h2>
            <p>Questions</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        f"""
        <div class="metric-card">
            <h2>{average_score:.1f}%</h2>
            <p>Average Score</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        f"""
        <div class="metric-card">
            <h2>{len(chat_history)}</h2>
            <p>Learning Conversations</p>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# =========================================================
# 24. TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 AI Tutor",
    "📝 AI Quiz",
    "📅 Study Planner",
    "📊 Progress",
    "🧠 My Learning Memory"
])


# =========================================================
# 25. AI TUTOR TAB
# =========================================================

with tab1:

    st.header("🤖 AI Tutor")

    st.write(
        "Ask LearnFlow anything related to your studies."
    )

    tutor_level = st.selectbox(
        "Tutor Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ],
        key="tutor_level"
    )

    # Display existing conversation
    for message in st.session_state.tutor_messages:

        if message["role"] == "user":

            with st.chat_message("user"):
                st.write(
                    message["content"]
                )

        else:

            with st.chat_message("assistant"):
                st.markdown(
                    message["content"]
                )

                sources = message.get(
                    "sources",
                    []
                )

                if sources:

                    st.caption(
                        "🌐 Web verified"
                    )

                    for i, source in enumerate(
                        sources,
                        1
                    ):

                        st.markdown(
                            f"""
                            <div class="source-card">
                                <span class="source-number">
                                    {i}
                                </span>
                                <b>{source["title"]}</b>
                                <br>
                                <small>
                                    {source["domain"]}
                                </small>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.link_button(
                            f"🔗 Open Source {i}",
                            source["url"]
                        )

    question = st.chat_input(
        "Ask your learning question..."
    )

    if question:

        question = clean_text(
            question,
            MAX_QUESTION_LENGTH
        )

        st.session_state.tutor_messages.append({
            "role": "user",
            "content": question
        })

        answer, sources, web_verified = ask_learnflow(
            question,
            tutor_level
        )

        st.session_state.tutor_messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

        save_chat(
            st.session_state.user_id,
            st.session_state.student_name,
            question,
            answer
        )

        st.rerun()


# =========================================================
# 26. AI QUIZ TAB
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
        index=4,
        key="quiz_education_level"
    )

    class_degree = st.text_input(
        "Class / Degree",
        placeholder="Example: BSCS 3rd Semester",
        key="quiz_class_degree"
    )

    st.markdown(
        "### 📚 Quiz Information"
    )

    subject = st.text_input(
        "Subject",
        placeholder="Example: Programming Fundamentals",
        key="quiz_subject_input"
    )

    topic = st.text_input(
        "Topic",
        placeholder="Example: Arrays",
        key="quiz_topic_input"
    )

    student_level = st.selectbox(
        "Student Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ],
        key="quiz_student_level"
    )

    difficulty = st.selectbox(
        "Difficulty",
        [
            "Easy",
            "Medium",
            "Hard"
        ],
        index=0,
        key="quiz_difficulty_input"
    )

    number_of_questions = st.slider(
        "Number of Questions",
        min_value=1,
        max_value=10,
        value=5,
        key="quiz_number_of_questions"
    )

    st.info(
        f"Selected: {difficulty} difficulty • "
        f"{student_level} level • "
        f"{number_of_questions} questions"
    )

    if st.button(
        "🚀 Generate Quiz",
        key="generate_quiz_button",
        type="primary",
        use_container_width=True
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

            # =================================================
            # FIX 1:
            # Fresh quiz gets a NEW widget generation ID.
            # This prevents old radio answers from carrying
            # into the new quiz.
            # =================================================

            st.session_state.quiz_generation_id += 1

            st.session_state.quiz_data = quiz_data

            st.session_state.quiz_submitted = False

            st.session_state.quiz_answers = [
                None
                for _ in quiz_data["questions"]
            ]

            st.session_state.quiz_saved = False

            st.session_state.quiz_subject = subject

            st.session_state.quiz_topic = topic

            st.session_state.quiz_difficulty = difficulty

            st.success(
                "🎉 Quiz generated successfully!"
            )

            st.rerun()

    # =========================================================
    # DISPLAY QUIZ
    # =========================================================

    if st.session_state.quiz_data is not None:

        quiz_data = st.session_state.quiz_data

        questions = quiz_data["questions"]

        generation_id = (
            st.session_state.quiz_generation_id
        )

        st.divider()

        st.subheader(
            f"📖 {st.session_state.quiz_subject}"
        )

        st.caption(
            f"Topic: {st.session_state.quiz_topic} | "
            f"Difficulty: {st.session_state.quiz_difficulty}"
        )

        for i, q in enumerate(questions):

            st.markdown(
                f"### Question {i + 1}"
            )

            st.markdown(
                q["question"]
            )

            # =================================================
            # FIX 2:
            # Show ACTUAL option text instead of only A/B/C/D.
            # =================================================

            option_labels = [
                f"A. {q['options'][0]}",
                f"B. {q['options'][1]}",
                f"C. {q['options'][2]}",
                f"D. {q['options'][3]}"
            ]

            selected = st.radio(
                "Select your answer:",
                option_labels,
                index=None,

                # =================================================
                # FIX 3:
                # Unique key for every generated quiz.
                # =================================================

                key=(
                    f"quiz_answer_"
                    f"{generation_id}_"
                    f"{i}"
                )
            )

            if selected is not None:

                selected_index = (
                    option_labels.index(selected)
                )

                st.session_state.quiz_answers[i] = (
                    selected_index
                )

            st.write("")

        # =====================================================
        # SUBMIT QUIZ
        # =====================================================

        if not st.session_state.quiz_submitted:

            if st.button(
                "✅ Submit Quiz",
                key="submit_quiz_button",
                type="primary",
                use_container_width=True
            ):

                # =============================================
                # FIX 4:
                # Do not allow incomplete quiz submission.
                # =============================================

                unanswered = [
                    i + 1
                    for i, answer in enumerate(
                        st.session_state.quiz_answers
                    )
                    if answer is None
                ]

                if unanswered:

                    st.warning(
                        "⚠️ Please answer all questions "
                        "before submitting.\n\n"
                        "Unanswered questions: "
                        + ", ".join(
                            map(
                                str,
                                unanswered
                            )
                        )
                    )

                else:

                    st.session_state.quiz_submitted = True

                    st.rerun()

        # =====================================================
        # QUIZ RESULT
        # =====================================================

        if st.session_state.quiz_submitted:

            (
                score,
                total,
                percentage,
                wrong_topics,
                concept_stats,
                feedback
            ) = evaluate_quiz(
                quiz_data,
                st.session_state.quiz_answers
            )

            if not st.session_state.quiz_saved:

                save_quiz_result(
                    st.session_state.user_id,
                    st.session_state.quiz_subject,
                    st.session_state.quiz_topic,
                    st.session_state.quiz_difficulty,
                    score,
                    total,
                    percentage,
                    wrong_topics,
                    concept_stats,
                    feedback
                )

                st.session_state.quiz_saved = True

            st.divider()

            st.subheader(
                "🎯 Quiz Result"
            )

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:

                st.metric(
                    "Score",
                    f"{score}/{total}"
                )

            with result_col2:

                st.metric(
                    "Percentage",
                    f"{percentage:.1f}%"
                )

            with result_col3:

                if percentage >= 75:
                    status = "Good"
                elif percentage >= 60:
                    status = "Needs Practice"
                else:
                    status = "Needs Revision"

                st.metric(
                    "Status",
                    status
                )

            st.info(
                feedback
            )

            # =================================================
            # CONCEPT PERFORMANCE
            # =================================================

            st.subheader(
                "🧠 Concept Performance"
            )

            for concept, values in concept_stats.items():

                concept_total = values["total"]

                concept_correct = values["correct"]

                concept_percentage = (
                    concept_correct
                    / concept_total
                    * 100
                    if concept_total
                    else 0
                )

                st.write(
                    f"**{concept}** — "
                    f"{concept_correct}/{concept_total} "
                    f"({concept_percentage:.1f}%)"
                )

                st.progress(
                    min(
                        concept_percentage / 100,
                        1.0
                    )
                )

            # =================================================
            # WEAK TOPICS
            # =================================================

            current_weak_topics = []

            for concept, values in concept_stats.items():

                if values["total"] <= 0:
                    continue

                concept_percentage = (
                    values["correct"]
                    / values["total"]
                    * 100
                )

                if concept_percentage < 70:

                    current_weak_topics.append({
                        "concept": concept,
                        "percentage": concept_percentage
                    })

            st.subheader(
                "📌 Recommended Improvement"
            )

            recommendation = (
                recommendation_for_score(
                    percentage,
                    current_weak_topics
                )
            )

            st.success(
                recommendation
            )

            if current_weak_topics:

                st.markdown(
                    "### Weak Concepts"
                )

                for item in current_weak_topics:

                    st.markdown(
                        f"""
                        <div class="weak-topic">
                            <b>{item["concept"]}</b><br>
                            Performance:
                            {item["percentage"]:.1f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # =================================================
            # EXPLANATIONS
            # =================================================

            st.subheader(
                "📘 Question Review"
            )

            for i, q in enumerate(questions):

                user_answer = (
                    st.session_state.quiz_answers[i]
                )

                correct_answer = q["answer"]

                if user_answer == correct_answer:

                    st.success(
                        f"Question {i + 1}: Correct"
                    )

                else:

                    st.error(
                        f"Question {i + 1}: "
                        f"Incorrect"
                    )

                    correct_letter = chr(
                        65 + correct_answer
                    )

                    st.write(
                        "Correct answer: "
                        + f"{correct_letter}. "
                        + q["options"][correct_answer]
                    )

                st.write(
                    q["explanation"]
                )

                st.divider()

            # =================================================
            # NEW QUIZ
            # =================================================

            if st.button(
                "🔄 Create New Quiz",
                key="new_quiz_button",
                use_container_width=True
            ):

                st.session_state.quiz_data = None
                st.session_state.quiz_submitted = False
                st.session_state.quiz_answers = []
                st.session_state.quiz_saved = False

                st.rerun()


# =========================================================
# 27. STUDY PLANNER
# =========================================================

with tab3:

    st.header("📅 AI Study Planner")

    st.write(
        "Create a personalized study plan for your topic."
    )

    plan_subject = st.text_input(
        "Subject",
        placeholder="Example: Calculus",
        key="plan_subject"
    )

    plan_topic = st.text_input(
        "Topic",
        placeholder="Example: Derivatives",
        key="plan_topic"
    )

    plan_days = st.number_input(
        "Number of Days",
        min_value=1,
        max_value=30,
        value=7,
        key="plan_days"
    )

    plan_hours = st.number_input(
        "Hours Per Day",
        min_value=0.5,
        max_value=12.0,
        value=2.0,
        step=0.5,
        key="plan_hours"
    )

    if st.button(
        "📅 Generate Study Plan",
        type="primary",
        use_container_width=True
    ):

        if not plan_subject or not plan_topic:

            st.warning(
                "Please enter subject and topic."
            )

        else:

            with st.spinner(
                "Creating your study plan..."
            ):

                plan = generate_study_plan(
                    plan_subject,
                    plan_topic,
                    plan_days,
                    plan_hours
                )

            st.markdown(
                "### 📚 Your Study Plan"
            )

            st.markdown(
                plan
            )

            save_study_plan(
                st.session_state.user_id,
                plan_subject,
                plan_topic,
                plan
            )

            st.success(
                "Study plan saved to your learning memory."
            )


# =========================================================
# 28. PROGRESS
# =========================================================

with tab4:

    st.header("📊 Your Progress")

    history = get_quiz_history(
        st.session_state.user_id
    )

    if not history:

        st.info(
            "Complete your first quiz to see "
            "your progress here."
        )

    else:

        st.subheader(
            "📈 Quiz History"
        )

        for item in history:

            percentage = float(
                item["percentage"]
            )

            st.markdown(
                f"""
                <div class="info-card">

                <b>{item["subject"]}</b>

                <br>

                Topic: {item["topic"]}

                <br>

                Difficulty: {item["difficulty"]}

                <br>

                Score: {item["score"]}/{item["total"]}

                <br>

                Percentage: {percentage:.1f}%

                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                min(
                    percentage / 100,
                    1.0
                )
            )

        st.divider()

        weak_topics = get_weak_topics(
            history
        )

        st.subheader(
            "📌 Learning Weak Areas"
        )

        if weak_topics:

            for item in weak_topics:

                st.markdown(
                    f"""
                    <div class="weak-topic">
                        <b>{item["concept"]}</b><br>
                        Average performance:
                        {item["percentage"]:.1f}%
                        <br>
                        Correct:
                        {item["correct"]}/{item["total"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.success(
                "🎉 No major weak concepts detected!"
            )


# =========================================================
# 29. LEARNING MEMORY
# =========================================================

with tab5:

    st.header("🧠 My Learning Memory")

    st.write(
        "LearnFlow stores your learning conversations "
        "and quiz performance for personalization."
    )

    memory_tab1, memory_tab2 = st.tabs([
        "💬 Tutor Conversations",
        "📝 Quiz Records"
    ])

    with memory_tab1:

        conversations = get_chat_history(
            st.session_state.user_id,
            limit=50
        )

        if not conversations:

            st.info(
                "No tutor conversations saved yet."
            )

        else:

            for item in conversations:

                with st.expander(
                    item["question"][:100]
                ):

                    st.markdown(
                        "**Student:**"
                    )

                    st.write(
                        item["question"]
                    )

                    st.markdown(
                        "**LearnFlow AI:**"
                    )

                    st.write(
                        item["answer"]
                    )

                    st.caption(
                        item["created_at"]
                    )

    with memory_tab2:

        quiz_records = get_quiz_history(
            st.session_state.user_id
        )

        if not quiz_records:

            st.info(
                "No quiz records saved yet."
            )

        else:

            for item in quiz_records:

                with st.expander(
                    f"{item['subject']} — "
                    f"{item['topic']} — "
                    f"{item['percentage']:.1f}%"
                ):

                    st.write(
                        f"**Score:** "
                        f"{item['score']}/{item['total']}"
                    )

                    st.write(
                        f"**Difficulty:** "
                        f"{item['difficulty']}"
                    )

                    st.write(
                        f"**Percentage:** "
                        f"{item['percentage']:.1f}%"
                    )

                    if item.get("feedback"):

                        st.write(
                            f"**Feedback:** "
                            f"{item['feedback']}"
                        )

                    if item.get("wrong_topics"):

                        try:

                            wrong = json.loads(
                                item["wrong_topics"]
                            )

                            if wrong:

                                st.write(
                                    "**Weak concepts in "
                                    "this quiz:**"
                                )

                                for concept in sorted(
                                    set(wrong)
                                ):

                                    st.write(
                                        f"- {concept}"
                                    )

                        except Exception:
                            pass


# =========================================================
# 30. FOOTER
# =========================================================

st.divider()

st.caption(
    "🧠 LearnFlow AI — Personal AI Learning Companion"
)
