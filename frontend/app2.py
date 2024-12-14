import streamlit as st
import mysql.connector
from mysql.connector import Error
import bcrypt
from dotenv import load_dotenv
import os
import requests
import threading
import time

load_dotenv()
API_URL = f"http://{os.getenv('BACKEND_HOST')}:8000/llm_workflow"

st.set_page_config(page_title="SQL Query Generator", page_icon="🔒", layout="wide")

st.markdown(
    """
<style>
    /* Base */
    .main { padding: 2rem; max-width: 800px; margin: 0 auto; }
    .big-font { font-size: clamp(1.5rem, 4vw, 2.5rem); font-weight: 700; color: #1e293b; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: white;
        padding: 1rem;
        border-radius: 1rem 1rem 0 0;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        font-weight: 600;
        color: #64748b;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        color: #3b82f6;
        border-bottom: 2px solid #3b82f6;
    }
    
    /* Form */
    .auth-container {
        background: white;
        padding: 2rem;
        border-radius: 0 0 1rem 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTextInput > div > div {
        background: #f8fafc;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 2px solid #e2e8f0;
        transition: all 0.2s;
    }
    .stTextInput > div > div:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: #3b82f6;
        color: white;
        width: 100%;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: none;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #2563eb;
        transform: translateY(-1px);
    }
    
    /* Chat */
    .chat-container {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Session Controls */
    .session-controls {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main { padding: 1rem; }
        .auth-container { padding: 1rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_session_state():
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "1"
    if "initial_question" not in st.session_state:
        st.session_state.initial_question = 1
    if "snapshot_values" not in st.session_state:
        st.session_state.snapshot_values = None
    if "user" not in st.session_state:
        st.session_state.user = None
    if "loading" not in st.session_state:
        st.session_state.loading = False
    if "session_count" not in st.session_state:
        st.session_state.session_count = 1
    if "is_end" not in st.session_state:
        st.session_state.is_end = 0


def reset_session():
    st.session_state.conversation_history = []
    st.session_state.thread_id = str(st.session_state.session_count + 1)
    st.session_state.initial_question = 1
    st.session_state.snapshot_values = None
    st.session_state.session_count += 1


def authenticate_user(username, password):
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM connect_user WHERE username = %s", (username,))
        user = cursor.fetchone()
        if user and bcrypt.checkpw(
            password.encode("utf-8"), user["password"].encode("utf-8")
        ):
            return user
        return None
    except Error as e:
        st.error(f"데이터베이스 오류: {e}")
        return None
    finally:
        if "cursor" in locals():
            cursor.close()
        if "conn" in locals():
            conn.close()


def check_department_access(department, question):
    department_keywords = {
        "accounting": ["환불", "거래", "장부", "비용", "회계", "ACC", "acc"],
        "cs": ["만족도", "처리", "요청", "고객서비스", "CS", "cs"],
        "common": ["상품", "주문", "고객", "common"],
    }
    question_lower = question.lower()
    for dept, keywords in department_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            if dept != department and department != "common":
                return False
    return True


def register_user(username, password, department, role):
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
        )
        cursor = conn.cursor()
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        cursor.execute(
            "INSERT INTO connect_user (username, password, department, role) VALUES (%s, %s, %s, %s)",
            (username, hashed, department, role),
        )
        conn.commit()
        return True
    except Error as e:
        st.error(f"등록 오류: {e}")
        return False
    finally:
        if "cursor" in locals():
            cursor.close()
        if "conn" in locals():
            conn.close()


def process_chat(prompt, llm_api):
    st.chat_message("user").write(prompt)
    st.session_state.conversation_history.append({"role": "user", "content": prompt})

    # if not check_department_access(st.session_state.user["department"], prompt):
    #     error_message = (
    #         "접근 권한이 없습니다. 해당 부서 관련 데이터만 조회할 수 있습니다."
    #     )
    #     st.error(error_message)
    #     st.chat_message("assistant").write(error_message)
    #     st.session_state.conversation_history.append(
    #         {"role": "assistant", "content": error_message}
    #     )
    #     return

    try:
        response = requests.post(
            API_URL,
            json={
                "user_question": prompt,
                "initial_question": st.session_state.initial_question,
                "thread_id": st.session_state.thread_id,
                "last_snapshot_values": st.session_state.snapshot_values,
                "llm_api": llm_api,
            },
        )

        if response.status_code == 200:
            processed_info = response.json()
            st.session_state.snapshot_values = processed_info
            ask_user = processed_info.get("ask_user", 0)
            st.session_state.initial_question = 0

            output = (
                processed_info["final_answer"]
                if ask_user == 0
                else processed_info["collected_questions"][-1]
            )
            st.session_state.initial_question = 1 if ask_user == 0 else 0

            st.chat_message("assistant").write(output)
            if ask_user == 0:
                st.session_state.is_end = 1
            st.session_state.conversation_history.append(
                {"role": "assistant", "content": output}
            )
        else:
            error_message = "서버 처리 중 오류가 발생했습니다."
            st.error(error_message)
            st.chat_message("assistant").write(error_message)
            st.session_state.conversation_history.append(
                {"role": "assistant", "content": error_message}
            )
    except requests.exceptions.RequestException as e:
        error_message = f"서버 연결 오류: {str(e)}"
        st.error(error_message)
        st.chat_message("assistant").write(error_message)
        st.session_state.conversation_history.append(
            {"role": "assistant", "content": error_message}
        )


# 백엔드 서버에 유저 피드백 전달
def send_feedback(feedback):
    requests.post(
        f"http://{os.getenv('BACKEND_HOST')}:8000/user_feedback",
        json={
            "user_feedback": feedback,
            "thread_id": st.session_state.thread_id,
        },
    )
    st.session_state.is_end = 0


def main():
    initialize_session_state()

    st.markdown('<p class="big-font">AI SQL 쿼리 생성기</p>', unsafe_allow_html=True)
    st.write(
        "이 시스템은 사용자 질문을 분석하여 SQL 쿼리로 변환하고, 그 결과를 자연어로 제공합니다."
    )
    st.markdown("* 자연어 질문 분석 및 SQL 쿼리 생성")
    st.markdown("* 쿼리 실행 및 결과 제공")
    st.markdown("* 사용자 권한 기반 데이터 접근 제어")

    if not st.session_state.user:
        tab1, tab2 = st.tabs(["로그인", "회원가입"])

        with tab1:
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            st.subheader("로그인")
            username = st.text_input("사용자명", key="login_user")
            password = st.text_input("비밀번호", type="password", key="login_pass")
            if st.button("로그인", use_container_width=True):
                if user := authenticate_user(username, password):
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("로그인 실패")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="auth-container">', unsafe_allow_html=True)
            st.subheader("회원가입")
            new_username = st.text_input("사용자명", key="reg_user")
            new_password = st.text_input("비밀번호", type="password", key="reg_pass")
            department = st.selectbox("부서", ["accounting", "cs", "common"])
            role = st.selectbox("역할", ["User", "Admin"])
            if st.button("회원가입", use_container_width=True):
                if register_user(new_username, new_password, department, role):
                    st.success("회원가입이 완료되었습니다")
                else:
                    st.error("회원가입 실패")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        # 사이드바에 사용자 정보와 컨트롤
        st.sidebar.markdown(
            f"""
            <div style='background: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                <h3 style='margin: 0; color: #1e293b;'>👤 {st.session_state.user['username']}</h3>
                <p style='margin: 0.5rem 0; color: #64748b;'>부서: {st.session_state.user['department']}</p>
                <p style='margin: 0; color: #64748b;'>세션 ID: {st.session_state.thread_id}</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("새 세션", use_container_width=True):
                reset_session()
                st.rerun()
        with col2:
            if st.button("로그아웃", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        with st.sidebar:
            llm_api = st.radio(
                "Which LLM would like you use?",
                ["Local", "ChatGPT-4o"],
                captions=[
                    "unsloth/qwen2.5-34B-Coder-bnb-4bit",
                    "OpenAI",
                ],
            )

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.conversation_history:
            st.chat_message(message["role"]).write(message["content"])
        st.markdown("</div>", unsafe_allow_html=True)

        if prompt := st.chat_input("데이터에 대해 질문하세요"):
            process_chat(prompt, llm_api)

        # 대화의 한 사이클이 끝났을 때
        if st.session_state.is_end:
            # 좋아요&싫어요 선택
            feedback = st.feedback("thumbs")
            if feedback is not None:
                # 백엔드 서버에 피드백 전달
                send_feedback(feedback)
                st.rerun()

        if st.session_state.loading:
            with st.spinner("응답을 기다리는 중입니다..."):
                time.sleep(1)


if __name__ == "__main__":
    main()
